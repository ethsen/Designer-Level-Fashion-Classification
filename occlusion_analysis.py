import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, Dataset
import torch.nn as nn
import matplotlib.pyplot as plt
from pathlib import Path
import random
import json
from fashionnet import FashionNetSmall, FashionNetMedium
from torchvision.models import (resnet18, resnet34, resnet50)


class ShardDataset(Dataset):
    def __init__(self, shard_paths, transform=None):
        self.shards = shard_paths
        self.samples = []  # (file_idx, local_idx)
        self.data = []
        self.transform = transform

        # Load all shards fully into memory
        for fi, f in enumerate(self.shards):
            d = torch.load(f, map_location="cpu")
            self.data.append(d)
            n = d["labels"].shape[0]
            self.samples.extend([(fi, i) for i in range(n)])

        self.num_classes = self.data[0]["one_hot"].shape[1]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        fi, ii = self.samples[idx]
        shard = self.data[fi]

        # [3,H,W] uint8 -> float in [0,1]
        x = shard["images"][ii].float() / 255.0
        y = shard["labels"][ii].long()

        if self.transform is not None:
            x = self.transform(x)  # x is a CHW float tensor

        return x, y


# ---------- Normalization helpers (ImageNet stats, same as training) ----------
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

def unnormalize(img: torch.Tensor) -> torch.Tensor:
    """
    img: [3, H, W] normalized with ImageNet mean/std.
    returns [3, H, W] in [0, 1]
    """
    return (img * IMAGENET_STD + IMAGENET_MEAN).clamp(0.0, 1.0)

# ---------- Occlusion sensitivity computation ----------
@torch.no_grad()
def compute_occlusion_map(
    model,
    img: torch.Tensor,          # [3, H, W], normalized
    device,
    patch_size: int = 32,
    stride: int = 16,
    occlusion_mode: str = "zero",  # or "mean"
):
    """
    Compute occlusion sensitivity map for a single image.
    Returns:
      heatmap: [H, W] tensor in [0,1] (higher = more important)
      pred_class: int (argmax class index)
    """
    model.eval()
    H, W = img.shape[1], img.shape[2]

    x = img.unsqueeze(0).to(device)  # [1,3,H,W]

    # Original prediction and logit for predicted class
    logits = model(x)
    pred_class = int(logits.argmax(dim=1)[0])
    base_logit = float(logits[0, pred_class])

    # Prepare occlusion fill value in normalized space
    if occlusion_mode == "zero":
        fill_value = 0.0
    elif occlusion_mode == "mean":
        # mean in normalized space is 0
        fill_value = 0.0
    else:
        raise ValueError(f"Unknown occlusion_mode: {occlusion_mode}")

    # Heatmap accumulator (we'll average over overlapping patches)
    heatmap = torch.zeros(1, 1, H, W, device=device)
    counts  = torch.zeros(1, 1, H, W, device=device) + 1e-8

    # Slide occlusion window
    for i in range(0, H, stride):
        for j in range(0, W, stride):
            i_end = min(i + patch_size, H)
            j_end = min(j + patch_size, W)

            x_occ = x.clone()
            x_occ[:, :, i:i_end, j:j_end] = fill_value

            logits_occ = model(x_occ)
            occ_logit = float(logits_occ[0, pred_class])

            delta = base_logit - occ_logit  # how much confidence drops

            # Accumulate delta over the occluded region
            heatmap[:, :, i:i_end, j:j_end] += delta
            counts[:, :, i:i_end, j:j_end] += 1.0

    # Average impact per pixel and normalize to [0,1]
    heatmap = heatmap / counts
    heatmap = heatmap.squeeze(0).squeeze(0)  # [H, W]

    # Shift/scale to [0,1]
    min_v = float(heatmap.min())
    max_v = float(heatmap.max())
    if max_v > min_v:
        heatmap = (heatmap - min_v) / (max_v - min_v)
    else:
        heatmap = torch.zeros_like(heatmap)

    return heatmap.cpu(), pred_class


def load_idx_to_label(data_root: Path):
    idx_path = data_root / "index_to_label.json"
    with open(idx_path, "r") as f:
        idx_to_label = json.load(f)  # likely str -> str
    return idx_to_label

def plot_occlusion_result(
    img: torch.Tensor,          # [3,H,W], normalized
    heatmap: torch.Tensor,      # [H,W], in [0,1]
    true_label: int,
    pred_label: int,
    idx_to_label,
    out_path: Path,
):
    """
    Save side-by-side visualization: 
    - Original image  
    - Occlusion heatmap overlay + colorbar legend  
    """
    img_vis =img.permute(1, 2, 0).numpy()  # [H,W,3], [0,1]
    heatmap_np = heatmap.numpy()

    plt.figure(figsize=(10, 4))

    # --------------------
    # Original image
    # --------------------
    ax = plt.subplot(1, 2, 1)
    ax.imshow(img_vis)
    gt_name = idx_to_label[str(int(true_label))]
    pred_name = idx_to_label[str(int(pred_label))]
    ax.set_title(f"Image\nGT: {gt_name}\nPred: {pred_name}", fontsize=10)
    ax.axis("off")

    # --------------------
    # Occlusion heatmap + colorbar
    # --------------------
    ax = plt.subplot(1, 2, 2)
    ax.imshow(img_vis)
    im = ax.imshow(heatmap_np, cmap="jet", alpha=0.55)

    ax.set_title("Occlusion Sensitivity", fontsize=10)
    ax.axis("off")

    # Colorbar legend
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Occlusion Impact (Importance)", fontsize=9)
    cbar.ax.tick_params(labelsize=8)

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()



def build_datasets(data_root: Path, train_count: int = None, val_count: int = None, seed: int = 19):
    """
    Rebuild train/val datasets similarly to your training script.
    """

    train_shards = sorted((data_root / "train").glob("*.pt"))
    val_shards   = sorted((data_root / "val").glob("*.pt"))

    ds_train_full = ShardDataset(train_shards)
    ds_val_full   = ShardDataset(val_shards)

    rng = random.Random(seed)

    if train_count is None:
        train_count = len(ds_train_full)
    if val_count is None:
        val_count = len(ds_val_full)

    train_idx = rng.sample(range(len(ds_train_full)), min(train_count, len(ds_train_full)))
    val_idx   = rng.sample(range(len(ds_val_full)),   min(val_count,   len(ds_val_full)))

    ds_train = Subset(ds_train_full, train_idx)
    ds_val   = Subset(ds_val_full,   val_idx)

    return ds_train, ds_val, ds_train_full.num_classes


def run_occlusion_experiment(
    model,
    ds_val,
    data_root: Path,
    device,
    out_dir: Path,
    num_examples: int = 8,
    patch_size: int = 32,
    stride: int = 16,
):
    """
    Runs occlusion sensitivity on a small subset of validation images
    and saves visualizations.
    """
    model.to(device)
    model.eval()

    idx_to_label = load_idx_to_label(data_root)

    N_val = len(ds_val)
    rng = random.Random()
    example_indices = rng.sample(range(N_val), min(num_examples, N_val))

    for k, idx in enumerate(example_indices):
        img, label = ds_val[idx]   # img: [3,H,W], label: int
        img_batch = img.unsqueeze(0).to(device)

        with torch.no_grad():
            logits = model(img_batch)
            pred_label = int(logits.argmax(dim=1)[0])

        # Compute occlusion map
        heatmap, _ = compute_occlusion_map(
            model=model,
            img=img,
            device=device,
            patch_size=patch_size,
            stride=stride,
            occlusion_mode="zero",
        )

        out_path = out_dir / f"occlusion_example_{k:02d}.png"
        plot_occlusion_result(
            img=img,
            heatmap=heatmap,
            true_label=label,
            pred_label=pred_label,
            idx_to_label=idx_to_label,
            out_path=out_path,
        )
        print(f"Saved {out_path}")


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_root = Path("torch_data")  # adjust if different
    out_dir   = Path("occlusion_rn50_pt_masked")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Rebuild datasets
    ds_train, ds_val, num_classes = build_datasets(
        data_root=data_root,
        train_count=None,   # or your original train_count
        val_count=None,     # or your original val_count
        seed=19,
    )

    # Load trained FashionNet-Small

    #model = FashionNetMedium(num_classes=num_classes)
    model = resnet50()
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model.to(device)

    ckpt_path = Path("models/best_resnet50.pt")  # adjust path
    model.load_state_dict(torch.load(ckpt_path, map_location="cpu")['model'])
    model.to(device)

    # Run occlusion experiment on a few val images
    run_occlusion_experiment(
        model=model,
        ds_val=ds_val,
        data_root=data_root,
        device=device,
        out_dir=out_dir,
        num_examples=8,
        patch_size=32,
        stride=16,
    )
