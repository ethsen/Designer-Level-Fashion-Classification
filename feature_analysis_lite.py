import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset
import matplotlib.pyplot as plt
import json
from pathlib import Path
import random
from fashionnet import FashionNetSmall, FashionNetMedium

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

# ---- ImageNet normalization used in ShardDataset ----
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)
IMAGENET_STD  = torch.tensor([0.229, 0.224, 0.225]).view(3,1,1)

def unnormalize(img: torch.Tensor) -> torch.Tensor:
    """
    img: [3,H,W] normalized with ImageNet mean/std.
    returns [3,H,W] in [0,1]
    """
    return (img * IMAGENET_STD + IMAGENET_MEAN).clamp(0.0, 1.0)


def extract_features_dataset(model, dataset, device, batch_size=64):
    """
    Runs model.forward_features over the entire dataset.
    Returns:
      feats:  [N, D]
      labels: [N]
    Assumes dataset[i] -> (x, y).
    """
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=6,
        pin_memory=True
    )

    model.eval()
    all_feats = []
    all_labels = []

    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device, non_blocking=True)
            feats = model.forward_features(xb)  # [B,D]
            all_feats.append(feats.cpu())
            all_labels.append(yb.clone())

    feats = torch.cat(all_feats, dim=0)   # [N,D]
    labels = torch.cat(all_labels, dim=0) # [N]

    return feats, labels


def load_idx_to_label(data_root: Path):
    idx_path = "../torch_data/index_to_label.json"
    with open(idx_path, "r") as f:
        idx_to_label = json.load(f)  # likely str -> str
    return idx_to_label


def plot_query_and_neighbors(
    query_img, query_label, query_pred,
    neighbor_imgs, neighbor_labels, neighbor_preds,
    idx_to_label, out_path: Path,
    k: int
):
    """
    Save a figure with 1 row: [query | k neighbors]
    """
    plt.figure(figsize=(3*(k+1), 4))
    # Query on the left
    ax = plt.subplot(1, k+1, 1)
    title_kwargs = dict(fontfamily="serif", fontweight="bold", fontsize=13)

    ax.imshow(query_img.permute(1,2,0).numpy())
    q_lbl = idx_to_label[str(int(query_label))]
    q_pred_lbl = idx_to_label[str(int(query_pred))]
    ax.set_title(f"Query\nGT: {q_lbl}\nPred: {q_pred_lbl}",**title_kwargs)
    ax.axis("off")

    # Neighbors
    for j in range(k):
        ax = plt.subplot(1, k+1, j+2)
        nb_img = neighbor_imgs[j]
        nb_lbl = neighbor_labels[j]
        nb_pred = neighbor_preds[j]

        ax.imshow(nb_img.permute(1,2,0).numpy())
        lbl = idx_to_label[str(int(nb_lbl))]
        pred_lbl = idx_to_label[str(int(nb_pred))]
        ax.set_title(f"NN{j+1}\nGT: {lbl}\nPred: {pred_lbl}",**title_kwargs)
        ax.axis("off")

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()


def run_nn_experiment(
    model,
    ds_train,
    ds_val,
    data_root: Path,
    device,
    out_dir: Path,
    k: int = 5,
    num_queries: int = 8,
    batch_size: int = 64,
    seed: int = 5145,
    enforce_unique_query_labels: bool = True,
    require_different_brand_neighbors: bool = True,
):
    """
    1) Build feature index for training set.
    2) Sample num_queries from validation set (optionally enforcing unique GT labels).
    3) For each query, find top-k nearest neighbors (cosine) in train set.
       Optionally filter neighbors to EXCLUDE the query's GT label (i.e., "not same brand").
    4) Save grids and print stats.

    Notes:
    - If require_different_brand_neighbors=True, we retrieve more than k internally and then
      filter out same-brand neighbors until we have k (or run out).
    - If enforce_unique_query_labels=True, we try to pick queries with distinct GT labels.
    """
    model.to(device)
    model.eval()

    idx_to_label = load_idx_to_label(data_root)

    print("Extracting training features...")
    train_feats, train_labels = extract_features_dataset(
        model, ds_train, device, batch_size=batch_size
    )  # feats [N,D] on CPU, labels [N] on CPU
    train_feats = F.normalize(train_feats, dim=1).to(device)  # [N,D] on GPU
    train_labels_cpu = train_labels.cpu()  # [N]
    print(f"Train features: {train_feats.shape}")

    # -------- choose queries (optionally unique ground-truth labels) --------
    N_val = len(ds_val)
    rng = random.Random(seed)

    if not enforce_unique_query_labels:
        query_indices = rng.sample(range(N_val), min(num_queries, N_val))
    else:
        # Try to choose up to num_queries with distinct GT labels
        all_val_indices = list(range(N_val))
        rng.shuffle(all_val_indices)
        used_labels = set()
        query_indices = []
        for idx in all_val_indices:
            _, yq = ds_val[idx]
            yq = int(yq)
            if yq in used_labels:
                continue
            used_labels.add(yq)
            query_indices.append(idx)
            if len(query_indices) >= num_queries:
                break

        if len(query_indices) < num_queries:
            print(
                f"[warn] Only found {len(query_indices)} unique labels in ds_val for queries "
                f"(requested {num_queries}). Proceeding with {len(query_indices)}."
            )

    same_label_counts = []
    diff_label_counts = []

    # We'll over-retrieve then filter when excluding same-brand neighbors
    overretrieve = max(50, 10 * k)  # adjust if you want

    for qi, idx in enumerate(query_indices):
        # Get query image + label
        xq, yq = ds_val[idx]
        yq_int = int(yq)
        xq_batch = xq.unsqueeze(0).to(device)

        with torch.no_grad():
            fq = model.forward_features(xq_batch)        # [1,D]
            logits_q = model(xq_batch)                   # [1,C]
            pred_q = logits_q.argmax(dim=1).cpu()[0]

        fq = F.normalize(fq, dim=1)                      # [1,D]
        sims = torch.matmul(train_feats, fq.squeeze(0))  # [N]

        # retrieve more than k so filtering doesn't empty us
        kk = min(overretrieve, sims.numel())
        vals, nn_indices = torch.topk(sims, k=kk, largest=True, sorted=True)

        neighbor_imgs = []
        neighbor_labels = []
        neighbor_preds = []

        same = 0
        diff = 0

        # Filter neighbors: if require_different_brand_neighbors, skip same GT label
        for j in nn_indices.tolist():
            y_nb = int(train_labels_cpu[j])
            if require_different_brand_neighbors and (y_nb == yq_int):
                continue

            x_nb, y_nb_tensor = ds_train[j]
            neighbor_imgs.append(x_nb)
            neighbor_labels.append(y_nb_tensor)

            with torch.no_grad():
                logits_nb = model(x_nb.unsqueeze(0).to(device))
                pred_nb = logits_nb.argmax(dim=1).cpu()[0]
            neighbor_preds.append(pred_nb)

            if y_nb == yq_int:
                same += 1
            else:
                diff += 1

            if len(neighbor_imgs) >= k:
                break

        if len(neighbor_imgs) < k:
            print(
                f"[warn] Query {qi:02d}: only found {len(neighbor_imgs)}/{k} neighbors "
                f"after filtering (exclude_same_brand={require_different_brand_neighbors})."
            )

        denom = max(1, len(neighbor_imgs))
        same_label_counts.append(same / denom)
        diff_label_counts.append(diff / denom)

        out_path = out_dir / f"nn_example_{qi:02d}.png"
        plot_query_and_neighbors(
            query_img=xq,
            query_label=yq,
            query_pred=pred_q,
            neighbor_imgs=neighbor_imgs,
            neighbor_labels=neighbor_labels,
            neighbor_preds=neighbor_preds,
            idx_to_label=idx_to_label,
            out_path=out_path,
            k=len(neighbor_imgs),  # plot exactly what we found
        )

        q_lbl = idx_to_label.get(str(yq_int), str(yq_int))
        print(
            f"Saved {out_path} | Query GT={q_lbl} | "
            f"same-label among shown: {same}/{denom} | diff-label among shown: {diff}/{denom}"
        )

    avg_same = sum(same_label_counts) / max(1, len(same_label_counts))
    avg_diff = sum(diff_label_counts) / max(1, len(diff_label_counts))
    print(f"\nAverage fraction same-designer among shown neighbors: {avg_same:.3f}")
    print(f"Average fraction different-designer among shown neighbors: {avg_diff:.3f}")



if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_root = Path("torch_data")           # adjust to your path
    out_dir   = Path("feature_analyis_fnMedium")

    # 1) Rebuild datasets the same way as during training
    train_shards = sorted((data_root / "train").glob("*.pt"))
    val_shards   = sorted((data_root / "val").glob("*.pt"))

    ds_train_full = ShardDataset(train_shards)
    ds_val_full   = ShardDataset(val_shards)

    # If you used subsampling, reuse same counts/seed as training
    rng = random.Random()
    train_count = len(ds_train_full)   # or your previous args.train_count
    val_count   = len(ds_val_full)     # or your previous args.val_count

    train_idx = rng.sample(range(len(ds_train_full)), min(train_count, len(ds_train_full)))
    val_idx   = rng.sample(range(len(ds_val_full)),   min(val_count,   len(ds_val_full)))

    ds_train = Subset(ds_train_full, train_idx)
    ds_val   = Subset(ds_val_full,   val_idx)

    # 2) Load trained FashionNetSmall
    num_classes = ds_train_full.num_classes
    model = FashionNetMedium(num_classes=num_classes)
    ckpt_path = Path("models/best_fashionnet_medium.pt")  # adjust path
    ckpt = torch.load(ckpt_path, map_location="cpu")

    test = torch.load(ckpt_path)

    model.load_state_dict(torch.load(ckpt_path, map_location="cpu")['model'])
    model.to(device)

    # 3) Run NN experiment
    run_nn_experiment(
        model=model,
        ds_train=ds_train,
        ds_val=ds_val,
        data_root=data_root,
        device=device,
        out_dir=out_dir,
        k=3,
        num_queries=8,
        batch_size=64,
    )
