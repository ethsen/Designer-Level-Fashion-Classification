import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torchvision.models.segmentation import (
    deeplabv3_resnet101,
    DeepLabV3_ResNet101_Weights,
)


def load_deeplab(device: str = "cuda"):
    """
    Load DeepLabV3-ResNet101 with COCO/VOC weights and its default transforms.
    """
    weights = DeepLabV3_ResNet101_Weights.DEFAULT
    model = deeplabv3_resnet101(weights=weights)
    model.eval()
    model.to(device)
    preprocess = weights.transforms()  # resize + normalize, works on PIL or Tensor
    return model, preprocess


@torch.no_grad()
def get_person_masks_batch(model, preprocess, imgs_uint8: torch.Tensor,
                           device: str = "cuda") -> torch.Tensor:
    """
    imgs_uint8: [B, 3, H, W] uint8, on CPU.
    Returns:
        masks: [B, H, W] float32 in {0,1} on CPU.
    """
    assert imgs_uint8.dim() == 4 and imgs_uint8.size(1) == 3

    # Preprocess each image individually (transforms are not vectorized)
    proc = []
    for i in range(imgs_uint8.size(0)):
        pil = TF.to_pil_image(imgs_uint8[i]) # uint8 -> PIL
        t = preprocess(pil) # [3, H', W'], float, normalized
        proc.append(t)
    inp = torch.stack(proc, dim=0).to(device) # [B,3,H',W']

    out = model(inp)["out"] # [B, 21, H', W']
    preds = out.argmax(1) # [B, H', W']

    PERSON_CLASS = 15  # VOC index for "person"
    person = (preds == PERSON_CLASS).float() # [B, H', W']

    # Resize masks back to original resolution
    B, H_out, W_out = person.shape
    H, W = imgs_uint8.shape[2:]
    person = person.unsqueeze(1)                       # [B,1,H',W']
    person_up = F.interpolate(person, size=(H, W), mode="nearest")
    masks = person_up[:, 0]                            # [B,H,W], float {0,1}

    return masks.cpu()


def process_split(split: str,
                  src_root: Path,
                  dst_root: Path,
                  model,
                  preprocess,
                  device: str,
                  batch_size: int,
                  min_person_ratio: float):
    """
    Process one split (train/val/test): read shards, mask them, and write out.
    """
    src_split = src_root / split
    dst_split = dst_root / split
    dst_split.mkdir(parents=True, exist_ok=True)

    shard_paths = sorted(src_split.glob("*.pt"))
    if not shard_paths:
        print(f"[{split}] No shards found in {src_split}, skipping.")
        return

    print(f"[{split}] Found {len(shard_paths)} shards in {src_split}")

    for shard_path in shard_paths:
        print(f"[{split}] Processing shard {shard_path.name}")
        data = torch.load(shard_path, map_location="cpu")

        imgs = data["images"]          # [N,3,H,W], uint8
        labels = data["labels"]        # [N]
        one_hot = data["one_hot"]      # [N,C]
        N, C, H, W = imgs.shape

        masked_imgs = torch.empty_like(imgs)
        failed = 0

        for start in range(0, N, batch_size):
            end = min(start + batch_size, N)
            batch = imgs[start:end]                    # [B,3,H,W] uint8 on CPU

            masks = get_person_masks_batch(
                model, preprocess, batch, device=device
            )                                          # [B,H,W] float {0,1}

            B = batch.size(0)
            for i in range(B):
                m = masks[i]                           # [H,W]
                person_area = float(m.sum().item()) / (H * W)

                # If mask is basically empty, treat as failure -> keep original
                if person_area < min_person_ratio:
                    masked_imgs[start + i] = batch[i]
                    failed += 1
                else:
                    mask3 = m.unsqueeze(0)            # [1,H,W]
                    fg = (batch[i].float() * mask3).round()
                    fg = fg.clamp_(0, 255).to(torch.uint8)
                    masked_imgs[start + i] = fg

        out = {
            "images": masked_imgs,
            "labels": labels,
            "one_hot": one_hot,
        }

        dst_path = dst_split / shard_path.name
        torch.save(out, dst_path)
        print(
            f"[{split}] Saved masked shard to {dst_path} "
            f"(failed masks: {failed}/{N})"
        )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src_root", type=str, default="torch_data",
                    help="Original shard root (with train/ val/ test/).")
    ap.add_argument("--dst_root", type=str, default="torch_data_masked",
                    help="Output shard root for masked dataset.")
    ap.add_argument("--splits", nargs="+",
                    default=["train", "val"],
                    help="Which splits to process.")
    ap.add_argument("--batch_size", type=int, default=8,
                    help="Segmentation batch size (DeepLab forward).")
    ap.add_argument("--device", type=str, default="cuda",
                    help="'cuda' or 'cpu'. Defaults to cuda if available.")
    ap.add_argument("--min_person_ratio", type=float, default=0.005,
                    help="If person mask covers < this fraction of image, "
                         "fall back to original image.")
    args = ap.parse_args()

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU.")
        device = "cpu"

    src_root = Path(args.src_root)
    dst_root = Path(args.dst_root)
    dst_root.mkdir(parents=True, exist_ok=True)

    # Copy meta.json so ShardDataset sees same metadata
    meta_src = src_root / "meta.json"
    meta_dst = dst_root / "meta.json"
    if meta_src.exists():
        with open(meta_src, "r") as f:
            meta = json.load(f)
        with open(meta_dst, "w") as f:
            json.dump(meta, f, indent=2)
        print(f"Copied meta.json -> {meta_dst}")
    else:
        print("WARNING: meta.json not found in src_root; not copying.")

    print(f"Loading DeepLabV3-ResNet101 on {device}...")
    model, preprocess = load_deeplab(device=device)
    torch.set_grad_enabled(False)

    for split in args.splits:
        process_split(
            split=split,
            src_root=src_root,
            dst_root=dst_root,
            model=model,
            preprocess=preprocess,
            device=device,
            batch_size=args.batch_size,
            min_person_ratio=args.min_person_ratio,
        )

    print("Done.")


if __name__ == "__main__":
    main()
