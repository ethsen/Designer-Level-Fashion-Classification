import random, csv
from pathlib import Path
from datetime import datetime
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
import torchvision.transforms as T
import matplotlib.pyplot as plt
from tqdm import tqdm

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

# ---------------- Dataset that loads full shards ----------------
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


# ---------------- Training helpers ----------------
def evaluate(model, loader, device):
    model.eval()
    ce = nn.CrossEntropyLoss()
    total_loss, total_correct, total = 0.0, 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss = ce(out, y)
            total_loss += float(loss) * x.size(0)
            total_correct += (out.argmax(1) == y).sum().item()
            total += x.size(0)
    return total_loss / total, total_correct / total


# ---------------- Main ----------------
def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", default="resnet18")
    ap.add_argument("--data_root", default="torch_data")
    ap.add_argument("--train_count", type=int, default=45000)
    ap.add_argument("--val_count", type=int, default=5000)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr_head", type=float, default=1e-3)
    ap.add_argument("--lr_backbone", type=float, default=1e-4)
    ap.add_argument("--weight_decay", type=float, default=0.05)
    ap.add_argument("--from_scratch", action="store_true",
                help="Train model from scratch instead of using pretrained weights")
    ap.add_argument("--aug", action="store_true",
                help="Enable base data augmentation transforms")
    ap.add_argument("--out_dir", default="outputs/fast_baseline")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out_dir) / datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    base_train_tf = [T.RandomResizedCrop(224, scale=(0.6, 1.0), ratio=(0.8, 1.25)),
                     T.RandomHorizontalFlip(0.5),
                     T.ColorJitter(0.3, 0.3, 0.3, 0.08),
                     T.RandomApply([T.GaussianBlur(3)], p=0.3),
                     T.RandomErasing(p=0.4, scale=(0.02, 0.2), ratio=(0.3, 3.3), value="random"),]
    base_train_tf = base_train_tf if args.aug else []

    

    from torchvision.models import (
    resnet18, resnet34, resnet50,
    ResNet18_Weights, ResNet34_Weights, ResNet50_Weights,
    )   
    arch = args.arch

    if args.from_scratch:
        print("===== Training from scratch =====")

        train_tf = T.Compose(base_train_tf)
        val_tf   = None  # raw [0,1]
        if arch == "resnet18":
            model = resnet18()
        elif arch == "resnet34":
            model = resnet34()
        elif arch == "resnet50":
            model = resnet50()
        else:
            raise ValueError(f"Unknown arch: {arch}")

    else:
        print("===== Training from pretrained models =====")
        train_tf = T.Compose(base_train_tf + [T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),])
        val_tf = T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)

        if arch == "resnet18":
            model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        elif arch == "resnet34":
            model = resnet34(weights=ResNet34_Weights.IMAGENET1K_V1)
        elif arch == "resnet50":
            model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        else:
            raise ValueError(f"Unknown arch: {arch}")

    # ---- Load shards quickly ----
    train_shards = sorted((Path(args.data_root) / "train").glob("*.pt"))
    val_shards   = sorted((Path(args.data_root) / "val").glob("*.pt"))

    print(f"Loading shards: {len(train_shards)} train, {len(val_shards)} val")
    ds_train_full = ShardDataset(train_shards,transform=train_tf)
    ds_val_full   = ShardDataset(val_shards,transform=val_tf)
    num_classes = ds_train_full.num_classes

    # ---- Random subsampling ----
    rng = random.Random()
    train_idx = rng.sample(range(len(ds_train_full)), min(args.train_count, len(ds_train_full)))
    val_idx   = rng.sample(range(len(ds_val_full)),   min(args.val_count, len(ds_val_full)))
    ds_train = Subset(ds_train_full, train_idx)
    ds_val   = Subset(ds_val_full,   val_idx)

    train_loader = DataLoader(
        ds_train,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=6,
        pin_memory=True
    )
    val_loader   = DataLoader(
        ds_val,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=6,
        pin_memory=True
    )
    
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model.to(device)

    head_params = list(model.fc.parameters())
    bb_params   = [p for n, p in model.named_parameters() if not n.startswith("fc.")]
    opt = torch.optim.AdamW(
        [
            {"params": head_params, "lr": args.lr_head},
            {"params": bb_params,   "lr": args.lr_backbone},
        ],
        weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)

    # ---- Train loop ----
    history = {"epoch": [], "train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_acc, best_path = 0, out_dir / "best_resnet34.pt"

    for epoch in range(1, args.epochs+1):
        model.train()
        tot_loss, tot_correct, tot = 0.0, 0, 0
        for x, y in tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}"):
            x, y = x.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            out = model(x)
            loss = loss_fn(out, y)
            loss.backward()
            opt.step()

            tot_loss += float(loss.item()) * x.size(0)
            tot_correct += (out.argmax(1) == y).sum().item()
            tot += x.size(0)
        scheduler.step()

        train_loss, train_acc = tot_loss / tot, tot_correct / tot
        val_loss, val_acc = evaluate(model, val_loader, device)
        print(f"[{epoch}] train_loss={train_loss:.4f} acc={train_acc:.3f} | "
              f"val_loss={val_loss:.4f} acc={val_acc:.3f}")

        history["epoch"].append(epoch)
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({"model": model.state_dict(), "num_classes": num_classes}, best_path)

    print(f"Best val acc: {best_acc:.3f}")

    # ---- Save README with model + hyperparams ----
    readme_path = out_dir / "readme.txt"
    with open(readme_path, "w") as f:
        f.write("Fashion baseline training run\n")
        f.write("============================\n\n")
        f.write(f"Timestamp: {datetime.now().isoformat()}\n")
        f.write(f"Data root: {args.data_root}\n")
        f.write(f"Model architecture: {model.__class__.__name__}\n")
        f.write(f"Num classes: {num_classes}\n\n")
        f.write("Hyperparameters:\n")
        f.write(f"- train_count: {args.train_count}\n")
        f.write(f"- val_count: {args.val_count}\n")
        f.write(f"- epochs: {args.epochs}\n")
        f.write(f"- batch_size: {args.batch_size}\n")
        f.write(f"- lr_head: {args.lr_head}\n")
        f.write(f"- lr_backbone: {args.lr_backbone}\n")
        f.write(f"- weight_decay: {args.weight_decay}\n")
        f.write(f"- from_scratch flag passed: {args.from_scratch}\n")
        f.write("\nOptimizer & scheduler:\n")
        f.write("- optimizer: AdamW\n")
        f.write("- scheduler: CosineAnnealingLR\n")

        f.write("\nAugmentations:\n")
        f.write(f"- aug flag passed: {args.aug}\n")


    # ---- Save metrics & plots ----
    csv_path = out_dir / "metrics.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch","train_loss","train_acc","val_loss","val_acc"])
        for i in range(len(history["epoch"])):
            writer.writerow([
                history["epoch"][i],
                f"{history['train_loss'][i]:.6f}",
                f"{history['train_acc'][i]:.6f}",
                f"{history['val_loss'][i]:.6f}",
                f"{history['val_acc'][i]:.6f}",
            ])

    epochs = history["epoch"]
    train_loss, val_loss = history["train_loss"], history["val_loss"]
    train_err  = [1 - a for a in history["train_acc"]]
    val_err    = [1 - a for a in history["val_acc"]]

    plt.figure()
    plt.plot(epochs, train_loss, label="train loss", marker="o")
    plt.plot(epochs, val_loss,   label="val loss", marker="o")
    plt.xlabel("epoch"); plt.ylabel("loss")
    plt.legend(); plt.grid(True, alpha=0.3)
    plt.tight_layout(); plt.savefig(out_dir / "loss.png", dpi=150); plt.close()

    plt.figure()
    plt.plot(epochs, train_err, label="train error", marker="o")
    plt.plot(epochs, val_err,   label="val error", marker="o")
    plt.xlabel("epoch"); plt.ylabel("error (1 - acc)")
    plt.legend(); plt.grid(True, alpha=0.3)
    plt.tight_layout(); plt.savefig(out_dir / "error.png", dpi=150); plt.close()

    print(f"\nSaved results to {out_dir}")

if __name__ == "__main__":
    main()
