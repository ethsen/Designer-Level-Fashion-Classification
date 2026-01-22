# Designer-Level Fashion Classification

This project investigates how **model capacity** and **training strategy** influence both
**classification performance** and **interpretability** in designer-level fashion recognition.
Using the iDesigner FGVC dataset, we study whether models learn meaningful garment-level
features or rely on contextual shortcuts such as runway background, lighting, and venue cues.

We compare standard ResNet architectures (ResNet-18/34/50) against custom lightweight models
(FashionNet-Small and FashionNet-Medium) under multiple training regimes, including unmasked
images, background-subtracted images, and heavy data augmentation. In addition to reporting
classification accuracy, we analyze learned representations using Grad-CAM, occlusion-style
analysis, and nearest-neighbor retrieval in feature space.


## Repository Structure

- `resnet-train.py`  
  Training script for ResNet baselines (ResNet-18/34/50), supporting training from scratch
  or ImageNet-pretrained initialization, with optional heavy data augmentation.

- `fashionnet-train.py`  
  Training script for the custom FashionNet architectures (FashionNet-Small and
  FashionNet-Medium).

- `gradcam-viz.py`  
  Script for generating Grad-CAM visualizations for trained ResNet models. Saves separate
  input images and Grad-CAM overlays for qualitative analysis.

- `occlusion_analysis.py`  
  Script for occlusion sensitivity analysis. Supports both ResNet and FashionNet models
  with a small change to the model-loading block.

- `feature_analysis_lite.py`  
  Script for nearest-neighbor retrieval in feature space using `model.forward_features`.
  Designed primarily for FashionNet models.

- `build-segmented-dataset.py`  
  Preprocessing script that constructs a background-subtracted (masked) version of the
  dataset using DeepLabV3 person segmentation.

- `torch_data/`  
  Directory containing the original (unmasked) shard-based dataset:
  - `train/`
  - `val/`.

- `models/`  
  Directory containing pretrained model checkpoints used for analysis (e.g.,
  best ResNet-50 and FashionNet-Medium weights).

- `outputs/`  
  Automatically created directory storing training runs, checkpoints, metrics, plots,
  and per-run metadata.

## Dataset Directory Layout

The dataset is stored as PyTorch shards with the following structure:

- `torch_data/`  
  Directory containing preprocessed dataset shards:
  ```text
  torch_data/
  ├── train/
  │   ├── shard_003.pt
  │   └── ...
  └── val/
      ├── shard_000.pt
      └── ...

We only include a small portion of our data because of size issues. Each shard contains images, labels, and one-hot class encodings.

## Dataset Format

Each shard (`.pt` file) contains:
- `images`: uint8 tensor of shape `[N, 3, H, W]`
- `labels`: integer tensor of shape `[N]`
- `one_hot`: one-hot encoded labels of shape `[N, C]`

Images are converted to floating point in `[0, 1]` during loading. ImageNet normalization is
applied only when using pretrained ResNet models.

## Included Checkpoints (Pretrained ResNet-50 + Masked FashionNet-Medium)

This repo includes two trained checkpoints:

1) **ResNet-50 (ImageNet-pretrained) trained on the *unmasked* dataset**  
   - Intended primarily for **Grad-CAM** analysis (`gradcam-viz.py`).  
   - If you keep the default paths unchanged, the provided checkpoint should run as-is.  
   - To test a different ResNet checkpoint, you only need to update `--ckpt_path` (and optionally `--arch` / `--data_root`).

2) **FashionNet-Medium trained on the *masked* dataset**  
   - Intended primarily for **occlusion sensitivity** (`occlusion_analysis.py`) and **feature-space nearest neighbors** (`feature_analysis_lite.py`).  
   - The occlusion script also supports ResNet checkpoints, but switching between ResNet vs. FashionNet requires a small edit in the model construction + checkpoint load block.



---

## Training ResNet Models (`resnet-train.py`)

The ResNet training script supports:
- ResNet-18, ResNet-34, and ResNet-50
- Training from scratch or with ImageNet pretraining
- Optional heavy data augmentation
- Separate learning rates for backbone and classifier head
- Automatic logging, checkpointing, and plotting

### Example: ResNet from Scratch (No Augmentation)

```bash
python resnet-train.py \
  --arch resnet18 \
  --data_root torch_data \
  --from_scratch \
  --epochs 20 \
  --batch_size 64


```

### Example: Pretrained ResNet with Augmentation

```bash
python resnet-train.py \
  --arch resnet50 \
  --data_root torch_data \
  --epochs 40 \
  --batch_size 64 \
  --aug
```

## Training FashionNet Models (`fashionnet-train.py`)

The `fashionnet-train.py` script trains the custom FashionNet architectures used in this
project:
- `FashionNetSmall` (`--arch fn-small`)
- `FashionNetMedium` (`--arch fn-medium`)

FashionNet models are trained from scratch on the same shard-based dataset format using
cross-entropy loss with label smoothing and a cosine learning-rate schedule.

### Example: Train FashionNet-Small

```bash
python fashionnet-train.py \
  --arch fn-small \
  --data_root torch_data \
  --epochs 20 \
  --batch_size 64
```
### Example: Train FashionNet-Medium with Augmentation

```bash
python fashionnet-train.py \
  --arch fn-medium \
  --data_root torch_data \
  --epochs 20 \
  --batch_size 64 \
  --aug
  ```


## Outputs and Logging

Each training run creates a timestamped directory under `outputs/` containing:
- Best model checkpoint (`best_*.pt`)
- `metrics.csv` with per-epoch losses and accuracy
- `loss.png` and `error.png` plots
- `readme.txt` summarizing the run configuration

These artifacts allow experiments to be fully inspected and reproduced.

---
## Model Analysis

After training, models are analyzed using:
- Grad-CAM visualizations to diagnose spatial attention and background reliance
- Occlusion-style analysis to probe feature sensitivity
- Nearest-neighbor retrieval in feature space to assess representation quality

These analyses are essential for understanding *how* models make predictions, not just
how accurate they are.


## Occlusion Sensitivity Analysis

For a given image, we slide a fixed-size occlusion window across the image and replace each
patch with a constant value. The drop in the model’s confidence for its predicted class is
recorded and aggregated into a heatmap. Regions where occlusion causes a large confidence
drop are interpreted as being more important to the model’s decision.

## How Occlusion Sensitivity Is Computed

The occlusion pipeline proceeds as follows:

1. A trained classification model is loaded (ResNet or FashionNet).
2. A validation image is passed through the model to obtain the predicted class and its
   corresponding logit.
3. A sliding occlusion window (e.g., 32×32 pixels) is applied across the image with a fixed
   stride.
4. For each occluded region, the model is re-evaluated and the drop in the predicted class
   logit is recorded.
5. The logit drops are accumulated and averaged across overlapping patches to produce a
   per-pixel importance map.
6. The resulting heatmap is normalized to [0,1] and overlaid on the original image.

Higher values in the heatmap indicate regions where occlusion most strongly degrades the
model’s confidence, suggesting greater importance to the prediction.

## Running Occlusion Analysis

The occlusion experiment can be run using the provided script. It reconstructs the validation
dataset, loads a trained model checkpoint, and generates occlusion visualizations for a
subset of images.

### Example Usage

```bash
python occlusion_analysis.py
```

## Switching Models in `occlusion_analysis.py` (ResNet ↔ FashionNet)

In the occlusion script, the model is defined and loaded in the “Load trained …” block.  
To run **ResNet-50**, use the ResNet block:

```python
# --- ResNet-50 ---
model = resnet50()
model.fc = nn.Linear(model.fc.in_features, num_classes)
model.to(device)

ckpt_path = Path("models/best_resnet50.pt")  # adjust
model.load_state_dict(torch.load(ckpt_path, map_location="cpu")["model"])
model.to(device)

# --- FashionNet-Medium ---
model = FashionNetMedium(num_classes=num_classes)
model.to(device)

ckpt_path = Path("models/best_fashionnet_medium.pt")  # adjust
model.load_state_dict(torch.load(ckpt_path, map_location="cpu")["model"])
model.to(device)
```


## Grad-CAM Visualization

This script generates Grad-CAM visualizations for trained ResNet models
(ResNet-18, ResNet-34, ResNet-50) using the same shard-based dataset format
as the training pipeline.

Grad-CAM is computed on a per-image basis for the predicted class using
the final convolutional block of the network. For each image, the script
saves:
- the original input image
- the Grad-CAM heatmap overlaid on the image

The script is primarily intended for analyzing ResNet models, which were
the strongest-performing architectures in this project.

## Running Grad-CAM Visualization

Grad-CAM visualizations are generated by running the script with the
appropriate model architecture and checkpoint.

### Basic Example (ResNet-50)

```bash
python gradcam_viz.py \
  --data_root torch_data \
  --split val \
  --arch resnet50 \
  --ckpt_path outputs/20251205_171110_rn50_pretrained/best_resnet50.pt \
  --out_dir gradcam_outputs \
  --num_images 16
```

## Implementation Notes

- Grad-CAM is computed using the final residual block (`layer4[-1]`)
  for all ResNet variants.
- The predicted class is used as the default Grad-CAM target.
- Hooks are registered and removed automatically during execution.
- Images are denormalized only for visualization purposes.


## Feature-Space Nearest Neighbor Retrieval (FashionNet)

This script performs nearest-neighbor retrieval in feature space using a trained
FashionNet model (Small or Medium). It:

1) Loads shard-based train/val datasets.
2) Extracts feature embeddings for the entire training set using `model.forward_features`.
3) Samples query images from the validation set.
4) Retrieves top-k nearest neighbors from the training set using cosine similarity.
5) Saves 1x(k+1) grids: [query | neighbors] to disk.

Outputs are saved as: `nn_example_00.png`, `nn_example_01.png`, ...

## Running the Script

Example:

```bash
python feature_analysis_lite.py
```
### Feature-Space Nearest Neighbor Retrieval

This repo includes `feature_analysis_lite.py` for nearest-neighbor retrieval in embedding space.

It loads a trained FashionNet checkpoint, extracts embeddings for the training set via
`model.forward_features`, and retrieves top-k cosine nearest neighbors for random validation queries.

Outputs are saved as image grids to the directory specified by `out_dir` near the bottom of the script.


## Creating the Masked Dataset (Background Removal)

This project includes a preprocessing script (build-segmented-dataset.py) that builds a *masked* version of the shard dataset by removing background pixels. The pipeline:

1) Loads a pretrained DeepLabV3-ResNet101 semantic segmentation model.
2) Runs the model on each image to predict per-pixel class labels.
3) Extracts the binary **person** mask (VOC class index 15).
4) Upsamples the mask to the original image resolution.
5) Applies the mask to the original uint8 image, zeroing out all non-person pixels.
6) If the predicted person mask is too small (below a threshold), the script falls back to the original image.
7) Writes new shards with the same `{images, labels, one_hot}` structure to a new dataset root (e.g., `torch_data_masked/`).

The output dataset preserves the exact shard format, so the same training and evaluation scripts can be run by pointing `--data_root` to the masked dataset directory.

### Running the Masking Script

Example (train + val):

```bash
python build-masked-dataset.py \
  --src_root torch_data \
  --dst_root torch_data_masked \
  --splits train val \
  --batch_size 8 \
  --device cuda \
  --min_person_ratio 0.005
