
# SegmentationGAN — Cityscapes Semantic Segmentation with GANs

This project implements a GAN-based semantic segmentation model using EfficientNet + FPN + ASPP as the generator and a PatchGAN discriminator, trained and evaluated on the Cityscapes image pairs dataset with 5-Fold Cross Validation.

---

## Overview

This repository contains a full pipeline for training a SegmentationGAN model for semantic segmentation:

- Preprocessing: Splits paired Cityscapes images into RGB and label halves  
- Color quantization: Uses `MiniBatchKMeans` to convert raw labels into class indices  
- Dataset loader: Custom PyTorch dataset for dynamic color-based label mapping  
- Model architecture:
  - Generator: `EfficientNet-B0 + FPN + ASPP + Decoder`
  - Discriminator: `PatchGAN` on `(RGB + 1 aux)` channels
- Loss functions:
  - Segmentation: CrossEntropyLoss
  - Adversarial: Hinge loss (`hinge_d_loss`, `hinge_g_loss`)
- Evaluation metrics:
  - Per-class IoU
  - Mean IoU (mIoU)
  - Mean Pixel Accuracy (mPA)
- Cross-validation: 5-fold training and testing pipeline with metrics aggregation

---

## Requirements

Install dependencies before running the notebook:

```bash
pip install torch torchvision timm scikit-learn tqdm pillow matplotlib numpy
```

Optional for Kaggle:
```bash
!pip install -q timm
```

---

## Dataset Structure

Expected directory layout:

```
/kaggle/input/cityscapes-image-pairs/
│
└── cityscapes_data/
    ├── train/
    │   ├── img_00001.png
    │   ├── img_00002.png
    │   └── ...
    └── val/
        ├── img_00001.png
        ├── img_00002.png
        └── ...
```

Each `.png` contains two halves:
- Left: RGB cityscape image  
- Right: semantic label map (color-coded)

---

## Pipeline Summary

### 1. Preprocessing
Splits each image into `(cityscape, label)` halves and collects label colors for clustering:

```python
def split_image(image):
    h, w, _ = image.shape
    mid = w // 2
    return image[:, :mid, :], image[:, mid:, :]
```

Uses `MiniBatchKMeans` to cluster label pixels:

```python
label_model = MiniBatchKMeans(n_clusters=num_classes, random_state=42)
label_model.fit(color_array)
```

---

### 2. Dataset
A PyTorch `Dataset` that dynamically applies the trained KMeans model to generate label maps:

```python
class CityscapeDataset(Dataset):
    ...
    def __getitem__(self, index):
        cityscape, label = split_image(image)
        cls = self.label_model.predict(label.reshape(-1, 3)).reshape(label.shape[:2])
        return self.transform(cityscape), torch.from_numpy(cls).long()
```

---

### 3. Model Components

#### Generator — EfficientNet + FPN + ASPP + Decoder
```python
class FPN_ASPP_Generator(nn.Module):
    def __init__(self, num_classes=10):
        self.backbone = create_model('efficientnet_b0', pretrained=True, features_only=True)
        self.fpn = FPN(self.backbone.feature_info.channels(), 256)
        self.aspp = ASPP(256, 256)
        self.decoder = Decoder(256, 128, num_classes)
```

#### Discriminator — PatchGAN
```python
class PatchDiscriminator(nn.Module):
    def __init__(self, in_channels=4):
        self.model = nn.Sequential(
            nn.Conv2d(in_channels, 64, 4, 2, 1),
            nn.LeakyReLU(0.2, True),
            ...
        )
```

#### Combined GAN
```python
class SegmentationGAN(nn.Module):
    def __init__(self, num_classes=10):
        self.generator = FPN_ASPP_Generator(num_classes)
        self.discriminator = PatchDiscriminator(in_channels=3 + 1)
```

---

### 4. Training

#### Loss Functions
```python
def hinge_d_loss(real_logits, fake_logits):
    return torch.mean(F.relu(1.0 - real_logits)) + torch.mean(F.relu(1.0 + fake_logits))

def hinge_g_loss(fake_logits):
    return -torch.mean(fake_logits)
```

#### GAN Training Loop
Each epoch alternates between:
1. Discriminator step: Real vs fake aux maps  
2. Generator step: Segmentation CE + adversarial hinge loss

```python
for imgs, masks in loader:
    d_loss = d_step(disc, imgs, aux_fake, aux_real, opt_d)
    g_loss, ce, g_adv = g_step(gen, disc, imgs, masks, opt_g, ce_loss, num_classes)
```

---

### 5. Evaluation

Computes confusion matrix and mIoU:

```python
def compute_metrics_from_confusion(conf):
    iou = np.diag(conf) / (conf.sum(1) + conf.sum(0) - np.diag(conf))
    return {'mIoU': np.nanmean(iou)}
```

---

### 6. Cross-Validation

Performs 5-fold K-Fold CV with training and testing on each split:

```python
kf = KFold(n_splits=5, shuffle=True, random_state=42)
for fold, (train_idx, val_idx) in enumerate(kf.split(indices)):
    ...
    test_metrics, _ = evaluate_model_on_loader(gan, test_loader, device, num_classes)
```

---

## Results Summary

At the end, prints model parameter count and mean metrics across folds:

```
Generator parameters: 12,345,678
Discriminator parameters: 4,567,890
Average mIoU over 5 folds: 0.7123 ± 0.0345
Average mPA  over 5 folds: 0.8532 ± 0.0278
```

---

## Future Improvements

- Replace KMeans-based pseudo-labels with real semantic masks  
- Add mixed precision (AMP) for faster training  
- Integrate visualization for predicted vs ground-truth segmentation  
- Experiment with other backbones (`ConvNeXt`, `Swin Transformer`)  

---

## References

- [Cityscapes Dataset](https://www.cityscapes-dataset.com/)
- [EfficientNet (Tan & Le, 2019)](https://arxiv.org/abs/1905.11946)
- [Feature Pyramid Networks](https://arxiv.org/abs/1612.03144)
- [DeepLabV3+](https://arxiv.org/abs/1802.02611)
- [Pix2Pix / PatchGAN](https://arxiv.org/abs/1611.07004)

---

**Author:** Your Name  
**License:** MIT  
**Frameworks:** PyTorch, TIMM, scikit-learn
