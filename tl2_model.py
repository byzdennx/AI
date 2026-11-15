# ============================================
# tl2_model.py
# ResNet18 pretrained + classifier 2 kelas
# ============================================
import torch
import torch.nn as nn
from torchvision import models


def buat_model(freeze_backbone=True):
    """
    freeze_backbone=True  -> hanya classifier yang dilatih (fase 1)
    freeze_backbone=False -> seluruh model dilatih (fase 2 fine-tuning)
    """
    # Load ResNet18 pretrained ImageNet
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

    # Bekukan semua layer kalau feature extraction
    if freeze_backbone:
        for p in model.parameters():
            p.requires_grad = False

    # Ganti layer fc (aslinya 512->1000) jadi 512->2
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(in_features, 256),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(256, 2)
    )

    # fc baru otomatis trainable (layer baru)
    return model


if __name__ == '__main__':
    m = buat_model(freeze_backbone=True)

    total   = sum(p.numel() for p in m.parameters())
    latih   = sum(p.numel() for p in m.parameters() if p.requires_grad)
    print(f"Total parameter : {total:,}")
    print(f"Trainable       : {latih:,}  ({latih/total*100:.2f}%)")

    # Test forward
    out = m(torch.randn(2, 3, 224, 224))
    print("Output shape    :", tuple(out.shape))   # (2, 2)