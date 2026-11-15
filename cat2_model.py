# ============================================
# cat2_model.py
# Arsitektur CNN adaptif terhadap IMG_SIZE
# ============================================
import torch
import torch.nn as nn


class CatDogCNN(nn.Module):
    def __init__(self, img_size=128):
        super().__init__()

        self.features = nn.Sequential(
            # Blok 1: 3 → 32
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),                    # /2
            nn.Dropout(0.25),

            # Blok 2: 32 → 64
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),                    # /2
            nn.Dropout(0.25),

            # Blok 3: 64 → 128
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),                    # /2
            nn.Dropout(0.25),
        )

        # 3× MaxPool(2) => pembagian 8
        final_size = img_size // 8
        flat_dim   = 128 * final_size * final_size

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat_dim, 256), nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 2)
        )

    def forward(self, x):
        return self.classifier(self.features(x))


# --- Test cepat ---
if __name__ == '__main__':
    for size in [64, 128, 224]:
        m = CatDogCNN(img_size=size)
        out = m(torch.randn(2, 3, size, size))
        n = sum(p.numel() for p in m.parameters())
        print(f"IMG_SIZE={size:3d} | out={tuple(out.shape)} | params={n:,}")