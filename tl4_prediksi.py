# ============================================
# tl4_prediksi.py
# Visualisasi hasil transfer learning
# ============================================
import torch
import matplotlib.pyplot as plt

from tl1_data import test_ds, IMG_SIZE, IMAGENET_MEAN, IMAGENET_STD
from tl2_model import buat_model


def denormalize(tensor):
    """Ubah dari ternormalisasi ImageNet -> gambar normal untuk display."""
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std  = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    return (tensor * std + mean).clamp(0, 1)


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = buat_model(freeze_backbone=True).to(device)
    model.load_state_dict(torch.load('resnet_catdog.pth', map_location=device))
    model.eval()
    print("✅ Model dimuat.")

    idx = [0, 5, 20, 50, 100, 200]
    fig, axes = plt.subplots(2, 3, figsize=(11, 7))

    benar_total = 0
    for ax, i in zip(axes.flatten(), idx):
        img, label = test_ds[i]
        with torch.no_grad():
            pred = model(img.unsqueeze(0).to(device)).argmax(1).item()

        ok = (pred == label)
        benar_total += ok
        ax.imshow(denormalize(img).permute(1, 2, 0))
        ax.set_title(
            f"Asli: {test_ds.classes[label]}\nPred: {test_ds.classes[pred]}",
            color='green' if ok else 'red', fontsize=10
        )
        ax.axis('off')

    plt.tight_layout()
    plt.savefig('resnet_hasil.png', dpi=100)
    print(f"✅ {benar_total}/{len(idx)} benar. Lihat resnet_hasil.png")