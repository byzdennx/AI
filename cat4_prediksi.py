# ============================================
# cat4_prediksi.py
# Visualisasi prediksi pada data test
# ============================================
import torch
import matplotlib.pyplot as plt

from cat1_data import test_ds, IMG_SIZE
from cat2_model import CatDogCNN


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device   : {device}")
    print(f"IMG_SIZE : {IMG_SIZE}")

    model = CatDogCNN(img_size=IMG_SIZE).to(device)
    model.load_state_dict(torch.load('catdog_cnn.pth', map_location=device))
    model.eval()
    print("✅ Model dimuat dari catdog_cnn.pth")

    idx = [0, 5, 20, 50, 100, 200]
    fig, axes = plt.subplots(2, 3, figsize=(10, 7))

    for ax, i in zip(axes.flatten(), idx):
        img, label = test_ds[i]

        with torch.no_grad():
            pred = model(img.unsqueeze(0).to(device)).argmax(1).item()

        # Denormalisasi: [-1,1] -> [0,1]
        ax.imshow(img.permute(1, 2, 0) * 0.5 + 0.5)
        benar = (pred == label)
        ax.set_title(
            f"Asli: {test_ds.classes[label]}\nPred: {test_ds.classes[pred]}",
            color='green' if benar else 'red',
            fontsize=10
        )
        ax.axis('off')

    plt.tight_layout()
    plt.savefig('catdog_hasil.png', dpi=100)
    print("✅ Gambar disimpan: catdog_hasil.png")