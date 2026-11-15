import torch, matplotlib.pyplot as plt
from torchvision import datasets, transforms
from dl2_model import CNN

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
transform = transforms.Compose([
    transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))
])

# Ambil 5 gambar dari test set
test_ds = datasets.MNIST('./data', train=False, transform=transform)
gambar_batch = torch.stack([test_ds[i][0] for i in range(5)])
label_asli   = [test_ds[i][1] for i in range(5)]

# Load model
model = CNN().to(device)
model.load_state_dict(torch.load('model_cnn.pth'))
model.eval()

with torch.no_grad():
    prediksi = model(gambar_batch.to(device)).argmax(1).cpu().numpy()

# Tampilkan
fig, axes = plt.subplots(1, 5, figsize=(12, 3))
for i, ax in enumerate(axes):
    ax.imshow(gambar_batch[i][0], cmap='gray')
    ax.set_title(f"Asli:{label_asli[i]} | Pred:{prediksi[i]}",
                 color='green' if label_asli[i]==prediksi[i] else 'red')
    ax.axis('off')
plt.tight_layout()
plt.savefig('hasil_prediksi.png')
print("✅ Gambar disimpan: hasil_prediksi.png")
print("Prediksi:", prediksi)