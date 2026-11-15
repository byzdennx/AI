import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Transform: ubah gambar jadi tensor + normalisasi (0-1 → -1..1)
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Download otomatis ke folder ./data
train_ds = datasets.MNIST('./data', train=True,  download=True, transform=transform)
test_ds  = datasets.MNIST('./data', train=False, download=True, transform=transform)

# DataLoader = pembungkus batch
train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
test_loader  = DataLoader(test_ds,  batch_size=64, shuffle=False)

print(f"Data train: {len(train_ds)} gambar")
print(f"Data test : {len(test_ds)} gambar")

# Cek 1 batch
gambar, label = next(iter(train_loader))
print(f"Shape batch gambar: {gambar.shape}")   # [64, 1, 28, 28]
print(f"Shape batch label : {label.shape}")    # [64]