# ============================================
# cat1_data.py
# Dataset + DataLoader untuk Cats vs Dogs
# ============================================
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import random

# --- Konfigurasi global (di-export ke file lain) ---
IMG_SIZE = 64           # 64 = cepat di CPU; 128 = akurasi lebih baik
BATCH    = 32
USE_SUBSET = True       # True = pakai 3000 gambar (prototipe cepat)
SUBSET_SIZE = 3000

# --- Transform ---
train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

test_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

# --- Dataset ---
train_ds = datasets.ImageFolder('data_catsdogs/training_set', transform=train_transform)
test_ds  = datasets.ImageFolder('data_catsdogs/test_set',     transform=test_transform)

# --- Subset opsional untuk prototipe ---
if USE_SUBSET:
    random.seed(42)
    idx = random.sample(range(len(train_ds)), min(SUBSET_SIZE, len(train_ds)))
    train_ds_used = Subset(train_ds, idx)
else:
    train_ds_used = train_ds

# --- DataLoader ---
# num_workers=0 wajib di CPU kecil / Codespace (hindari error spawn)
train_loader = DataLoader(train_ds_used, batch_size=BATCH, shuffle=True,  num_workers=0)
test_loader  = DataLoader(test_ds,      batch_size=BATCH, shuffle=False, num_workers=0)


# --- Test cepat (hanya jalan jika file ini dieksekusi langsung) ---
if __name__ == '__main__':
    print("Kelas        :", train_ds.classes)
    print(f"Train total  : {len(train_ds)}")
    print(f"Train dipakai: {len(train_ds_used)}")
    print(f"Test         : {len(test_ds)}")
    print(f"IMG_SIZE     : {IMG_SIZE}")

    gambar, label = next(iter(train_loader))
    print("Shape batch  :", gambar.shape)   # [32, 3, IMG_SIZE, IMG_SIZE]
    print("Shape label  :", label.shape)