# ============================================
# tl1_data.py
# Dataset untuk transfer learning ResNet18
# ============================================
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import random

IMG_SIZE     = 224      # ResNet18 butuh 224 (standar ImageNet)
BATCH        = 32
USE_SUBSET   = True     # True = 4000 gambar (cukup untuk fine-tuning)
SUBSET_SIZE  = 4000

# Normalization ImageNet — WAJIB untuk model pretrained
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
])

test_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
])

train_ds = datasets.ImageFolder('data_catsdogs/training_set', transform=train_transform)
test_ds  = datasets.ImageFolder('data_catsdogs/test_set',     transform=test_transform)

if USE_SUBSET:
    random.seed(42)
    idx = random.sample(range(len(train_ds)), min(SUBSET_SIZE, len(train_ds)))
    train_ds_used = Subset(train_ds, idx)
else:
    train_ds_used = train_ds

train_loader = DataLoader(train_ds_used, batch_size=BATCH, shuffle=True,  num_workers=0)
test_loader  = DataLoader(test_ds,      batch_size=BATCH, shuffle=False, num_workers=0)


if __name__ == '__main__':
    print("Kelas        :", train_ds.classes)
    print(f"Train dipakai: {len(train_ds_used)}")
    print(f"Test         : {len(test_ds)}")
    g, l = next(iter(train_loader))
    print("Shape batch  :", g.shape)   # [32, 3, 224, 224]