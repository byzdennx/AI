# ============================================
# tl3_train.py
# Training transfer learning ResNet18
# ============================================
import torch
import torch.nn as nn
import time
from tqdm import tqdm

from tl1_data import train_loader, test_loader, IMG_SIZE
from tl2_model import buat_model


def evaluasi(model, loader, device):
    model.eval()
    benar, total = 0, 0
    with torch.no_grad():
        for g, l in loader:
            g, l = g.to(device), l.to(device)
            benar += (model(g).argmax(1) == l).sum().item()
            total += l.size(0)
    return benar / total * 100


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device   : {device}")
    print(f"IMG_SIZE : {IMG_SIZE}")

    # --- FASE 1: Feature extraction (backbone beku) ---
    print("\n=== FASE 1: Feature Extraction ===")
    model = buat_model(freeze_backbone=True).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=0.001
    )

    EPOCHS_F1 = 3
    best_acc = 0.0

    for epoch in range(EPOCHS_F1):
        model.train()
        t0 = time.time()
        total_loss, benar, total = 0, 0, 0

        pbar = tqdm(train_loader, desc=f"F1 Epoch {epoch+1}/{EPOCHS_F1}", ncols=80)
        for g, l in pbar:
            g, l = g.to(device), l.to(device)
            optimizer.zero_grad()
            out = model(g)
            loss = criterion(out, l)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            benar += (out.argmax(1) == l).sum().item()
            total += l.size(0)
            pbar.set_postfix({'loss': f"{total_loss/len(train_loader):.4f}",
                              'acc': f"{benar/total*100:.2f}%"})

        acc_tr = benar/total*100
        acc_val = evaluasi(model, test_loader, device)
        print(f"F1 Epoch {epoch+1} | Loss {total_loss/len(train_loader):.4f} | "
              f"Train {acc_tr:.2f}% | Val {acc_val:.2f}% | {time.time()-t0:.1f}s")

        if acc_val > best_acc:
            best_acc = acc_val
            torch.save(model.state_dict(), 'resnet_catdog.pth')
            print(f"  ✅ Model disimpan (val {acc_val:.2f}%)")

    # --- FASE 2: Fine-tuning (buka layer layer4) ---
    print("\n=== FASE 2: Fine-tuning layer4 ===")
    model = buat_model(freeze_backbone=True).to(device)
    model.load_state_dict(torch.load('resnet_catdog.pth', map_location=device))

    # Buka blok terakhir ResNet
    for p in model.layer4.parameters():
        p.requires_grad = True

    # Learning rate LEBIH KECIL (jangan rusak bobot pretrained)
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=0.0001
    )

    EPOCHS_F2 = 2

    for epoch in range(EPOCHS_F2):
        model.train()
        t0 = time.time()
        total_loss, benar, total = 0, 0, 0

        pbar = tqdm(train_loader, desc=f"F2 Epoch {epoch+1}/{EPOCHS_F2}", ncols=80)
        for g, l in pbar:
            g, l = g.to(device), l.to(device)
            optimizer.zero_grad()
            out = model(g)
            loss = criterion(out, l)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            benar += (out.argmax(1) == l).sum().item()
            total += l.size(0)
            pbar.set_postfix({'loss': f"{total_loss/len(train_loader):.4f}",
                              'acc': f"{benar/total*100:.2f}%"})

        acc_tr = benar/total*100
        acc_val = evaluasi(model, test_loader, device)
        print(f"F2 Epoch {epoch+1} | Loss {total_loss/len(train_loader):.4f} | "
              f"Train {acc_tr:.2f}% | Val {acc_val:.2f}% | {time.time()-t0:.1f}s")

        if acc_val > best_acc:
            best_acc = acc_val
            torch.save(model.state_dict(), 'resnet_catdog.pth')
            print(f"  ✅ Model disimpan (val {acc_val:.2f}%)")

    print(f"\n🎉 Selesai. Akurasi validasi terbaik: {best_acc:.2f}%")
    print("Model tersimpan: resnet_catdog.pth")