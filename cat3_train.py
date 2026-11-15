# ============================================
# cat3_train.py
# Training loop Cats vs Dogs
# ============================================
import torch
import torch.nn as nn
import time
from tqdm import tqdm

from cat1_data import train_loader, test_loader, IMG_SIZE
from cat2_model import CatDogCNN


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

    model = CatDogCNN(img_size=IMG_SIZE).to(device)
    n_param = sum(p.numel() for p in model.parameters())
    print(f"Parameter: {n_param:,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    EPOCHS = 5
    best_acc = 0.0

    for epoch in range(EPOCHS):
        model.train()
        t0 = time.time()
        total_loss, benar, total = 0, 0, 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}", ncols=80)
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

            pbar.set_postfix({
                'loss': f"{total_loss/len(train_loader):.4f}",
                'acc' : f"{benar/total*100:.2f}%"
            })

        acc_train = benar / total * 100
        acc_val   = evaluasi(model, test_loader, device)
        durasi    = time.time() - t0

        print(f"Epoch {epoch+1}/{EPOCHS} | "
              f"Loss {total_loss/len(train_loader):.4f} | "
              f"Train {acc_train:.2f}% | Val {acc_val:.2f}% | "
              f"{durasi:.1f}s")

        # Simpan model terbaik
        if acc_val > best_acc:
            best_acc = acc_val
            torch.save(model.state_dict(), 'catdog_cnn.pth')
            print(f"  ✅ Model terbaik disimpan (val {acc_val:.2f}%)")

    print(f"\n🎉 Selesai. Akurasi validasi terbaik: {best_acc:.2f}%")