import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from nlp3_model import buat_model
from nlp4_dataset import get_datasets


BATCH_SIZE   = 2       # kecil -> hemat RAM
ACCUM_STEPS  = 2       # 2 x 3 = 6 -> batch efektif sama dengan batch=6
EPOCHS       = 3


def evaluasi(model, loader, device):
    model.eval()
    benar, total = 0, 0
    with torch.no_grad():
        for batch in loader:
            ids  = batch['input_ids'].to(device)
            mask = batch['attention_mask'].to(device)
            lbl  = batch['label'].to(device)
            out  = model(input_ids=ids, attention_mask=mask).logits
            benar += (out.argmax(1) == lbl).sum().item()
            total += lbl.size(0)
    return benar / total * 100


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device:", device, flush=True)

    train_ds, test_ds, _ = get_datasets()
    print(f"Train: {len(train_ds)} | Test: {len(test_ds)}", flush=True)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE,
                              shuffle=True, num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE,
                              shuffle=False, num_workers=0)

    print(f"Batch size     : {BATCH_SIZE}", flush=True)
    print(f"Accum steps    : {ACCUM_STEPS}", flush=True)
    print(f"Efektif batch  : {BATCH_SIZE * ACCUM_STEPS}", flush=True)

    model = buat_model(num_labels=2).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)

    best_acc = 0.0

    for epoch in range(EPOCHS):
        model.train()
        t0 = time.time()
        total_loss, benar, total = 0, 0, 0
        optimizer.zero_grad()

        for step, batch in enumerate(train_loader):
            ids  = batch['input_ids'].to(device)
            mask = batch['attention_mask'].to(device)
            lbl  = batch['label'].to(device)

            out  = model(input_ids=ids, attention_mask=mask).logits
            loss = criterion(out, lbl) / ACCUM_STEPS
            loss.backward()

            if (step + 1) % ACCUM_STEPS == 0:
                optimizer.step()
                optimizer.zero_grad()

            total_loss += loss.item() * ACCUM_STEPS
            benar += (out.argmax(1) == lbl).sum().item()
            total += lbl.size(0)

            print(f"  [batch {step+1}/{len(train_loader)}] "
                  f"size={lbl.size(0)} loss={loss.item()*ACCUM_STEPS:.4f}",
                  flush=True)

        # Sisa gradient kalau step tidak habis dibagi ACCUM_STEPS
        if len(train_loader) % ACCUM_STEPS != 0:
            optimizer.step()
            optimizer.zero_grad()

        acc_tr  = benar / total * 100
        acc_val = evaluasi(model, test_loader, device)
        print(f"Epoch {epoch+1}/{EPOCHS} | "
              f"Loss {total_loss/len(train_loader):.4f} | "
              f"Train {acc_tr:.2f}% | Val {acc_val:.2f}% | "
              f"{time.time()-t0:.1f}s", flush=True)

        if acc_val > best_acc:
            best_acc = acc_val
            model.save_pretrained('./model_sentimen')
            print(f"  ✅ Model disimpan (val {acc_val:.2f}%)", flush=True)

    print(f"\n🎉 Selesai. Val terbaik: {best_acc:.2f}%", flush=True)