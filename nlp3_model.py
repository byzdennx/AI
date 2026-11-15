# ============================================
# nlp3_model.py
# Load model pretrained untuk klasifikasi
# ============================================
import torch
from transformers import AutoModelForSequenceClassification

# Model ringan + multilingual (pasti jalan, RAM ~700 MB)
MODEL_NAME = "microsoft/Multilingual-MiniLM-L12-H384"

# Alternatif (uncomment kalau mau coba):
# MODEL_NAME = "distilbert-base-multilingual-cased"      # ~2 GB
# MODEL_NAME = "indobenchmark/indobert-base-p1"          # ~2.5 GB, akurasi terbaik Indonesia


def buat_model(num_labels=2):
    return AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=num_labels
    )


if __name__ == '__main__':
    model = buat_model()
    total = sum(p.numel() for p in model.parameters())
    print(f"Model          : {MODEL_NAME}")
    print(f"Total parameter: {total:,}")

    dummy = torch.randint(0, 1000, (2, 16))
    out = model(input_ids=dummy).logits
    print(f"Output shape   : {tuple(out.shape)}")