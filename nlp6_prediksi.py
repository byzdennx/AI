# ============================================
# nlp6_prediksi.py
# Inference model sentimen
# ============================================
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_DIR = './model_sentimen'
LABELS = {0: 'NEGATIF ❌', 1: 'POSITIF ✅'}


def prediksi(teks_list, model, tokenizer, device):
    hasil = []
    model.eval()
    for teks in teks_list:
        enc = tokenizer(teks, return_tensors='pt',
                        padding=True, truncation=True, max_length=32).to(device)
        with torch.no_grad():
            logits = model(**enc).logits
            probs  = torch.softmax(logits, dim=1)[0]
            pred   = logits.argmax(1).item()
        hasil.append({
            'teks'   : teks,
            'label'  : pred,
            'prob_pos': probs[1].item(),
            'prob_neg': probs[0].item()
        })
    return hasil


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device:", device, flush=True)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR).to(device)
    print("✅ Model dimuat dari", MODEL_DIR, flush=True)

    teks_baru = [
        "Produknya bagus, saya puas sekali!",
        "Pengiriman sangat lambat, kecewa",
        "Lumayan, sesuai harga",
        "Barang rusak, tidak berfungsi",
        "Suka banget, akan order lagi"
    ]

    print("\n=== Hasil Prediksi ===", flush=True)
    for r in prediksi(teks_baru, model, tokenizer, device):
        print(f"\n'{r['teks']}'")
        print(f"  → {LABELS[r['label']]}")
        print(f"     positif={r['prob_pos']:.2%} | negatif={r['prob_neg']:.2%}")