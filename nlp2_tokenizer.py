# ============================================
# nlp2_tokenizer.py
# Demo tokenizer: teks -> token -> ID
# ============================================
from transformers import AutoTokenizer
from nlp3_model import MODEL_NAME


if __name__ == '__main__':
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    kalimat = "Produk bagus, saya suka sekali!"
    tokens = tokenizer.tokenize(kalimat)
    print("Token   :", tokens)

    ids = tokenizer.encode(kalimat, add_special_tokens=True)
    print("Token ID:", ids)

    print("Decoded :", tokenizer.decode(ids))

    enc = tokenizer(kalimat, padding='max_length', truncation=True,
                    max_length=16, return_tensors='pt')
    print("\ninput_ids shape:", enc['input_ids'].shape)
    print("attention_mask :", enc['attention_mask'])