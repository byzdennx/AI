from transformers import AutoTokenizer

MODEL_NAME = "microsoft/Multilingual-MiniLM-L12-H384"
tok = AutoTokenizer.from_pretrained(MODEL_NAME)

print("Tipe tokenizer:", type(tok).__name__)
print("Vocab size    :", tok.vocab_size)
print("\nTes tokenisasi:")
print(tok.tokenize("Produk bagus, saya suka sekali!"))