# ============================================
# nlp4_dataset.py
# PyTorch Dataset untuk teks
# ============================================
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer

from nlp1_data import load_data
from nlp3_model import MODEL_NAME

MAX_LEN = 16   # cukup untuk kalimat pendek seperti dataset Anda


class SentimenDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=MAX_LEN):
        self.texts = list(texts)
        self.labels = list(labels)
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            str(self.texts[idx]),
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        return {
            'input_ids'     : enc['input_ids'].squeeze(0),
            'attention_mask': enc['attention_mask'].squeeze(0),
            'label'         : torch.tensor(self.labels[idx], dtype=torch.long)
        }


def get_datasets():
    df = load_data()
    X_tr, X_te, y_tr, y_te = train_test_split(
        df['text'], df['label'],
        test_size=0.2, random_state=42, stratify=df['label']
    )
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    return (
        SentimenDataset(X_tr, y_tr, tok),
        SentimenDataset(X_te, y_te, tok),
        tok
    )


if __name__ == '__main__':
    tr, te, _ = get_datasets()
    print(f"Train: {len(tr)} | Test: {len(te)}")
    item = tr[0]
    print("Keys :", list(item.keys()))
    print("Shape:", item['input_ids'].shape)   # [MAX_LEN]