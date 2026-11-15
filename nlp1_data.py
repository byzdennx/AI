# ============================================
# nlp1_data.py
# Load dataset sentimen dari CSV
# ============================================
import pandas as pd


def load_data(csv_path='nlp_dataset.csv'):
    df = pd.read_csv(csv_path)
    df = df.dropna().drop_duplicates().reset_index(drop=True)
    df['label'] = (df['label'].str.lower().str.strip() == 'positif').astype(int)
    return df


if __name__ == '__main__':
    df = load_data()
    print(f"Total data : {len(df)}")
    print(f"Positif    : {(df['label'] == 1).sum()}")
    print(f"Negatif    : {(df['label'] == 0).sum()}")
    print("\nContoh:")
    print(df.head())