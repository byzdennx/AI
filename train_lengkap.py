# ============================================
# TRAINING LENGKAP — dari data sampai simpan
# ============================================
import joblib
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# --- 1. Load data ---
iris = load_iris()
X, y = iris.data, iris.target

# --- 2. Split ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# --- 3. Scaling ---
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

# --- 4. Latih model ---
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# --- 5. Evaluasi ---
y_pred = model.predict(X_test)
print(f"Akurasi: {accuracy_score(y_test, y_pred)*100:.2f}%")

# --- 6. Simpan (model & scaler SUDAH ada di memori file ini) ---
joblib.dump(model,  'model_iris.pkl')
joblib.dump(scaler, 'scaler_iris.pkl')
print("✅ Model & scaler tersimpan!")