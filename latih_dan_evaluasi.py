# ============================================
# LATIH + EVALUASI MODEL IRIS (ALL-IN-ONE)
# ============================================
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

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
print("✅ Model selesai dilatih!\n")

# --- 5. Prediksi ---
y_pred = model.predict(X_test)

# --- 6. Evaluasi ---
acc = accuracy_score(y_test, y_pred)
print(f"🎯 Akurasi: {acc*100:.2f}%\n")

print("=== Classification Report ===")
print(classification_report(y_test, y_pred, target_names=iris.target_names))

print("=== Confusion Matrix ===")
print(confusion_matrix(y_test, y_pred))