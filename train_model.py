import numpy as np
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import joblib


MODE = input("Select training mode (gesture / asl): ").strip().lower()

if MODE == "gesture":
    DATA_DIR = "data/basic_gestures"
    LABELS = ["fist", "palm", "point"]
    MODEL_PATH = "models/gesture_model.pkl"

elif MODE == "asl":
    DATA_DIR = "data/asl_static"
    
    LABELS = ["A", "B", "C", "D", "E"]
    MODEL_PATH = "models/asl_static_model.pkl"

else:
    raise ValueError("Mode must be 'gesture' or 'asl'")

os.makedirs("models", exist_ok=True)
RANDOM_STATE = 42


X = []
y = []


for label_idx, label_name in enumerate(LABELS):
    folder = os.path.join(DATA_DIR, label_name)
    if not os.path.exists(folder):
        print(f"Warning: {folder} not found, skipping.")
        continue

    for file in os.listdir(folder):
        if file.endswith(".npy"):
            data = np.load(os.path.join(folder, file))
            X.append(data)
            y.append(label_idx)

X = np.array(X)
y = np.array(y)

print(f"\nMode: {MODE.upper()}")
print(f"Labels: {LABELS}")
print(f"Loaded {len(X)} samples")

if len(X) == 0:
    raise RuntimeError("No data found. Please collect data first.")


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=RANDOM_STATE,
    stratify=y
)


model = RandomForestClassifier(
    n_estimators=300,
    random_state=RANDOM_STATE,
    n_jobs=-1
)

model.fit(X_train, y_train)


preds = model.predict(X_test)
acc = accuracy_score(y_test, preds)
cm = confusion_matrix(y_test, preds)

print(f"\nAccuracy: {acc:.4f}")
print("Confusion Matrix:")
print(cm)


joblib.dump(
    {
        "model": model,
        "labels": LABELS,
        "mode": MODE
    },
    MODEL_PATH
)

print(f"\nModel saved at: {MODEL_PATH}")
