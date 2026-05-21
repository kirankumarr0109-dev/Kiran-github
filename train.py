import pandas as pd
import numpy as np
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.multiclass import unique_labels
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("data/processed/landmarks.csv")

print("Samples per letter:")
counts = df["label"].value_counts().sort_index()
print(counts)

# Drop letters with less than 20 samples
valid = counts[counts >= 20].index
df = df[df["label"].isin(valid)]
print(f"\nUsing {len(valid)} letters, {len(df)} total samples")

X = df.drop("label", axis=1).values
y = df["label"].values

le = LabelEncoder()
y_enc = le.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(
    X, y_enc, test_size=0.2, random_state=42)

print(f"\nTraining on {len(X_train)} samples...")

model = RandomForestClassifier(
    n_estimators=100,
    max_depth=20,
    random_state=42,
    n_jobs=-1,
    verbose=1
)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
present = unique_labels(y_test, y_pred)
names = le.inverse_transform(present)

print("\n=== Results ===")
print(classification_report(y_test, y_pred,
      labels=present, target_names=names))

# Confusion matrix
os.makedirs("models", exist_ok=True)
cm = confusion_matrix(y_test, y_pred, labels=present)
plt.figure(figsize=(16, 14))
sns.heatmap(cm, annot=True, fmt="d",
            xticklabels=names, yticklabels=names,
            cmap="Blues")
plt.title("Confusion Matrix — ASL A-Z")
plt.ylabel("True")
plt.xlabel("Predicted")
plt.tight_layout()
plt.savefig("models/confusion_matrix.png", dpi=150)
plt.show()

with open("models/sign_model.pkl", "wb") as f:
    pickle.dump({"model": model, "encoder": le}, f)

print("\nModel saved! Now run: python inference.py")