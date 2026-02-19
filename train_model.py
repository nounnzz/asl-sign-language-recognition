"""
ASL Model Trainer (scikit-learn version)
==========================================
Trains a Random Forest + MLP ensemble on hand landmark features.
No TensorFlow required — scikit-learn only.

Usage:
    python train_model.py --data data/landmarks.csv
    python train_model.py --data data/landmarks.csv --model model/asl_model.pkl
"""

import argparse
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.pipeline import Pipeline

parser = argparse.ArgumentParser()
parser.add_argument("--data",  default="data/landmarks.csv",    help="Path to landmark CSV")
parser.add_argument("--model", default="model/asl_model.pkl",   help="Where to save the model")
args = parser.parse_args()

LABELS = list("ABCDEFGHIKLMNOPQRSTUVWXY")  # 24 ASL letters

def load_data(csv_path):
    print(f"[INFO] Loading data from: {csv_path}")
    df = pd.read_csv(csv_path, header=None)
    labels   = df.iloc[:, 0].astype(str).str.upper().values
    features = df.iloc[:, 1:].values.astype(np.float32)
    print(f"[INFO] Loaded {len(df)} samples across {len(set(labels))} classes")
    print(f"[INFO] Feature shape: {features.shape}")
    return features, labels

def plot_results(y_true, y_pred, label_names, save_dir="model"):
    os.makedirs(save_dir, exist_ok=True)

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(14, 12))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.colorbar(im, ax=ax)
    ax.set_xticks(range(len(label_names)))
    ax.set_yticks(range(len(label_names)))
    ax.set_xticklabels(label_names, rotation=45)
    ax.set_yticklabels(label_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "confusion_matrix.png"), dpi=120)
    print(f"[INFO] Confusion matrix saved to {save_dir}/confusion_matrix.png")
    plt.close()

    # Per-class accuracy bar chart
    per_class_acc = cm.diagonal() / cm.sum(axis=1)
    fig, ax = plt.subplots(figsize=(12, 5))
    bars = ax.bar(label_names, per_class_acc * 100,
                  color=["#2ecc71" if v >= 90 else "#e67e22" if v >= 75 else "#e74c3c"
                         for v in per_class_acc * 100])
    ax.set_ylim(0, 105)
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Per-Letter Accuracy")
    ax.axhline(y=90, color="gray", linestyle="--", alpha=0.5, label="90% line")
    for bar, val in zip(bars, per_class_acc):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f"{val*100:.0f}%", ha="center", va="bottom", fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "per_class_accuracy.png"), dpi=120)
    print(f"[INFO] Per-class accuracy chart saved to {save_dir}/per_class_accuracy.png")
    plt.close()

def train():
    # ── Load data ──────────────────────────────────────────────────────────
    if not os.path.exists(args.data):
        print(f"[ERROR] Data file not found: {args.data}")
        print("[INFO] Run 'python collect_data.py' first to collect training data.")
        return

    X, y_str = load_data(args.data)

    # Encode labels
    le = LabelEncoder()
    le.fit(LABELS)
    y = le.transform(y_str)

    # Train/val split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"[INFO] Train: {len(X_train)} | Val: {len(X_val)}")

    # ── Build ensemble model ───────────────────────────────────────────────
    print("\n[INFO] Training model (this may take 1-2 minutes)...")

    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        min_samples_split=2,
        random_state=42,
        n_jobs=-1,
    )

    mlp = MLPClassifier(
        hidden_layer_sizes=(256, 128, 64),
        activation="relu",
        max_iter=500,
        random_state=42,
        early_stopping=True,
        validation_fraction=0.1,
        learning_rate_init=0.001,
    )

    # Voting ensemble: RF + MLP
    ensemble = VotingClassifier(
        estimators=[("rf", rf), ("mlp", mlp)],
        voting="soft",
    )

    # Wrap in a pipeline with scaling (MLP benefits from scaled input)
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model",  ensemble),
    ])

    pipeline.fit(X_train, y_train)

    # ── Evaluate ───────────────────────────────────────────────────────────
    y_pred    = pipeline.predict(X_val)
    val_acc   = accuracy_score(y_val, y_pred)

    print(f"\n{'='*50}")
    print(f"  Validation Accuracy: {val_acc*100:.2f}%")
    print(f"{'='*50}\n")
    print(classification_report(y_val, y_pred, target_names=le.classes_))

    # ── Save model ─────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(args.model), exist_ok=True)
    with open(args.model, "wb") as f:
        pickle.dump({"pipeline": pipeline, "label_encoder": le}, f)
    print(f"[INFO] Model saved to: {args.model}")

    # ── Plot results ───────────────────────────────────────────────────────
    y_val_labels  = le.inverse_transform(y_val)
    y_pred_labels = le.inverse_transform(y_pred)
    plot_results(y_val_labels, y_pred_labels, le.classes_)

    print("\n[INFO] Training complete!")
    if val_acc >= 0.95:
        print("[INFO] Excellent accuracy! Ready for real-time inference.")
    elif val_acc >= 0.85:
        print("[INFO] Good accuracy. Consider collecting more samples for lower-accuracy letters.")
    else:
        print("[INFO] Accuracy is low. Try collecting 150-200 samples per letter and retrain.")

if __name__ == "__main__":
    train()
