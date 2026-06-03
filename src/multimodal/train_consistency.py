"""
Insurance Claim Intelligence — Multimodal Consistency Model

Trainiert einen Klassifikator der erkennt ob Schadenbild und
Schadenbeschreibung fachlich zusammenpassen.

Input:  data/processed/multimodal_consistency_cases.csv
        (generiert von build_consistency_dataset.py)
Output: models/consistency_model.pkl
        models/consistency_results.json

Features:
  - cv_label_enc:       Encoded predicted damage type
  - cv_confidence:      Simuliert aus ViT F1-Scores
  - nlp_incident_type:  Encoded incident type aus Beschreibung
  - fraud_signal_count: Anzahl Fraud-Signalwörter
  - description_length: Wortanzahl
  - cosine_similarity:  CV-Label Embedding vs. Beschreibungs-Embedding
  - severity_mismatch:  1 wenn Schweregrad nicht passt

Zielvariable: consistency_label (1=konsistent, 0=inkonsistent)
"""

import pandas as pd
import numpy as np
import json
import joblib
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (classification_report, roc_auc_score,
                              f1_score, confusion_matrix, ConfusionMatrixDisplay)
from sklearn.preprocessing import LabelEncoder
from sentence_transformers import SentenceTransformer
import xgboost as xgb
import warnings
warnings.filterwarnings("ignore")

SEED = 42
np.random.seed(SEED)
os.makedirs("models", exist_ok=True)

INSURANCE_CLASSES = ["dent", "scratch", "crack", "glass_shatter", "no_damage"]
label2id = {l: i for i, l in enumerate(INSURANCE_CLASSES)}

FRAUD_SIGNALS = ["totalschaden", "gestohlen", "keine zeugen", "dringend",
                 "sofort", "bar bezahlt", "keine quittung", "auszahlung"]
INCIDENT_TYPES = {
    "parking":   ["parkschaden", "parklücke", "parkplatz"],
    "vandalism": ["vandalismus", "schlüssel", "zerkratzt"],
    "collision": ["kollision", "auffahrunfall", "aufprall"],
    "theft":     ["gestohlen", "einbruch", "diebstahl"],
    "weather":   ["hagel", "sturm", "steinschlag"],
}
F1_MAP = {"dent": 0.66, "scratch": 0.98, "crack": 0.98,
          "glass_shatter": 0.83, "no_damage": 0.80}
SEVERITY_WORDS_HIGH = ["totalschaden", "massiv", "komplett", "nicht fahrbereit",
                        "airbags", "wirtschaftlich"]
SEVERITY_WORDS_LOW  = ["kleiner", "oberflächlich", "kaum", "minimal", "leicht"]

# ──────────────────────────────────────────────────────────────
# 1. Dataset laden und Features berechnen
# ──────────────────────────────────────────────────────────────
print("Lade Consistency Dataset...")
df = pd.read_csv("data/processed/multimodal_consistency_cases.csv")
print(f"  {len(df)} Faelle geladen")
print(f"  Konsistent: {df['consistency_label'].sum()} / Inkonsistent: {(df['consistency_label']==0).sum()}")

print("Berechne SentenceTransformer Embeddings...")
embedder = SentenceTransformer("all-MiniLM-L6-v2", backend="torch")

# Embeddings für alle Damage Labels (einmalig)
label_embeddings = {
    cls: embedder.encode([cls])[0]
    for cls in INSURANCE_CLASSES
}

# Features extrahieren
features = []
for _, row in df.iterrows():
    cls       = row["true_damage_label"]
    desc      = str(row["description"]).lower()

    # CV Features
    cv_label_enc  = label2id.get(cls, 0)
    cv_confidence = F1_MAP.get(cls, 0.75) + np.random.uniform(-0.03, 0.03)

    # NLP Features
    incident_type = 0
    for i, (itype, keywords) in enumerate(INCIDENT_TYPES.items()):
        if any(kw in desc for kw in keywords):
            incident_type = i + 1
            break
    fraud_count  = sum(1 for s in FRAUD_SIGNALS if s in desc)
    desc_length  = len(desc.split())

    # Cosine Similarity CV-Label vs. Beschreibung
    desc_emb = embedder.encode([desc])[0]
    cv_emb   = label_embeddings[cls]
    cos_sim  = float(np.dot(cv_emb, desc_emb) /
                     (np.linalg.norm(cv_emb) * np.linalg.norm(desc_emb) + 1e-8))

    # Severity Mismatch Flag
    has_high = any(w in desc for w in SEVERITY_WORDS_HIGH)
    has_low  = any(w in desc for w in SEVERITY_WORDS_LOW)
    severity_mismatch = 0
    if cls in ("dent", "scratch", "crack") and has_high:
        severity_mismatch = 1
    if cls == "glass_shatter" and has_low:
        severity_mismatch = 1

    features.append({
        "cv_label_enc":      cv_label_enc,
        "cv_confidence":     float(np.clip(cv_confidence, 0.5, 0.99)),
        "nlp_incident_type": incident_type,
        "fraud_signal_count": fraud_count,
        "description_length": desc_length,
        "cosine_similarity":  float(np.clip(cos_sim, 0.0, 1.0)),
        "severity_mismatch":  severity_mismatch,
    })

X = pd.DataFrame(features)
y = df["consistency_label"].values

print(f"\nFeature-Matrix: {X.shape}")
print(f"Cosine Similarity Ø: {X['cosine_similarity'].mean():.3f}")
print(f"Fraud Signals Ø: {X['fraud_signal_count'].mean():.2f}")

# ──────────────────────────────────────────────────────────────
# 2. Split
# ──────────────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=SEED, stratify=y)
print(f"\nSplit: Train={len(X_train)}, Test={len(X_test)}")

# ──────────────────────────────────────────────────────────────
# 3. Modelle trainieren — 3 Iterationen
# ──────────────────────────────────────────────────────────────
results = {}

# Iteration 1: Logistic Regression (Baseline)
print("\n--- ITERATION 1: Logistic Regression (Baseline) ---")
lr = LogisticRegression(random_state=SEED, max_iter=1000)
lr.fit(X_train, y_train)
lr_pred = lr.predict(X_test)
lr_auc  = roc_auc_score(y_test, lr.predict_proba(X_test)[:, 1])
lr_f1   = f1_score(y_test, lr_pred)
print(f"  F1={lr_f1:.3f} | AUC={lr_auc:.3f}")
results["logistic_regression"] = {"f1": float(lr_f1), "auc": float(lr_auc)}

# Iteration 2: Random Forest
print("\n--- ITERATION 2: Random Forest ---")
rf = RandomForestClassifier(n_estimators=200, random_state=SEED, n_jobs=-1)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)
rf_auc  = roc_auc_score(y_test, rf.predict_proba(X_test)[:, 1])
rf_f1   = f1_score(y_test, rf_pred)
print(f"  F1={rf_f1:.3f} | AUC={rf_auc:.3f}")
results["random_forest"] = {"f1": float(rf_f1), "auc": float(rf_auc)}

# Iteration 3: XGBoost (Winner)
print("\n--- ITERATION 3: XGBoost ---")
xgb_model = xgb.XGBClassifier(
    n_estimators=200, random_state=SEED, verbosity=0,
    eval_metric="logloss", use_label_encoder=False)
xgb_model.fit(X_train, y_train)
xgb_pred = xgb_model.predict(X_test)
xgb_auc  = roc_auc_score(y_test, xgb_model.predict_proba(X_test)[:, 1])
xgb_f1   = f1_score(y_test, xgb_pred)
print(f"  F1={xgb_f1:.3f} | AUC={xgb_auc:.3f}")
results["xgboost"] = {"f1": float(xgb_f1), "auc": float(xgb_auc)}

# ──────────────────────────────────────────────────────────────
# 4. Evaluation — bestes Modell
# ──────────────────────────────────────────────────────────────
best_name  = max(results, key=lambda k: results[k]["auc"])
best_model = {"logistic_regression": lr,
              "random_forest": rf,
              "xgboost": xgb_model}[best_name]
best_pred  = {"logistic_regression": lr_pred,
              "random_forest": rf_pred,
              "xgboost": xgb_pred}[best_name]

print(f"\n=== WINNER: {best_name} ===")
print(classification_report(y_test, best_pred,
                             target_names=["Inkonsistent", "Konsistent"]))

# Error Analysis nach mismatch_type
test_df = df.iloc[X_test.index].copy()
test_df["predicted"] = best_pred
test_df["correct"]   = (test_df["consistency_label"] == test_df["predicted"])

print("Error Analysis nach mismatch_type:")
for mtype in test_df["mismatch_type"].unique():
    sub = test_df[test_df["mismatch_type"] == mtype]
    acc = sub["correct"].mean()
    print(f"  {mtype:30}: Accuracy={acc:.2%} (n={len(sub)})")

# Konfusionsmatrix
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

cm = confusion_matrix(y_test, best_pred)
disp = ConfusionMatrixDisplay(cm, display_labels=["Inkonsistent", "Konsistent"])
disp.plot(ax=axes[0], colorbar=False, cmap="Blues")
axes[0].set_title(f"Konfusionsmatrix — {best_name}")

# Modellvergleich
models = list(results.keys())
aucs   = [results[m]["auc"] for m in models]
f1s    = [results[m]["f1"]  for m in models]
x      = np.arange(len(models))
width  = 0.35
axes[1].bar(x - width/2, aucs, width, label="AUC",  color="#2563eb")
axes[1].bar(x + width/2, f1s,  width, label="F1",   color="#10b981")
axes[1].set_xticks(x)
axes[1].set_xticklabels(["Log. Reg.", "Random Forest", "XGBoost"])
axes[1].set_ylim(0, 1.1)
axes[1].set_title("Modellvergleich — Consistency Detection")
axes[1].legend()
for i, (auc, f1) in enumerate(zip(aucs, f1s)):
    axes[1].text(i - width/2, auc + 0.02, f"{auc:.2f}", ha="center", fontsize=9)
    axes[1].text(i + width/2, f1  + 0.02, f"{f1:.2f}",  ha="center", fontsize=9)

plt.tight_layout()
plt.savefig("models/consistency_results.png", dpi=150)
plt.show()

# ──────────────────────────────────────────────────────────────
# 5. Speichern
# ──────────────────────────────────────────────────────────────
joblib.dump(best_model, "models/consistency_model.pkl")
joblib.dump(embedder,   "models/consistency_embedder.pkl")

FEATURE_COLS = list(X.columns)
with open("models/consistency_feature_cols.json", "w") as f:
    json.dump(FEATURE_COLS, f)

results["winner"]       = best_name
results["feature_cols"] = FEATURE_COLS
with open("models/consistency_results.json", "w") as f:
    json.dump(results, f, indent=2)

print(f"\nModellvergleich Consistency Detection:")
for name, res in results.items():
    if isinstance(res, dict) and "auc" in res:
        winner = " <- WINNER" if name == best_name else ""
        print(f"  {name:25}: F1={res['f1']:.3f} | AUC={res['auc']:.3f}{winner}")

print(f"\nGespeichert:")
print(f"  models/consistency_model.pkl")
print(f"  models/consistency_results.json")
print(f"  models/consistency_results.png")
print("Multimodal Consistency Training abgeschlossen.")
