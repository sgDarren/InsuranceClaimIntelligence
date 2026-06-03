"""
Insurance Claim Intelligence — Multimodal Consistency Dataset Builder

Erstellt einen kuratierten Benchmark-Dataset für Consistency Detection:
Erkennt ob Schadenbild und Schadenbeschreibung fachlich zusammenpassen.

Dieser Task ist für Fraud-Triage relevant: Inkonsistenz zwischen Bild und
Text ist ein starkes Signal für manuelle Prüfung.

Input:  CV-Labels aus ViT Fine-Tuning (label_map.json + insurance_labels_balanced.csv)
Output: data/processed/multimodal_consistency_cases.csv

Spalten:
  idx               - Index im HuggingFace Dataset
  true_damage_label - Wahrer Schadenstyp (aus ViT Training)
  description       - Schadenbeschreibung (passend oder absichtlich falsch)
  consistency_label - 1 = konsistent, 0 = inkonsistent
  mismatch_type     - none / damage_type_mismatch / severity_mismatch / fraud_signal
  severity_label    - minor / medium / severe
"""

import pandas as pd
import numpy as np
import os
import json

SEED = 42
np.random.seed(SEED)
os.makedirs("data/processed", exist_ok=True)

INSURANCE_CLASSES = ["dent", "scratch", "crack", "glass_shatter", "no_damage"]

# ── Templates: Passende Beschreibungen pro Schadenstyp ────────
TEMPLATES_MATCH = {
    "dent": [
        "Delle an der Fahrertür nach Parkschaden durch unbekannten Dritten.",
        "Eingedrückte Karosserie nach leichter Kollision auf dem Parkplatz.",
        "Beule an der hinteren Stossstange, Fahrzeug war geparkt.",
        "Delle im Kotflügel nach Auffahrunfall mit geringer Geschwindigkeit.",
    ],
    "scratch": [
        "Kratzer im Lack an der Seitenwand durch Schlüssel oder spitzen Gegenstand.",
        "Oberflächlicher Lackschaden beim Einparken in enger Parklücke.",
        "Schramme am Stossfänger, Farbe abgeschürft, kein struktureller Schaden.",
        "Lackschaden durch Vandalismus, langer Kratzer auf der Fahrerseite.",
    ],
    "crack": [
        "Riss im Stossfänger nach leichtem Aufprall.",
        "Gebrochenes Kunststoffteil an der vorderen Verkleidung.",
        "Riss in der Frontschürze, strukturell nicht beeinträchtigt.",
        "Bruch am Stofffänger nach Kollision mit Randstein.",
    ],
    "glass_shatter": [
        "Windschutzscheibe gesplittert durch Steinschlag auf Autobahn.",
        "Glasbruch an der Frontscheibe, Splitter im Fahrzeug.",
        "Heckscheibe zerbrochen durch Vandalismusschaden.",
        "Seitenscheibe eingeschlagen, vermutlich Einbruchsversuch.",
    ],
    "no_damage": [
        "Kein sichtbarer Schaden am Fahrzeug nach angeblichem Unfall.",
        "Fahrzeug wirkt unbeschädigt, keine Spuren einer Kollision.",
        "Überprüfung ergab keine äusserlichen Schäden am Fahrzeug.",
        "Fahrzeug in einwandfreiem Zustand, keine Beeinträchtigungen.",
    ],
}

# ── Templates: Falsche Schadenart (damage_type_mismatch) ──────
TEMPLATES_WRONG_TYPE = {
    "dent":          ["glass_shatter", "scratch"],
    "scratch":       ["glass_shatter", "dent"],
    "crack":         ["dent", "glass_shatter"],
    "glass_shatter": ["dent", "scratch"],
    "no_damage":     ["dent", "scratch"],
}

# ── Templates: Schweregrad widerspricht Bild (severity_mismatch)
TEMPLATES_SEVERITY_MISMATCH = {
    "dent": [
        "Totalschaden, Fahrzeug wirtschaftlich nicht mehr reparierbar.",
        "Massiver Frontschaden, Airbags ausgelöst, Fahrzeug nicht fahrbereit.",
    ],
    "scratch": [
        "Totalschaden nach schwerem Auffahrunfall mit Personenschaden.",
        "Fahrzeug vollständig demoliert, Totalschaden.",
    ],
    "crack": [
        "Fahrzeug komplett zerstört, Totalschaden nach Unfall.",
        "Massiver Strukturschaden, Fahrzeug muss abgeschrieben werden.",
    ],
    "glass_shatter": [
        "Kleiner Kratzer, kaum sichtbar, nur oberflächlich.",
        "Minimaler Schaden, kaum der Rede wert, kleiner Kratzer.",
    ],
    "no_damage": [
        "Schwerer Frontaufprall, Totalschaden, Fahrzeug nicht fahrbereit.",
        "Massiver Schaden an Karosserie und Motor nach Kollision.",
    ],
}

# ── Templates: Fraud-Signal-Beschreibungen ────────────────────
TEMPLATES_FRAUD_SIGNAL = {
    "dent": [
        "Kleiner Kratzer laut Kunde, kaum sichtbar — dringend Auszahlung benötigt.",
        "Fahrzeug komplett gestohlen, keine Zeugen, sofortige Entschädigung erbeten.",
    ],
    "scratch": [
        "Totalschaden nach Unfall, Fahrzeug gestohlen, keine Polizeirapport vorhanden.",
        "Massiver Schaden, bar bezahlt, keine Quittung, dringend Rückerstattung.",
    ],
    "crack": [
        "Unbekannte haben Fahrzeug gestohlen und beschädigt, keine Zeugen.",
        "Sofortige Auszahlung benötigt, keine weiteren Details verfügbar.",
    ],
    "glass_shatter": [
        "Kleiner Kratzer laut Kunde, kaum sichtbar — Totalschaden beantragt.",
        "Keine Zeugen, sofort gemeldet, dringend Entschädigung CHF 15.000.",
    ],
    "no_damage": [
        "Massiver Schaden nach Unfall mit unbekanntem Dritten, keine Zeugen.",
        "Totalschaden, gestohlen, dringend Entschädigung benötigt, bar bezahlt.",
    ],
}

# ── Severity Mapping ──────────────────────────────────────────
SEVERITY_MAP = {
    "dent":          "medium",
    "scratch":       "minor",
    "crack":         "minor",
    "glass_shatter": "medium",
    "no_damage":     "none",
}


def build_dataset(labels_csv: str = "data/raw/insurance_labels_balanced.csv",
                  n_per_class: int = 50) -> pd.DataFrame:
    """
    Erstellt Consistency-Dataset mit 4 Fällen pro Bild:
    - Fall A: Konsistente Beschreibung (label=1)
    - Fall B: Falsche Schadenart (label=0, mismatch=damage_type_mismatch)
    - Fall C: Falscher Schweregrad (label=0, mismatch=severity_mismatch)
    - Fall D: Fraud-Signal (label=0, mismatch=fraud_signal)
    """
    # Labels laden
    if os.path.exists(labels_csv):
        df_labels = pd.read_csv(labels_csv)
        label_col = "label" if "label" in df_labels.columns else "label_final"
    else:
        print(f"Labels CSV nicht gefunden: {labels_csv}")
        print("Erzeuge synthetisches Dataset aus Klassen-Definitionen...")
        rows = []
        for cls in INSURANCE_CLASSES:
            for i in range(n_per_class):
                rows.append({"idx": i, label_col: cls})
        df_labels = pd.DataFrame(rows)
        label_col = "label"

    records = []
    for cls in INSURANCE_CLASSES:
        subset = df_labels[df_labels[label_col] == cls].head(n_per_class)
        if len(subset) == 0:
            print(f"  Keine Samples für Klasse: {cls}")
            continue

        for _, row in subset.iterrows():
            idx = row.get("idx", 0)

            # Fall A: Konsistent
            desc_match = np.random.choice(TEMPLATES_MATCH[cls])
            records.append({
                "idx":               int(idx),
                "true_damage_label": cls,
                "description":       desc_match,
                "consistency_label": 1,
                "mismatch_type":     "none",
                "severity_label":    SEVERITY_MAP[cls],
            })

            # Fall B: Falsche Schadenart
            wrong_cls = np.random.choice(TEMPLATES_WRONG_TYPE[cls])
            desc_wrong = np.random.choice(TEMPLATES_MATCH[wrong_cls])
            records.append({
                "idx":               int(idx),
                "true_damage_label": cls,
                "description":       desc_wrong,
                "consistency_label": 0,
                "mismatch_type":     "damage_type_mismatch",
                "severity_label":    SEVERITY_MAP[cls],
            })

            # Fall C: Falscher Schweregrad
            desc_sev = np.random.choice(TEMPLATES_SEVERITY_MISMATCH[cls])
            records.append({
                "idx":               int(idx),
                "true_damage_label": cls,
                "description":       desc_sev,
                "consistency_label": 0,
                "mismatch_type":     "severity_mismatch",
                "severity_label":    SEVERITY_MAP[cls],
            })

            # Fall D: Fraud Signal
            desc_fraud = np.random.choice(TEMPLATES_FRAUD_SIGNAL[cls])
            records.append({
                "idx":               int(idx),
                "true_damage_label": cls,
                "description":       desc_fraud,
                "consistency_label": 0,
                "mismatch_type":     "fraud_signal",
                "severity_label":    SEVERITY_MAP[cls],
            })

    df = pd.DataFrame(records).sample(frac=1, random_state=SEED).reset_index(drop=True)
    return df


if __name__ == "__main__":
    print("Erstelle Multimodal Consistency Dataset...")
    df = build_dataset(n_per_class=50)

    out_path = "data/processed/multimodal_consistency_cases.csv"
    df.to_csv(out_path, index=False)

    print(f"\nDataset erstellt: {len(df)} Faelle")
    print(f"Gespeichert: {out_path}")
    print(f"\nVerteilung consistency_label:")
    print(df["consistency_label"].value_counts())
    print(f"\nVerteilung mismatch_type:")
    print(df["mismatch_type"].value_counts())
    print(f"\nVerteilung true_damage_label:")
    print(df["true_damage_label"].value_counts())
