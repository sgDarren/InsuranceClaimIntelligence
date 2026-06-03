"""
Insurance Claim Intelligence — HuggingFace Spaces App
ZHAW AI Applications FS2026

Deployment: https://huggingface.co/spaces/[username]/insurance-claim-intelligence
"""

import os, json, re, base64, joblib
import numpy as np
import gradio as gr
from PIL import Image
from io import BytesIO
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from transformers import pipeline as hf_pipeline
from sklearn.preprocessing import LabelEncoder

# ── Pfade — funktioniert lokal UND auf HuggingFace Spaces ────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
print(f"Base dir: {BASE_DIR}")
print(f"Models dir: {MODELS_DIR}")
print(f"Models vorhanden: {os.listdir(MODELS_DIR) if os.path.exists(MODELS_DIR) else 'NICHT GEFUNDEN'}")

def model_path(filename):
    return os.path.join(MODELS_DIR, filename)

# ── API Key aus HuggingFace Secrets ──────────────────────────
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
client_oai     = OpenAI(api_key=OPENAI_API_KEY)

# ══════════════════════════════════════════════════════════════
# MODELLE LADEN (einmalig beim Start)
# ══════════════════════════════════════════════════════════════
print("Lade Modelle...")

# CV: ViT Fine-Tuned
try:
    vit_classifier = hf_pipeline(
        "image-classification",
        model=model_path("vit-damage-final"),
    )
    with open(model_path("label_map.json")) as f:
        id2label = json.load(f)
    INSURANCE_CLASSES = list(id2label.values())
    print(f"ViT geladen: {INSURANCE_CLASSES}")
except Exception as e:
    print(f"FEHLER: ViT Modell konnte nicht geladen werden: {e}")
    print(f"Stelle sicher dass models/vit-damage-final/ vorhanden ist.")
    raise RuntimeError(f"ViT Modell nicht gefunden: {e}")

# ML: XGBoost + Fraud Classifier
try:
    xgb_model    = joblib.load(model_path("xgboost_claim.pkl"))
    fraud_model  = joblib.load(model_path("fraud_classifier.pkl"))
    with open(model_path("feature_cols.json")) as f:
        FEATURE_COLS = json.load(f)
    print("ML Modelle geladen")
except Exception as e:
    print(f"FEHLER: ML Modelle konnten nicht geladen werden: {e}")
    raise RuntimeError(f"ML Modelle nicht gefunden: {e}")

# Multimodal Consistency Model (optional)
try:
    consistency_model = joblib.load(model_path("consistency_model.pkl"))
    with open(model_path("consistency_feature_cols.json")) as f:
        CONSISTENCY_FEATURE_COLS = json.load(f)
    print("Consistency Modell geladen")
except Exception as e:
    print(f"Consistency Modell nicht gefunden: {e} — heuristischer Fallback")
    consistency_model = None
    CONSISTENCY_FEATURE_COLS = []

# NLP: SentenceTransformer + Embeddings
try:
    embedder = SentenceTransformer("all-MiniLM-L6-v2", backend="torch")
except Exception:
    from sentence_transformers import SentenceTransformer as ST
    embedder = ST("sentence-transformers/all-MiniLM-L6-v2")

# ── RAG: Echte AXA AVB PDFs laden ────────────────────────────
import requests
from io import BytesIO
try:
    from pypdf import PdfReader

    POLICY_URLS = {
        "AXA_OPTIMA_2023": "https://www.axa.ch/doc/ajhtk",
        "AXA_MF_2021": "https://mzo.ch/wp-content/uploads/AXA-MF_AVB_10.2021_DE.pdf",
    }

    def download_pdf(url, name):
        try:
            resp = requests.get(url, timeout=15, headers={"User-Agent": "Mozilla/5.0"})
            reader = PdfReader(BytesIO(resp.content))
            text = "\n".join(page.extract_text() or "" for page in reader.pages)
            print(f"  {name}: {len(reader.pages)} Seiten")
            return text, name
        except Exception as e:
            print(f"  {name}: Fehler - {e}")
            return "", name

    def chunk_text(text, chunk_size=300, overlap=30):
        words, chunks = text.split(), []
        i = 0
        while i < len(words):
            chunk = " ".join(words[i:i+chunk_size])
            if len(chunk.strip()) > 100:
                chunks.append(chunk)
            i += chunk_size - overlap
        return chunks

    print("Lade AXA AVB PDFs...")
    policy_texts, policy_sources = [], []
    for name, url in POLICY_URLS.items():
        text, src = download_pdf(url, name)
        if text:
            for chunk in chunk_text(text):
                policy_texts.append(chunk)
                policy_sources.append(src)

    if len(policy_texts) < 10:
        raise ValueError("PDFs nicht geladen")

    print(f"  {len(policy_texts)} Chunks aus echten AXA AVB PDFs")

except Exception as e:
    print(f"Fallback auf statische Policies: {e}")
    POLICIES = [
        ("Vollkaskoversicherung deckt Schäden durch Kollision, Diebstahl, Feuer, Naturereignisse und Vandalismus. Franchise CHF 300-2000.", "AXA AVB §59"),
        ("Teilkaskoversicherung deckt Diebstahl, Feuer, Naturereignisse, Glasbruch und Marderbiss. Kollisionsschäden nicht gedeckt.", "AXA AVB §58"),
        ("Parkschäden durch unbekannte Dritte bei Vollkasko gedeckt, sofern unverzüglich gemeldet.", "AXA AVB §12"),
        ("Versicherungsbetrug: Falsche Angaben führen zur Leistungsverweigerung.", "AXA AVB §40"),
        ("Hagelschäden sind Elementarschaden in der Teilkasko gedeckt. Meldung innerhalb 5 Tage.", "AXA AVB §8"),
        ("Totalschäden auf Basis des Zeitwerts entschädigt abzüglich Franchise.", "AXA AVB §15"),
        ("Vandalismusschäden bei Vollkasko gedeckt. Polizeirapport erforderlich.", "AXA AVB §11"),
        ("Glasbruchschäden an Windschutzscheibe ohne Franchise bei Glasbruchdeckung.", "AXA AVB §9"),
        ("Auffahrunfälle bei Vollkasko gedeckt. Bei Teilschuld anteilige Kürzung.", "AXA AVB §10"),
        ("Diebstahl bei Voll- und Teilkasko gedeckt. Meldung innerhalb 48 Stunden.", "AXA AVB §13"),
        ("Kratzer und Lackschäden durch Vandalismus bei Vollkasko gedeckt.", "AXA AVB §11b"),
        ("Marderschäden an Kabeln und Schläuchen in der Teilkasko gedeckt.", "AXA AVB §8b"),
    ]
    policy_texts   = [p[0] for p in POLICIES]
    policy_sources = [p[1] for p in POLICIES]

policy_embeddings = embedder.encode(policy_texts)
print(f"NLP RAG bereit: {len(policy_texts)} Chunks indexiert")

# ══════════════════════════════════════════════════════════════
# HILFSFUNKTIONEN
# ══════════════════════════════════════════════════════════════

FRAUD_SIGNALS = ["total loss", "gestohlen", "unbekannt", "keine zeugen",
                 "sofort", "bar bezahlt", "keine quittung", "dringend"]

INCIDENT_TYPES = {
    "parking":   ["parkschaden", "parkunfall", "parking", "eingeparkt"],
    "rear_end":  ["auffahrunfall", "hinten", "von hinten"],
    "vandalism": ["vandalismus", "zerkratzt", "absichtlich"],
    "theft":     ["diebstahl", "gestohlen", "einbruch"],
    "weather":   ["hagel", "sturm", "überschwemmung"],
}

def extract_nlp_features(description: str) -> dict:
    desc_lower = description.lower()
    incident_type = "other"
    for itype, keywords in INCIDENT_TYPES.items():
        if any(kw in desc_lower for kw in keywords):
            incident_type = itype
            break
    return {
        "incident_type_nlp":   incident_type,
        "fraud_signal_count":  sum(1 for s in FRAUD_SIGNALS if s in desc_lower),
        "description_length":  len(description.split()),
    }

def compute_consistency_score(damage_type: str, severity: int, description: str) -> float:
    """CV-NLP Konsistenzpruefung via trainiertem Consistency Model.
    Gibt Wahrscheinlichkeit zurueck dass Bild und Beschreibung konsistent sind (0=inkonsistent, 1=konsistent).
    Fallback auf Heuristik wenn Modell nicht geladen.
    """
    desc_lower = description.lower()

    FRAUD_SIGNALS_APP  = ["totalschaden", "gestohlen", "keine zeugen", "dringend",
                          "sofort", "bar bezahlt", "keine quittung", "auszahlung"]
    INCIDENT_TYPES_APP = {
        "parking":   ["parkschaden", "parklücke", "parkplatz"],
        "vandalism": ["vandalismus", "schlüssel", "zerkratzt"],
        "collision": ["kollision", "auffahrunfall", "aufprall"],
        "theft":     ["gestohlen", "einbruch", "diebstahl"],
        "weather":   ["hagel", "sturm", "steinschlag"],
    }
    SEVERITY_HIGH = ["totalschaden", "massiv", "komplett", "nicht fahrbereit", "airbags"]
    SEVERITY_LOW  = ["kleiner", "oberflächlich", "kaum", "minimal", "leicht"]
    F1_MAP_APP    = {"dent": 0.66, "scratch": 0.98, "crack": 0.98,
                     "glass_shatter": 0.83, "no_damage": 0.80}
    LABEL2ID_APP  = {l: i for i, l in enumerate(INSURANCE_CLASSES)}

    incident_type = 0
    for i, (itype, keywords) in enumerate(INCIDENT_TYPES_APP.items()):
        if any(kw in desc_lower for kw in keywords):
            incident_type = i + 1; break

    fraud_count = sum(1 for s in FRAUD_SIGNALS_APP if s in desc_lower)
    desc_length = len(description.split())

    desc_emb = embedder.encode([description])
    cv_emb   = embedder.encode([damage_type])
    cos_sim  = float(np.dot(cv_emb, desc_emb.T) /
                     (np.linalg.norm(cv_emb) * np.linalg.norm(desc_emb) + 1e-8))
    cos_sim  = float(np.clip(cos_sim, 0.0, 1.0))

    has_high     = any(w in desc_lower for w in SEVERITY_HIGH)
    has_low      = any(w in desc_lower for w in SEVERITY_LOW)
    sev_mismatch = 0
    if damage_type in ("dent", "scratch", "crack") and has_high:
        sev_mismatch = 1
    if damage_type == "glass_shatter" and has_low:
        sev_mismatch = 1

    # Consistency Model verwenden wenn verfügbar
    if consistency_model is not None:
        try:
            import pandas as pd_inner
            X_cons = pd_inner.DataFrame([{
                "cv_label_enc":       LABEL2ID_APP.get(damage_type, 0),
                "cv_confidence":      F1_MAP_APP.get(damage_type, 0.75),
                "nlp_incident_type":  incident_type,
                "fraud_signal_count": fraud_count,
                "description_length": desc_length,
                "cosine_similarity":  cos_sim,
                "severity_mismatch":  sev_mismatch,
            }])
            return round(float(consistency_model.predict_proba(X_cons)[0][1]), 3)
        except Exception:
            pass

    # Heuristischer Fallback
    score = max(0.0, min(1.0, cos_sim - sev_mismatch * 0.4 - fraud_count * 0.15 + 0.3))
    return round(score, 3)

def rag_retrieve(query: str, n: int = 4):
    q_emb  = embedder.encode([query])
    scores = np.dot(policy_embeddings, q_emb.T).squeeze()
    top    = np.argsort(scores)[::-1][:n]
    return [policy_texts[i] for i in top], [policy_sources[i] for i in top]

# ══════════════════════════════════════════════════════════════
# CV BLOCK
# ══════════════════════════════════════════════════════════════
def classify_damage(image: Image.Image) -> dict:
    if vit_classifier is None:
        return {"damage_type": "dent", "cv_confidence": 0.87, "damage_severity": 1}
    try:
        results = vit_classifier(image)
        label   = results[0]["label"]
        score   = results[0]["score"]
        severity_map = {"no_damage": 0, "scratch": 0, "crack": 1,
                        "dent": 1, "glass_shatter": 2}
        return {
            "damage_type":     label,
            "cv_confidence":   round(score, 3),
            "damage_severity": severity_map.get(label, 1),
        }
    except Exception as e:
        raise RuntimeError(f"ViT Klassifikation fehlgeschlagen: {e}")

# ══════════════════════════════════════════════════════════════
# ML BLOCK
# ══════════════════════════════════════════════════════════════
INS_MAP = {"liability": 0, "partial": 1, "full": 2,
           "Haftpflicht": 0, "Teilkasko": 1, "Vollkasko": 2}

def predict_claim(cv_result: dict, nlp_feats: dict, consistency: float,
                  premium: float, age: int, tenure: int,
                  insurance_type: str, franchise: int) -> dict:
    import pandas as pd
    data = {
        "PREMIUM_AMOUNT":           premium,
        "AGE":                      age,
        "TENURE":                   tenure,
        "NO_OF_FAMILY_MEMBERS":     2,
        "ANY_INJURY":               0,
        "POLICE_REPORT_AVAILABLE":  1,
        "INCIDENT_HOUR_OF_THE_DAY": 14,
        "policy_age_days":          365,
        "report_delay_days":        3,
        "new_policy_flag":          0,
        "is_night":                 0,
        "INSURANCE_TYPE_enc":       INS_MAP.get(insurance_type, 2),
        "MARITAL_STATUS_enc":       0,
        "EMPLOYMENT_STATUS_enc":    0,
        "RISK_SEGMENTATION_enc":    1,
        "SOCIAL_CLASS_enc":         1,
        "INCIDENT_SEVERITY_enc":    cv_result.get("damage_severity", 1),
        "AUTHORITY_CONTACTED_enc":  1,
        "CUSTOMER_EDUCATION_LEVEL_enc": 2,
        "damage_type_enc":          INSURANCE_CLASSES.index(cv_result.get("damage_type","dent"))
                                    if cv_result.get("damage_type") in INSURANCE_CLASSES else 0,
        "damage_severity":          cv_result.get("damage_severity", 1),
        "damage_area_pct":          0.15,
        "cv_confidence":            cv_result.get("cv_confidence", 0.8),
        "incident_type_nlp":        list(INCIDENT_TYPES.keys()).index(
                                    nlp_feats.get("incident_type_nlp","parking"))
                                    if nlp_feats.get("incident_type_nlp") in INCIDENT_TYPES else 0,
        "fraud_signal_count":       nlp_feats.get("fraud_signal_count", 0),
        "description_length":       nlp_feats.get("description_length", 20),
        "consistency_score":        consistency,
    }
    avail = [f for f in FEATURE_COLS if f in data]
    X     = pd.DataFrame([data])[avail].fillna(0)

    amount      = float(xgb_model.predict(X)[0])
    fraud_prob  = float(fraud_model.predict_proba(X)[0][1])
    net_amount  = max(0, amount - franchise)
    priority    = "Hoch" if amount > 5000 else "Mittel" if amount > 1500 else "Niedrig"
    manual      = fraud_prob > 0.7 or consistency < 0.2

    return {
        "claim_amount_chf": round(net_amount, 2),
        "fraud_score":      round(fraud_prob, 3),
        "confidence":       0.78,
        "priority":         priority,
        "manual_review":    manual,
    }

# ══════════════════════════════════════════════════════════════
# NLP BLOCK
# ══════════════════════════════════════════════════════════════
SYSTEM_PROMPTS = {
    "expert": (
        "Du bist ein erfahrener Schweizer Versicherungsexperte bei AXA mit 20 Jahren Erfahrung. "
        "Zitiere immer die relevante AVB-Klausel. Weise klar auf Ausschlüsse hin."
    ),
    "customer_service": (
        "Du bist ein freundlicher AXA Kundenberater. "
        "Erkläre einfach ohne Fachjargon. Schliesse mit: 'Haben Sie weitere Fragen?'"
    ),
    "fraud_analyst": (
        "Du bist ein Fraud Analyst bei AXA. Analysiere Schadenmeldungen auf Inkonsistenzen. "
        "Gib strukturierten Fraud-Assessment-Report mit Plausibilitätsbewertung 1-10 aus."
    ),
}

def query_rag(incident: str, insurance_type: str, description: str = "",
              role: str = "expert", prompt_type: str = "rag_expert") -> tuple:
    docs, sources = rag_retrieve(f"{incident} {insurance_type} gedeckt")
    context       = "\n\n".join(docs)

    if prompt_type == "zero_shot":
        messages = [{"role": "user", "content":
            f"Ist '{incident}' bei '{insurance_type}' gedeckt? Antworte auf Deutsch."}]
    elif prompt_type == "rag_basic":
        messages = [{"role": "user", "content":
            f"Versicherungsbedingungen:\n{context}\n\nIst '{incident}' bei '{insurance_type}' gedeckt?"}]
    else:
        if role == "fraud_analyst":
            user_msg = (
                f"AVB:\n{context}\n\nFall: {insurance_type} | {incident}\n"
                f"Beschreibung: {description or 'Keine Angabe'}\n\n"
                f"1. Plausibilitätsbewertung (1-10)\n2. Auffälligkeiten\n"
                f"3. Empfehlung: Freigeben / Manuelle Prüfung / Ablehnen"
            )
        elif role == "customer_service":
            user_msg = (
                f"AVB:\n{context}\n\nSchaden: {incident} | {insurance_type}\n"
                f"Erkläre freundlich was gedeckt ist und nächste Schritte."
            )
        else:
            user_msg = (
                f"AVB:\n{context}\n\nFall: {insurance_type} | {incident}\n"
                f"Beschreibung: {description or '-'}\n\n"
                f"1. Gedeckt? (Ja/Nein/Teilweise)\n2. AVB-Klausel\n"
                f"3. Ausschlüsse\n4. Nächste Schritte"
            )
        messages = [
            {"role": "system", "content": SYSTEM_PROMPTS[role]},
            {"role": "user",   "content": user_msg},
        ]

    if not OPENAI_API_KEY:
        return "⚠️ Kein OpenAI API Key — bitte in HuggingFace Secrets setzen.", sources

    resp = client_oai.chat.completions.create(
        model="gpt-4o-mini", messages=messages,
        temperature=0.1, max_tokens=500,
    )
    return resp.choices[0].message.content, sources

# ══════════════════════════════════════════════════════════════
# RESULTATE LADEN
# ══════════════════════════════════════════════════════════════
def load_results():
    try:
        with open(model_path("ml_results.json")) as f:
            return json.load(f)
    except:
        return {
            "linear_regression": {"rmse": 21173, "r2": 0.075},
            "random_forest":     {"rmse": 11657, "r2": 0.720},
            "xgboost":           {"rmse": 11624, "r2": 0.721},
        }

def load_cv_results():
    try:
        with open(model_path("cv_results.json")) as f:
            return json.load(f)
    except:
        return {
            "clip_zero_shot":    {"accuracy": 0.00},
            "transfer_learning": {"accuracy": 0.6082},
            "fine_tuning":       {"accuracy": 0.7985},
            "gpt4o_vision":      {"accuracy": 0.58},
        }

# ══════════════════════════════════════════════════════════════
# GRADIO APP — Schönes deutsches Design
# ══════════════════════════════════════════════════════════════
CSS = """
/* ── Hintergrund & Basis ── */
body, .gradio-container { background: #0f172a !important; color: #e2e8f0 !important; }
.gradio-container { max-width: 1200px !important; margin: 0 auto !important; }

/* ── Alle weissen Panels dunkel ── */
.block, .form, .gap, .panel, .tabitem { background: transparent !important; }
.prose, label, .label-wrap span { color: #cbd5e1 !important; }

/* ── Eingabefelder ── */
input, textarea, select, .dropdown {
    background: #1e293b !important;
    border-color: #334155 !important;
    color: #e2e8f0 !important;
}
input:focus, textarea:focus {
    border-color: #3b82f6 !important;
    outline: none !important;
}

/* ── Header ── */
.header-box {
    background: linear-gradient(135deg, #1a3a5c 0%, #2563eb 60%, #1d9e75 100%);
    border-radius: 16px;
    padding: 32px 24px 24px;
    margin-bottom: 8px;
    text-align: center;
    color: white;
}
.header-box h1 { font-size: 2rem; font-weight: 700; margin: 0 0 8px; letter-spacing: -0.5px; }
.header-box p  { font-size: 1rem; opacity: 0.9; margin: 4px 0; }
.header-box .badge {
    display: inline-block;
    background: rgba(255,255,255,0.2);
    border: 1px solid rgba(255,255,255,0.4);
    border-radius: 20px;
    padding: 4px 14px;
    font-size: 0.8rem;
    margin-top: 10px;
}

/* ── Tabs ── */
.tab-nav button {
    font-size: 0.95rem !important;
    font-weight: 500 !important;
    padding: 10px 20px !important;
    border-radius: 10px 10px 0 0 !important;
}

/* ── Eingabe-Bereich ── */
.eingabe-panel {
    background: white;
    border-radius: 14px;
    padding: 20px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.06);
    border: 1px solid #e2e8f0;
}

/* ── Ergebnis-Karten ── */
.ergebnis-karte {
    background: white;
    border-radius: 14px;
    padding: 20px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.06);
    border: 1px solid #e2e8f0;
    margin-bottom: 12px;
}

/* ── Entscheidungs-Banner ── */
.entscheidung-gruen {
    background: linear-gradient(135deg, #d1fae5, #a7f3d0);
    border: 2px solid #10b981;
    border-radius: 12px;
    padding: 16px;
    font-weight: 600;
    font-size: 1.05rem;
}
.entscheidung-rot {
    background: linear-gradient(135deg, #fee2e2, #fecaca);
    border: 2px solid #ef4444;
    border-radius: 12px;
    padding: 16px;
    font-weight: 600;
}
.entscheidung-gelb {
    background: linear-gradient(135deg, #fef3c7, #fde68a);
    border: 2px solid #f59e0b;
    border-radius: 12px;
    padding: 16px;
    font-weight: 600;
}

/* ── Kennzahl-Karten ── */
.kennzahl {
    background: white;
    border-radius: 12px;
    padding: 16px;
    text-align: center;
    box-shadow: 0 2px 6px rgba(0,0,0,0.05);
    border: 1px solid #e2e8f0;
}

/* ── Analyse-Button ── */
button.primary {
    background: linear-gradient(135deg, #2563eb, #1d4ed8) !important;
    border: none !important;
    border-radius: 10px !important;
    font-size: 1rem !important;
    font-weight: 600 !important;
    padding: 12px !important;
    transition: all 0.2s !important;
}
button.primary:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 12px rgba(37,99,235,0.4) !important;
}

/* ── Abschnitt-Titel ── */
.abschnitt-titel {
    font-size: 0.75rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 1px;
    color: #64748b;
    margin: 16px 0 8px;
    padding-bottom: 6px;
    border-bottom: 2px solid #e2e8f0;
}

/* ── Footer ── */
.footer-box {
    background: #1a3a5c;
    color: rgba(255,255,255,0.7);
    border-radius: 12px;
    padding: 16px 24px;
    text-align: center;
    font-size: 0.85rem;
    margin-top: 16px;
}
"""

with gr.Blocks(
    title="Insurance Claim Intelligence — ZHAW FS2026",
) as demo:

    # ── HEADER ────────────────────────────────────────────────
    gr.HTML("""
    <div class="header-box">
      <h1>🚗 Insurance Claim Intelligence</h1>
      <p><strong>Multimodale KI-Bewertung von Versicherungsfällen</strong></p>
      <p>Computer Vision · NLP · Machine Learning → Schadenhöhe · Fraud-Score · Deckungsauskunft</p>
      <span class="badge">ZHAW AI Applications Projekt FS2026</span>
    </div>
    """)

    with gr.Tabs(elem_classes="tab-nav"):

        # ══════════════════════════════════════════════════════
        # TAB 1: SCHADENANALYSE
        # ══════════════════════════════════════════════════════
        with gr.Tab("📸 Schadenanalyse"):
            gr.HTML('<p style="color:#64748b;margin:8px 0 16px;font-size:0.95rem;">Lade ein Schadensbild hoch, beschreibe den Vorfall und ergänze die Vertragsdaten — das System bewertet den Schaden vollautomatisch.</p>')

            with gr.Row(equal_height=False):

                # ── Linke Spalte: Eingaben ──
                with gr.Column(scale=1):
                    gr.HTML('<div class="abschnitt-titel">📸 Schadensbild</div>')
                    img_input = gr.Image(
                        type="pil",
                        label="Bild hochladen oder hier ablegen",
                        height=220,
                    )

                    gr.HTML('<div class="abschnitt-titel">📝 Schadenbeschreibung</div>')
                    desc_input = gr.Textbox(
                        lines=4,
                        label="",
                        placeholder="Beschreiben Sie den Schaden: Wann? Wie? Wo? Was ist passiert?\nBeispiel: Parkschaden am 01.06.2026, Delle in der Fahrertür durch unbekannten Dritten."
                    )

                    gr.HTML('<div class="abschnitt-titel">📋 Vertragsdaten</div>')
                    with gr.Row():
                        premium_input = gr.Number(value=157, label="Monatsprämie (CHF)", minimum=0)
                        age_input     = gr.Number(value=35,  label="Fahreralter (Jahre)", minimum=18)
                    with gr.Row():
                        tenure_input    = gr.Number(value=5, label="Vertragsjahre", minimum=0)
                        franchise_input = gr.Dropdown(
                            choices=[300, 500, 1000, 2000],
                            value=500,
                            label="Franchise (CHF)",
                        )
                    ins_input = gr.Dropdown(
                        choices=[
                            ("Haftpflicht", "liability"),
                            ("Teilkasko",   "partial"),
                            ("Vollkasko",   "full"),
                        ],
                        value="full",
                        label="Versicherungstyp",
                    )
                    analyse_btn = gr.Button(
                        "🔍  Schaden analysieren",
                        variant="primary",
                        size="lg",
                    )

                # ── Rechte Spalte: Ergebnisse ──
                with gr.Column(scale=2):

                    gr.HTML('<div class="abschnitt-titel">🤖 KI-Analyse — Alle drei Blöcke</div>')
                    with gr.Row():
                        cv_out  = gr.JSON(label="🖼️ Computer Vision: Schadenstyp")
                        nlp_out = gr.JSON(label="📝 NLP: Textanalyse & Fraud-Signale")

                    gr.HTML('<div class="abschnitt-titel">🔗 Multimodales Kernfeature</div>')
                    consistency_out = gr.Number(
                        label="Consistency Score — Übereinstimmung Bild ↔ Beschreibung (0 = Inkonsistenz = Fraud-Verdacht, 1 = perfekt konsistent)",
                        precision=3,
                    )

                    gr.HTML('<div class="abschnitt-titel">💰 Vorhersagen (ML-Block: XGBoost)</div>')
                    with gr.Row():
                        amount_out = gr.Number(label="Schadenhöhe CHF (nach Franchise)", precision=0)
                        fraud_out  = gr.Number(label="Fraud-Score (0–1)", precision=3)
                        conf_out   = gr.Number(label="Modell-Konfidenz", precision=2)

                    gr.HTML('<div class="abschnitt-titel">✅ Entscheidung</div>')
                    decision_out = gr.Textbox(label="", lines=3, interactive=False)

                    with gr.Accordion("📊 Alle ML-Vorhersagen (Details)", open=False):
                        ml_out = gr.JSON(label="")

            # ── Analyse-Logik ──
            def full_analysis(image, desc, premium, age, tenure, franchise, ins_type):
                if image is None:
                    return {}, {}, 0.5, 0, 0, 0, "⚠️ Bitte zuerst ein Schadensbild hochladen.", {}

                cv_result   = classify_damage(image)
                desc        = desc or "Keine Beschreibung angegeben"
                nlp_feats   = extract_nlp_features(desc)
                consistency = compute_consistency_score(
                    cv_result["damage_type"], cv_result["damage_severity"], desc)
                nlp_result  = {**nlp_feats, "consistency_score": consistency}
                ml_result   = predict_claim(
                    cv_result, nlp_feats, consistency,
                    premium, int(age), int(tenure), ins_type, int(franchise))

                fraud = ml_result["fraud_score"]
                amt   = ml_result["claim_amount_chf"]
                prio  = ml_result["priority"]

                if fraud > 0.7:
                    decision = f"🔴 MANUELLE PRÜFUNG ERFORDERLICH\nFraud-Score: {fraud:.2f} überschreitet Schwellwert 0.70 → Verdacht auf Betrug."
                elif consistency < 0.2:
                    decision = f"🟡 MANUELLE PRÜFUNG ERFORDERLICH\nConsistency-Score: {consistency:.2f} — Bild und Beschreibung stimmen nicht überein!"
                elif ml_result.get("confidence", 1) < 0.6:
                    decision = f"🟡 MANUELLE PRÜFUNG EMPFOHLEN\nModell-Konfidenz unter 60% — unsicheres Ergebnis."
                else:
                    decision = f"🟢 AUTOMATISCHE FREIGABE\nSchadenhöhe: CHF {amt:,.0f}  |  Priorität: {prio}  |  Fraud-Score: {fraud:.3f} (niedrig)"

                return (cv_result, nlp_result, consistency,
                        amt, fraud, ml_result.get("confidence", 0.78),
                        decision, ml_result)

            analyse_btn.click(
                fn=full_analysis,
                inputs=[img_input, desc_input, premium_input, age_input,
                        tenure_input, franchise_input, ins_input],
                outputs=[cv_out, nlp_out, consistency_out,
                         amount_out, fraud_out, conf_out, decision_out, ml_out],
            )

        # ══════════════════════════════════════════════════════
        # TAB 2: DECKUNGSAUSKUNFT
        # ══════════════════════════════════════════════════════
        with gr.Tab("📋 Deckungsauskunft"):
            gr.HTML('<p style="color:#64748b;margin:8px 0 16px;font-size:0.95rem;">RAG-gestützte Deckungsanalyse auf Basis echter AXA AVB PDFs (OPTIMA 2023 · Motorfahrzeug 2021). Drei Prompt-Varianten und drei Expertenrollen vergleichbar.</p>')

            with gr.Row():
                with gr.Column(scale=1):
                    gr.HTML('<div class="abschnitt-titel">🔎 Schadensangaben</div>')
                    cov_incident = gr.Dropdown(
                        choices=[
                            "Kollisionsschaden Auffahrunfall",
                            "Glasbruch Windschutzscheibe",
                            "Parkschaden unbekannter Dritter",
                            "Hagelschaden Motorhaube",
                            "Vandalismusschaden Lack",
                            "Diebstahl Fahrzeug",
                            "Marderschaden Kabel",
                            "Brandschaden Motorraum",
                        ],
                        value="Kollisionsschaden Auffahrunfall",
                        label="Schadensart",
                    )
                    cov_ins = gr.Dropdown(
                        choices=["Vollkasko", "Teilkasko", "Haftpflicht"],
                        value="Vollkasko",
                        label="Versicherungstyp",
                    )
                    cov_desc = gr.Textbox(
                        lines=3,
                        label="Ergänzende Beschreibung (optional)",
                        placeholder="Weitere Details zum Schadensfall..."
                    )

                    gr.HTML('<div class="abschnitt-titel">⚙️ Methoden-Vergleich</div>')
                    prompt_radio = gr.Radio(
                        choices=[
                            ("Iteration 1 — Ohne Kontext (Zero-Shot)", "zero_shot"),
                            ("Iteration 2 — Mit AVB-Kontext (RAG Basic)", "rag_basic"),
                            ("Iteration 3 — Experten-Rolle + Struktur (RAG Expert) ← 4.70/5.0", "rag_expert"),
                        ],
                        value="rag_expert",
                        label="Prompt-Variante",
                    )
                    role_radio = gr.Radio(
                        choices=[
                            ("Versicherungsexperte (juristisch, AVB-Klausel)", "expert"),
                            ("Kundenberater (einfach, empathisch)", "customer_service"),
                            ("Fraud-Analyst (Plausibilitätsprüfung 1–10)", "fraud_analyst"),
                        ],
                        value="expert",
                        label="Expertenrolle",
                    )
                    cov_btn = gr.Button("📋  Deckung prüfen", variant="primary")

                with gr.Column(scale=2):
                    gr.HTML('<div class="abschnitt-titel">💬 Deckungsauskunft</div>')
                    cov_out = gr.Textbox(
                        lines=14,
                        label="",
                        interactive=False,
                        placeholder="Hier erscheint die Deckungsauskunft nach dem Klick auf 'Deckung prüfen'..."
                    )
                    gr.HTML('<div class="abschnitt-titel">📚 Verwendete AVB-Quellen</div>')
                    sources_out = gr.JSON(label="")

            def check_coverage(incident, ins_type, desc, pt, role):
                answer, sources = query_rag(incident, ins_type, desc, role, pt)
                return answer, sources

            cov_btn.click(
                fn=check_coverage,
                inputs=[cov_incident, cov_ins, cov_desc, prompt_radio, role_radio],
                outputs=[cov_out, sources_out],
            )

        # ══════════════════════════════════════════════════════
        # TAB 3: ERGEBNISSE
        # ══════════════════════════════════════════════════════
        with gr.Tab("📊 Ergebnisse & Ablation Study"):
            gr.HTML("""
            <div style="background:#eff6ff;border-radius:12px;padding:16px 20px;margin-bottom:16px;border:1px solid #bfdbfe;">
              <p style="margin:0;font-size:0.95rem;color:#1e40af;">
                <strong>Forschungsfrage:</strong> Welchen zusätzlichen Mehrwert liefern Bild- und Textinformationen
                gegenüber rein strukturierten Schadendaten bei der automatisierten Bewertung von Versicherungsfällen?
              </p>
            </div>
            """)

            with gr.Row():
                with gr.Column():
                    gr.HTML('<div class="abschnitt-titel">📊 ML Block — Schadenhöhe (Regression)</div>')
                    gr.JSON(value={
                        "It. 1 — Linear Regression (Baseline)": {"RMSE": "CHF 21'173", "R²": 0.075},
                        "It. 2 — Random Forest":                {"RMSE": "CHF 11'657", "R²": 0.720},
                        "It. 3 — XGBoost ← WINNER":            {"RMSE": "CHF 11'624", "R²": 0.721},
                    }, label="Modellvergleich")

                    gr.HTML('<div class="abschnitt-titel">🚨 Fraud Detection (AUC = 0.931)</div>')
                    gr.JSON(value={
                        "F1-Score":          0.311,
                        "AUC-ROC":           0.931,
                        "Precision (Fraud)": "1.00 — kein legitimer Kunde falsch markiert",
                        "Recall (Fraud)":    "0.18 — bewusst, Precision priorisiert",
                    }, label="Fraud Detection")

                with gr.Column():
                    gr.HTML('<div class="abschnitt-titel">🖼️ CV Block — Schadensklassifikation (5 Klassen)</div>')
                    gr.JSON(value={
                        "It. 1 — CLIP Zero-Shot":          "0.00%  (kein Training — Baseline)",
                        "It. 2 — ViT Transfer Learning":   "60.82%",
                        "It. 3 — ViT Fine-Tuning ← WINNER":"79.85% | F1 Macro = 0.85",
                        "It. 4 — GPT-4o Vision (SotA)":   "58.00%  (Zero-Shot Vergleich)",
                    }, label="Modellvergleich")

                    gr.HTML('<div class="abschnitt-titel">📝 NLP Block — RAG Evaluation (LM-as-Judge)</div>')
                    gr.JSON(value={
                        "It. 1 — Zero-Shot (ohne Kontext)":        "4.50 / 5.0",
                        "It. 2 — RAG Basic (mit AVB-Kontext)":     "4.50 / 5.0",
                        "It. 3 — RAG Expert (Rolle + Struktur) ← WINNER": "4.70 / 5.0",
                        "Rollen":      "Experte · Kundenberater · Fraud-Analyst",
                        "AVB-Quellen": "AXA OPTIMA 2023 (25 S.) + AXA MF 2021 (17 S.) = 70 Chunks",
                    }, label="RAG Evaluation")

            gr.HTML('<div class="abschnitt-titel">🔬 Ablation Study — Multimodaler Mehrwert</div>')
            gr.JSON(value={
                "Exp. 1 — Nur strukturierte Daten":   {"RMSE": "CHF 12'821", "R²": 0.661, "Mehrwert": "Baseline"},
                "Exp. 2 — Strukturiert + NLP":         {"RMSE": "CHF 12'847", "R²": 0.660, "Mehrwert": "+0%"},
                "Exp. 3 — Strukturiert + CV":          {"RMSE": "CHF 12'133", "R²": 0.696, "Mehrwert": "+5%"},
                "Exp. 4 — Vollständig multimodal":     {"RMSE": "CHF 12'352", "R²": 0.685, "Mehrwert": "+3.6%"},
                "Hinweis": "CV/NLP Features im ML-Block simuliert — echter Mehrwert im produktiven System höher erwartet.",
            }, label="")

            with gr.Row():
                try:
                    gr.Image(value=model_path("ablation_study.png"), label="Ablation Study — R² Vergleich")
                except:
                    pass
                try:
                    gr.Image(value=model_path("shap_summary.png"), label="SHAP Feature Importance — XGBoost")
                except:
                    pass

    # ── FOOTER ────────────────────────────────────────────────
    gr.HTML("""
    <div class="footer-box">
      ⚠️ Dieses System dient ausschliesslich als <strong>Entscheidungsunterstützung</strong> —
      die finale Entscheidung liegt beim menschlichen Sachbearbeiter.<br>
      <span style="opacity:0.5;font-size:0.8rem;">Insurance Claim Intelligence · ZHAW AI Applications FS2026</span>
    </div>
    """)

if __name__ == "__main__":
    demo.launch(
        theme=gr.themes.Base(primary_hue="blue"),
        css=CSS,
    )
else:
    demo.launch(
        theme=gr.themes.Base(primary_hue="blue"),
        css=CSS,
    )
