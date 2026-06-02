"""
Insurance Claim Intelligence — NLP Block
RAG Pipeline über AXA AVB PDFs (OPTIMA 2023 + Motorfahrzeug 2021)
SentenceTransformer Embeddings + Cosine Similarity Retrieval
3 Prompt-Varianten + 3 Rollen im Vergleich
LM-as-Judge Evaluation (10 Testfragen)
"""

import os, json
import numpy as np
import requests
from io import BytesIO
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from openai import OpenAI

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
client_oai     = OpenAI(api_key=OPENAI_API_KEY)
os.makedirs("models", exist_ok=True)

# ──────────────────────────────────────────────────────────────
# 1. AXA AVB PDFs laden + Chunking + Embeddings
# ──────────────────────────────────────────────────────────────
POLICY_URLS = {
    "AXA_OPTIMA_2023": "https://www.axa.ch/doc/ajhtk",
    "AXA_MF_2021":     "https://mzo.ch/wp-content/uploads/AXA-MF_AVB_10.2021_DE.pdf",
}

def download_pdf(url, name):
    try:
        resp   = requests.get(url, timeout=15, headers={"User-Agent": "Mozilla/5.0"})
        reader = PdfReader(BytesIO(resp.content))
        text   = "\n".join(page.extract_text() or "" for page in reader.pages)
        print(f"  {name}: {len(reader.pages)} Seiten")
        return text
    except Exception as e:
        print(f"  {name}: Fehler - {e}")
        return ""

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
all_texts = {n: download_pdf(u, n) for n, u in POLICY_URLS.items()}

all_chunks, all_sources = [], []
for doc_name, text in all_texts.items():
    if not text:
        continue
    for chunk in chunk_text(text):
        all_chunks.append(chunk)
        all_sources.append(doc_name)

print(f"{len(all_chunks)} Chunks aus PDFs")

# Fallback: Synthetische Policies falls PDFs nicht laden
if len(all_chunks) < 10:
    print("Fallback: Synthetische AXA AVB Policies...")
    POLICIES = [
        ("Vollkaskoversicherung deckt Kollision, Diebstahl, Feuer, Naturereignisse, Vandalismus. Franchise CHF 300-2000.", "AXA AVB §59"),
        ("Teilkaskoversicherung deckt Diebstahl, Feuer, Naturereignisse, Glasbruch, Marderbiss. Keine Kollision.", "AXA AVB §58"),
        ("Parkschäden durch unbekannte Dritte bei Vollkasko gedeckt, sofern unverzüglich gemeldet.", "AXA AVB §12"),
        ("Versicherungsbetrug: Falsche Angaben führen zur Leistungsverweigerung.", "AXA AVB §40"),
        ("Hagelschäden: Elementarschaden in Teilkasko gedeckt. Meldung innerhalb 5 Tage.", "AXA AVB §8"),
        ("Totalschäden auf Basis des Zeitwerts entschädigt, abzüglich Franchise.", "AXA AVB §15"),
        ("Vandalismusschäden bei Vollkasko gedeckt. Polizeirapport erforderlich.", "AXA AVB §11"),
        ("Glasbruch ohne Franchise bei Glasbruchdeckung.", "AXA AVB §9"),
        ("Auffahrunfälle bei Vollkasko gedeckt. Bei Teilschuld anteilige Kürzung.", "AXA AVB §10"),
        ("Diebstahl bei Voll- und Teilkasko. Meldung innerhalb 48 Stunden.", "AXA AVB §13"),
        ("Kratzer und Lackschäden durch Vandalismus bei Vollkasko.", "AXA AVB §11b"),
        ("Marderschäden an Kabeln und Schläuchen in Teilkasko.", "AXA AVB §8b"),
    ]
    all_chunks  = [p[0] for p in POLICIES]
    all_sources = [p[1] for p in POLICIES]

print("Berechne Embeddings (all-MiniLM-L6-v2)...")
embedder         = SentenceTransformer("all-MiniLM-L6-v2", backend="torch")
chunk_embeddings = embedder.encode(all_chunks, show_progress_bar=True)
print(f"Embeddings: {chunk_embeddings.shape}")


# ──────────────────────────────────────────────────────────────
# 2. Retrieval
# ──────────────────────────────────────────────────────────────
def retrieve(query, n=4):
    q_emb  = embedder.encode([query])
    scores = np.dot(chunk_embeddings, q_emb.T).squeeze()
    top    = np.argsort(scores)[::-1][:n]
    return [all_chunks[i] for i in top], [all_sources[i] for i in top]


# ──────────────────────────────────────────────────────────────
# 3. Prompt-Varianten + Rollen
# ──────────────────────────────────────────────────────────────
SYSTEM_PROMPTS = {
    "expert": (
        "Du bist ein erfahrener Schweizer Versicherungsexperte bei AXA mit 20 Jahren Erfahrung. "
        "Zitiere immer die relevante AVB-Klausel. Weise klar auf Ausschluesse hin."
    ),
    "customer_service": (
        "Du bist ein freundlicher AXA Kundenberater. "
        "Erklaere einfach ohne Fachjargon. Schliesse mit: Haben Sie weitere Fragen?"
    ),
    "fraud_analyst": (
        "Du bist ein Fraud Analyst bei AXA. Analysiere Schadenmeldungen auf Inkonsistenzen. "
        "Gib strukturierten Fraud-Assessment-Report mit Plausibilitaetsbewertung 1-10 aus."
    ),
}

def query_rag(incident, insurance_type, description="", role="expert", prompt_type="rag_expert"):
    docs, sources = retrieve(f"{incident} {insurance_type} gedeckt")
    context       = "\n\n".join(docs)

    if prompt_type == "zero_shot":
        # Iteration 1: kein Kontext
        messages = [{"role": "user", "content":
            f"Ist '{incident}' bei '{insurance_type}' gedeckt? Antworte auf Deutsch."}]

    elif prompt_type == "rag_basic":
        # Iteration 2: AVB-Kontext, keine Persona
        messages = [{"role": "user", "content":
            f"Versicherungsbedingungen:\n{context}\n\n"
            f"Ist '{incident}' bei '{insurance_type}' gedeckt?"}]

    else:
        # Iteration 3: AVB-Kontext + Rolle + Struktur (WINNER)
        if role == "fraud_analyst":
            user_msg = (
                f"AVB:\n{context}\n\nFall: {insurance_type} | {incident}\n"
                f"Beschreibung: {description or 'Keine Angabe'}\n\n"
                f"1. Plausibilitaetsbewertung (1-10)\n"
                f"2. Auffaelligkeiten\n"
                f"3. Empfehlung: Freigeben / Manuelle Pruefung / Ablehnen"
            )
        elif role == "customer_service":
            user_msg = (
                f"AVB:\n{context}\n\nSchaden: {incident} | {insurance_type}\n"
                f"Erklaere freundlich was gedeckt ist und naechste Schritte."
            )
        else:
            user_msg = (
                f"AVB:\n{context}\n\nFall: {insurance_type} | {incident}\n"
                f"Beschreibung: {description or '-'}\n\n"
                f"1. Gedeckt? (Ja/Nein/Teilweise)\n"
                f"2. AVB-Klausel\n"
                f"3. Ausschluesse\n"
                f"4. Naechste Schritte"
            )
        messages = [
            {"role": "system", "content": SYSTEM_PROMPTS[role]},
            {"role": "user",   "content": user_msg},
        ]

    resp = client_oai.chat.completions.create(
        model="gpt-4o-mini", messages=messages,
        temperature=0.1, max_tokens=400,
    )
    return resp.choices[0].message.content, sources


# ──────────────────────────────────────────────────────────────
# 4. NLP Feature Extraktion (fuer ML-Block)
# ──────────────────────────────────────────────────────────────
FRAUD_SIGNALS  = ["total loss", "gestohlen", "unbekannt", "keine zeugen",
                  "sofort", "bar bezahlt", "keine quittung", "dringend"]
INCIDENT_TYPES = {
    "parking":   ["parkschaden", "parkunfall", "parking", "eingeparkt"],
    "rear_end":  ["auffahrunfall", "hinten", "von hinten"],
    "vandalism": ["vandalismus", "zerkratzt", "absichtlich"],
    "theft":     ["diebstahl", "gestohlen", "einbruch"],
    "weather":   ["hagel", "sturm", "ueberschwemmung"],
}

def extract_nlp_features(description: str) -> dict:
    desc_lower    = description.lower()
    incident_type = "other"
    for itype, keywords in INCIDENT_TYPES.items():
        if any(kw in desc_lower for kw in keywords):
            incident_type = itype
            break
    return {
        "incident_type_nlp":  incident_type,
        "fraud_signal_count": sum(1 for s in FRAUD_SIGNALS if s in desc_lower),
        "description_length": len(description.split()),
    }

def compute_consistency_score(damage_type: str, severity: int, description: str) -> float:
    """CV-NLP Konsistenzpruefung — stärkstes Fraud-Signal."""
    severity_words = {
        0: ["klein", "leicht", "kratzer", "minor", "slight"],
        1: ["mittel", "moderate", "medium"],
        2: ["schwer", "total", "stark", "severe", "crash"],
    }
    desc_lower = description.lower()
    expected   = severity_words.get(severity, [])
    opposite   = [w for s, ws in severity_words.items() if s != severity for w in ws]
    match      = sum(1 for w in expected if w in desc_lower) / max(len(expected), 1)
    mismatch   = sum(1 for w in opposite if w in desc_lower) / max(len(opposite), 1)
    score      = max(0.0, min(1.0, match - mismatch * 0.5 + 0.5))
    if damage_type == "glass_shatter" and any(w in desc_lower for w in ["klein", "kratzer", "minor"]):
        score = 0.08
    return round(score, 3)


# ──────────────────────────────────────────────────────────────
# 5. LM-as-Judge Evaluation
# ──────────────────────────────────────────────────────────────
def evaluate_rag():
    TEST_CASES = [
        ("Parkschaden unbekannter Dritter", "Vollkasko"),
        ("Glasbruch Windschutzscheibe",     "Teilkasko"),
        ("Hagelschaden Motorhaube",         "Teilkasko"),
        ("Vandalismusschaden Lack",         "Vollkasko"),
        ("Auffahrunfall Frontschaden",      "Vollkasko"),
        ("Diebstahl Fahrzeug",              "Teilkasko"),
        ("Kollisionsschaden Parkhaus",      "Haftpflicht"),
        ("Marderschaden Kabel",             "Teilkasko"),
        ("Brandschaden Motorraum",          "Vollkasko"),
        ("Kratzer Parkrempler",             "Haftpflicht"),
    ]

    scores = {"zero_shot": 0, "rag_basic": 0, "rag_expert": 0}
    print(f"LM-as-Judge Evaluation ({len(TEST_CASES)} Testfaelle):")

    for incident, ins_type in TEST_CASES:
        for pt in scores:
            answer, _ = query_rag(incident, ins_type, prompt_type=pt)
            judge = client_oai.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content":
                    f"Bewerte (1-5): Ist '{incident}' bei '{ins_type}' gedeckt?\n"
                    f"Antwort: {answer[:200]}\nNur Zahl 1-5."}],
                temperature=0, max_tokens=3,
            )
            try:
                scores[pt] += int(judge.choices[0].message.content.strip())
            except:
                scores[pt] += 3

    print("\nErgebnisse (Durchschnitt 1-5):")
    for pt, total in scores.items():
        avg = total / len(TEST_CASES)
        print(f"  {pt:15}: {avg:.2f}/5.0")
        scores[pt] = avg

    with open("models/rag_results.json", "w") as f:
        json.dump({
            "chunks_indexed": len(all_chunks),
            "prompt_scores":  scores,
            "roles":          list(SYSTEM_PROMPTS.keys()),
        }, f, indent=2)
    return scores


if __name__ == "__main__":
    print("=== NLP Block: Prompt-Vergleich ===")
    for pt in ["zero_shot", "rag_basic", "rag_expert"]:
        answer, sources = query_rag(
            "Kollisionsschaden Auffahrunfall", "Vollkasko",
            prompt_type=pt)
        print(f"\nPrompt: {pt}")
        print(f"Quellen: {set(sources)}")
        print(f"Antwort: {answer[:200]}...")

    print("\n=== Rollen-Vergleich ===")
    for role in ["expert", "customer_service", "fraud_analyst"]:
        answer, _ = query_rag(
            "Glasbruch Windschutzscheibe", "Vollkasko",
            description="Kleiner Kratzer laut Kunde, kaum sichtbar",
            role=role, prompt_type="rag_expert")
        print(f"\nRolle: {role}")
        print(f"{answer[:300]}...")

    print("\n=== LM-as-Judge Evaluation ===")
    evaluate_rag()
    print("NLP Block abgeschlossen.")
