# Insurance Claim Intelligence — ZHAW AI Applications FS2026

---

## Project Metadata

- **Project title:** Insurance Claim Intelligence — Multimodale KI-Bewertung von Versicherungsfällen
- **Student:** Darren Glatzl
- **GitHub repository URL:** https://github.com/sgDarren/InsuranceClaimIntelligence
- **Deployment URL:** https://huggingface.co/spaces/DarrenOG/InsuranceClaim
- **Submission date:** 07.06.2026

### Mandatory Setup Checks

- [x] At least 2 blocks selected
- [x] Multiple and different data sources used
- [x] Deployment URL provided
- [x] Required GitHub users added to repository (`jasminh`, `bkuehnis`)

## Selected AI Blocks

- [x] ML Numeric Data
- [x] NLP
- [x] Computer Vision

Primary blocks used for core solution (choose 2):

- Primary block 1: ML Numeric Data
- Primary block 2: Computer Vision

If a third block is selected, it is documented and graded separately as extra work.

---

## 1. Project Foundation (Short)

### 1.1 Problem Definition

- **Problem statement:** Die manuelle Bewertung von Versicherungsschäden dauert 3-5 Tage, ist fehleranfällig und rund 10% der Claims sind betrügerisch. Bestehende Systeme nutzen nur strukturierte Daten und ignorieren Schadenbilder und -beschreibungen.

  *Quellen: Insurance Europe (2023), Swiss Insurance Association SVV (2022). Das verwendete Kaggle-Dataset weist eine Fraud-Rate von 5% auf — in der Praxis liegt diese gemäss Branchenquellen bei 10-15%.*

- **Goal:** Ein multimodales System das Schadenbilder (CV), Schadenbeschreibungen (NLP) und strukturierte Vertragsdaten (ML) kombiniert, um Schadenhöhe, Fraud-Score und Deckungswahrscheinlichkeit automatisch vorherzusagen.

- **Success criteria:**
  - ML: R² ≥ 0.70 → erreicht: **R² = 0.721** ✅
  - CV: Accuracy ≥ 0.75 → erreicht: **79.85%, F1 Macro = 0.85** ✅
  - RAG: LM-Judge ≥ 4.0/5.0 → erreicht: **4.70/5.0** ✅
  - Fraud Detection AUC ≥ 0.75 → erreicht: **AUC = 0.931** ✅
  - consistency_score (CV↔NLP) als messbares multimodales Fraud-Feature ✅

### 1.2 Integration Logic

- **How the selected blocks interact:**

```
📸 Foto → [CV Block: ViT] → damage_type, cv_confidence
📝 Text → [NLP Block: RAG] → incident_type, fraud_signals
                ↓
    consistency_score = f(CV_label, NLP_description)
    → Stärkstes Fraud-Signal (nur multimodal berechenbar!)
                ↓
📋 Daten → [ML Block: XGBoost] → Schadenhöhe CHF, Fraud-Score
                ↓
         [RAG: AXA AVB PDFs] → Deckungsauskunft
```

- **Data and output flow between blocks:**
  - CV extrahiert `damage_type` + `cv_confidence` → ML-Features
  - NLP extrahiert `incident_type` + `fraud_signal_count` → ML-Features
  - CV + NLP zusammen berechnen `consistency_score` → stärkstes Fraud-Feature im ML-Block
  - ML-Output (`damage_type`, `insurance_type`) → RAG-Query für Deckungsauskunft

See [`src/app/app.py`](https://github.com/sgDarren/InsuranceClaimIntelligence/blob/main/src/app/app.py) für den vollständigen End-to-End Pipeline-Code. See [`notebooks/insurance_claim_intelligence.ipynb`](https://github.com/sgDarren/InsuranceClaimIntelligence/blob/main/notebooks/insurance_claim_intelligence.ipynb) für das reproduzierbare Training-Notebook.

---

## 2. Block Documentation

### 2A. ML Numeric Data

#### 2A.1 Data Source(s)

| Entry | Source name or link | Type | Size | Role in this block |
|---|---|---|---|---|
| 1 | [Insurance Claims Fraud Data (Kaggle: mastmustu)](https://www.kaggle.com/datasets/mastmustu/insurance-claims-fraud-data) | CSV tabular | 10'000 Claims, 38 Features | Training Schadenhöhe (Regression) + Fraud-Label (Classification) |
| 2 | CV Block Output (ViT) | Structured features | 4 Features pro Bild | damage_type_enc, damage_severity, damage_area_pct, cv_confidence |
| 3 | NLP Block Output | Structured features | 4 Features pro Text | incident_type_nlp, fraud_signal_count, description_length, consistency_score |

#### 2A.2 Preprocessing and Features

**EDA — Key Findings** (See [`models/eda_ml.png`](https://github.com/sgDarren/InsuranceClaimIntelligence/blob/main/models/eda_ml.png)):

- **Schadenhöhe:** Stark rechtsschief — Median ~CHF 5'000, 75% aller Claims unter CHF 20'000, Ausreisser bis CHF 100'000. → Log-Transformation erwogen, aber XGBoost robust gegenüber Schiefe.
- **Fraud Rate:** 5.0% (503/10'000) — starke Klassenimbalance → `class_weight="balanced"` im Fraud-Classifier zwingend notwendig.
- **Versicherungstypen:** Gleichmässig verteilt (Property/Mobile/Health/Life/Travel/Motor je ~1'600 Claims) → kein Sampling-Bias.
- **Incident Severity vs. Schadenhöhe:** Major Loss und Total Loss haben ähnliche Mediane aber viele Ausreisser → Schweregrad allein reicht nicht zur Kostenschätzung, CV-Features nötig.
- **Policy-Alter:** Gleichmässig verteilt (0-4000 Tage), Spike bei neuen Policen < 90 Tage → `new_policy_flag` als Feature gerechtfertigt.
- **Ø Schadenhöhe Legitim vs. Fraud:** Nahezu identisch (~CHF 16'500) → Schadenhöhe allein kein Fraud-Indikator, multimodale Features (consistency_score) notwendig.

- **Cleaning steps:** Datumsspalten in datetime konvertiert, kategorische Features Label-Encoded, fehlende Werte mit 0 aufgefüllt.

- **Preprocessing steps:** Feature Engineering: `policy_age_days` (LOSS_DT - POLICY_EFF_DT), `report_delay_days`, `new_policy_flag` (< 90 Tage), `is_night` (Stunde < 6 oder > 22).

- **Feature engineering and selection:** 19 strukturierte Features + 4 NLP-Features + 4 CV-Features = 27 Features total.

  | Feature | Typ | Quelle | Begründung |
  |---|---|---|---|
  | `policy_age_days` | Engineered | Structured | LOSS_DT - POLICY_EFF_DT → neue Policen (<90 Tage) = höheres Fraud-Risiko |
  | `report_delay_days` | Engineered | Structured | REPORT_DT - LOSS_DT → verzögerte Meldung = Fraud-Signal |
  | `new_policy_flag` | Binary | Structured | policy_age_days < 90 → binäres Fraud-Feature |
  | `is_night` | Binary | Structured | Stunde < 6 oder > 22 → Nachtunfälle häufiger fraudulös |
  | `INSURANCE_TYPE_enc` | Encoded | Structured | LabelEncoder auf kategorische Spalte |
  | `INCIDENT_SEVERITY_enc` | Encoded | Structured | LabelEncoder: Minor/Major/Total Loss |
  | `damage_type_enc` | Encoded | **CV-Output** | ViT-Klassifikation → dent/scratch/crack/glass_shatter/no_damage |
  | `damage_severity` | Numeric | **CV-Output** | 0=Minor, 1=Moderate, 2=Severe |
  | `cv_confidence` | Numeric | **CV-Output** | ViT-Modellkonfidenz (0-1) |
  | `consistency_score` | Numeric | **CV+NLP** | Cosine-Ähnlichkeit CV-Label ↔ NLP-Beschreibung → stärkstes Fraud-Feature |
  | `fraud_signal_count` | Numeric | **NLP-Output** | Anzahl Fraud-Signalwörter in Beschreibung |
  | `incident_type_nlp` | Encoded | **NLP-Output** | Unfalltyp aus Keyword-Matching (parking/rear_end/vandalism/theft/weather) |

  Feature Selection: alle 27 Features verwendet — SHAP zeigt `INSURANCE_TYPE_enc`, `PREMIUM_AMOUNT` und `consistency_score` als Top-3.

See [`src/ml/train.py`](https://github.com/sgDarren/InsuranceClaimIntelligence/blob/main/src/ml/train.py), Zeilen 1-80.

#### 2A.3 Model Selection

- **Models tested:** Linear Regression, Random Forest (200 Trees), XGBoost (GridSearch)

- **Why these models were chosen:**
  - Linear Regression: interpretierbare Baseline (direkt aus Kursunterlagen)
  - Random Forest: robustes Ensemble, keine starken Verteilungsannahmen
  - XGBoost: State-of-the-Art für tabellarische Daten, GridSearch-optimiert

#### 2A.4 Model Comparison and Iterations

| Iteration | Objective | Key changes | Models used | Main metric | Change vs previous |
|---|---|---|---|---|---|
| 1 | Baseline | Alle Features, kein Tuning | Linear Regression | RMSE=CHF 21'173, R²=0.075 | — |
| 2 | Ensemble | 200 Trees, Feature Engineering | Random Forest | RMSE=CHF 11'657, R²=0.720 | R² +645% |
| 3 | Optimiert | GridSearch (n_estimators, max_depth, learning_rate), class_weight="balanced" | XGBoost | RMSE=CHF 11'624, R²=0.721, Fraud AUC=**0.931** | Fraud AUC: 0.720→0.931 (+29%) |

**Warum XGBoost als Winner:** Der marginale R²-Gewinn (+0.001) ist nicht ausschlaggebend. Entscheidend ist die Fraud Detection: XGBoost mit GridSearch und `class_weight="balanced"` erreicht AUC=0.931 und Precision_Fraud=1.00 — kein legitimer Kunde wird fälschlicherweise als Fraud markiert. Random Forest erreichte denselben R², aber XGBoost ist durch GridSearch vollständig konfigurierbar und liefert native SHAP-Kompatibilität für Explainability.

See [`src/ml/train.py`](https://github.com/sgDarren/InsuranceClaimIntelligence/blob/main/src/ml/train.py), Iterationen Zeilen 90-160.

#### 2A.5 Evaluation and Error Analysis

- **Metrics used:** RMSE (CHF), R², F1, AUC-ROC (Fraud), Ablation Study (4 Experimente)

- **Final results:**

  Ablation Study:
  | Experiment | Features | RMSE CHF | R² |
  |---|---|---|---|
  | Structured Only | 19 Features | 12'821 | 0.661 |
  | + NLP | +4 Features | 12'847 | 0.660 |
  | + CV | +4 Features | 12'133 | 0.696 |
  | Full Multimodal | +consistency | 12'352 | 0.685 |

  Fraud Detection: F1=0.311, **AUC=0.931**, Precision_Fraud=**1.00**

- **Error patterns and likely causes:** CV/NLP Features sind im ML-Block simuliert (synthetisch), da echte Modell-Outputs erst im produktiven System fliessen. **Wichtige Einschränkung:** Die Ablation Study beweist daher primär die Architektur-Entscheidung, nicht den tatsächlichen quantitativen Mehrwert. Im produktiven System — wo ViT echte `damage_type` Features liefert — wird der Mehrwert grösser erwartet. Die EDA belegt dies: Ø Schadenhöhe von Legitim vs. Fraud ist nahezu identisch (CHF ~16'500), was zeigt dass strukturierte Daten allein Fraud nicht erkennen können → multimodale Features (consistency_score) sind zwingend notwendig. Fraud Recall niedrig (0.18) wegen 5% Klassenimbalance; Precision=1.00 bewusst priorisiert (kein legitimer Kunde falsch markiert).

  **Warum kein End-to-End Training möglich war:** Das Kaggle-Dataset (`insurance_data.csv`) enthält ausschliesslich strukturierte Spalten — keine Schadenfotos, keine Freitextbeschreibungen. Das CV-Dataset (HuggingFace `SaiVaibhavS`) enthält Fotos, aber keine strukturierten Vertragsdaten. Beide Datasets teilen keinen gemeinsamen Claim-Identifier und können daher nicht verbunden werden. Ein echtes multimodales Dataset (Foto + Beschreibungstext + Vertragsdaten für denselben Claim) existiert nicht öffentlich, da Versicherungsfotos zusammen mit Vertragsdaten datenschutzrechtlich als personenbezogene Daten gelten (DSGVO) und von keinem Versicherer öffentlich geteilt werden. Die Simulation der CV/NLP Features ist daher die einzig mögliche Lösung im akademischen Kontext — in einem produktiven System würden diese Features direkt von den trainierten Modellen befüllt.

See [`models/ablation_study.png`](https://github.com/sgDarren/InsuranceClaimIntelligence/blob/main/models/ablation_study.png), [`models/shap_summary.png`](https://github.com/sgDarren/InsuranceClaimIntelligence/blob/main/models/shap_summary.png).

#### 2A.6 Integration with Other Block(s)

- **Inputs received from other block(s):**
  - Von CV: `damage_type_encoded`, `damage_severity`, `damage_area_pct`, `cv_confidence`
  - Von NLP: `incident_type_nlp`, `fraud_signal_count`, `description_length`, `consistency_score`

- **Outputs provided to other block(s):**
  - An RAG: `damage_type` + `insurance_type` → RAG-Query "Ist [Schadenstyp] bei [Versicherungstyp] gedeckt?"
  - An User: `claim_amount_chf`, `fraud_score`, `priority`, `recommendation`

---

### 2B. NLP

#### 2B.1 Data Source(s)

| Entry | Source name or link | Type | Size | Role in this block |
|---|---|---|---|---|
| 1 | [AXA OPTIMA AVB 2023](https://www.axa.ch/doc/ajhtk) | PDF (öffentlich) | 25 Seiten | RAG Wissensbasis — Deckungsauskunft |
| 2 | [AXA Motorfahrzeug AVB 2021](https://mzo.ch/wp-content/uploads/AXA-MF_AVB_10.2021_DE.pdf) | PDF (öffentlich) | 17 Seiten | RAG Wissensbasis — Deckungsauskunft |
| 3 | User-Eingabe (Schadenbeschreibung) | Freitext | Variable | Feature Extraktion: incident_type, fraud_signals, consistency_score |

#### 2B.2 Preprocessing and Prompt Design

- **Text preprocessing:**
  - PDFs → PyPDF Text-Extraktion → 70 Chunks (300 Wörter, 30 Wörter Overlap)
  - Embeddings: `all-MiniLM-L6-v2` (SentenceTransformer, open-source)
  - Retrieval: Cosine Similarity via NumPy (Top-4 Chunks)
  - Schadenbeschreibung → Keyword-Matching für `incident_type` und `fraud_signal_count`

- **Prompt design or retrieval setup:**
  - 3 Prompt-Varianten verglichen: Zero-Shot / RAG Basic / RAG Expert
  - 3 Rollen: `expert` (AVB-Klausel), `customer_service` (einfach), `fraud_analyst` (Plausibilität 1-10)
  - Grounded Prompt: AVB-Chunks als Kontext + strukturierte Antwort-Anforderung

See [`src/nlp/rag_pipeline.py`](https://github.com/sgDarren/InsuranceClaimIntelligence/blob/main/src/nlp/rag_pipeline.py).

#### 2B.3 Approach Selection

- **Approach used:** RAG (Retrieval-Augmented Generation) mit SentenceTransformer Embeddings + GPT-4o-mini

- **Alternatives considered:** Fine-Tuning auf Versicherungsdaten abgelehnt — AVB ändern sich regelmässig, RAG erlaubt einfaches Update der Wissensbasis ohne Retraining. Plain LLM abgelehnt wegen Halluzinationsrisiko bei spezifischen AVB-Klauseln.

#### 2B.4 Comparison and Iterations

| Iteration | Objective | Key changes | Model or prompt setup | Main metric | Change vs previous |
|---|---|---|---|---|---|
| 1 | Baseline | Kein Kontext | Zero-Shot GPT-4o-mini | LM-Judge: 4.50/5.0 | — |
| 2 | +Kontext | AVB-Chunks hinzugefügt | RAG Basic | LM-Judge: 4.50/5.0 | +0.00 |
| 3 | +Persona | Rollenspez. System Prompt + Struktur | RAG Expert | LM-Judge: **4.70/5.0** | **+0.20** |

#### 2B.5 Evaluation and Error Analysis

- **Evaluation strategy:** LM-as-Judge (GPT-4o-mini, Score 1-5) auf 10 Testfragen (Parkschaden, Glasbruch, Hagel, Vandalismus, Diebstahl, Marderschaden etc.)

- **Results:** RAG Expert: 4.70/5.0 — Fraud Analyst Rolle bewertet Plausibilität korrekt (4/10 für "kleiner Kratzer aber Glasbruch erkannt" = Inkonsistenz erkannt ✅)

- **Error patterns and likely causes:**
  - **Dünn belegte Schadensarten:** Sehr spezifische Fälle wie "Marderschaden an Bremsschläuchen" oder "Aquaplaning-Unfall" sind in den 70 AVB-Chunks nur marginal abgedeckt. Das Modell antwortet dann mit allgemeinen Formulierungen statt präzisen Klausel-Zitaten — erwünschtes Verhalten ("refuse when evidence is thin") statt Halluzination.
  - **Haftpflicht-Fragen:** Das RAG antwortet schwächer bei Haftpflicht-spezifischen Fragen, da die AXA OPTIMA und MF-AVB primär Kasko-Deckungen behandeln. Ein separates Haftpflicht-AVB-Dokument würde die Abdeckung verbessern.
  - **Zero-Shot vs. RAG:** Zero-Shot (4.50) erreicht ähnliche Scores wie RAG Basic (4.50) weil GPT-4o-mini starkes Allgemeinwissen über Schweizer Versicherungsrecht hat. Der Mehrwert von RAG Expert (4.70) liegt primär in der Präzision der AVB-Klausel-Zitate und der konsistenten Rollenstruktur — nicht in der Faktenkorrektheit.

See [`models/rag_results.json`](https://github.com/sgDarren/InsuranceClaimIntelligence/blob/main/models/rag_results.json).

#### 2B.6 Integration with Other Block(s)

- **Inputs received from other block(s):**
  - Von CV: `damage_type` → RAG-Query "Deckung für [Schadenstyp]"
  - Von ML: `insurance_type` → filtert relevante Policy-Chunks
  - Von CV+ML: `consistency_score` < 0.3 → Fraud Analyst Rolle aktiviert

- **Outputs provided to other block(s):**
  - An User: Deckungsauskunft + AVB-Klausel-Referenz + Nächste Schritte
  - `consistency_score` → ML-Block als stärkstes Fraud-Feature

---

### 2C. Computer Vision

#### 2C.1 Data Source(s)

| Entry | Source name or link | Type | Size | Role in this block |
|---|---|---|---|---|
| 1 | [SaiVaibhavS/comprehensive-car-damage (HuggingFace)](https://huggingface.co/datasets/SaiVaibhavS/comprehensive-car-damage) | Bilder | 2'300 Bilder, 6 Originalklassen | ViT Training nach GPT-4o Relabeling |
| 2 | GPT-4o Vision Relabeling Output | CSV (generiert) | 2'300 Labels | Neue versicherungskonforme Klassen |

**Dataset-Hinweis:** Originale 6 Klassen (F/R_Breakage/Crushed/Normal) wurden via LLM-gestütztem Relabeling in 5 Versicherungsklassen überführt: Breakage→`glass_shatter` (Regel), Normal→`no_damage` (Regel), Crushed→GPT-4o entscheidet `dent`/`scratch`/`crack`. Nach WeightedSampler-Balancing: 2'671 Bilder.

#### 2C.2 Preprocessing and Augmentation

- **Image preprocessing:** Resize(256×256) → CenterCrop(224) für Validation; Resize(256×256) → RandomCrop(224) für Training.

- **Augmentation strategy:**
  - RandomHorizontalFlip(p=0.5): Schäden links/rechts symmetrisch
  - RandomPerspective(0.3, p=0.4): Handy-Fotos aus verschiedenen Winkeln
  - ColorJitter(brightness=0.4): Tageslicht bis Tiefgarage
  - GaussianBlur(σ=0.1-2.0): Bewegungsunschärfe
  - RandomErasing(p=0.1): verhindert Overfitting auf Hintergrund
  - WeightedRandomSampler: Klassenimbalance ausgeglichen
  - KEIN VerticalFlip: Fahrzeuge haben klare Oben/Unten-Orientierung

See [`src/cv/train.py`](https://github.com/sgDarren/InsuranceClaimIntelligence/blob/main/src/cv/train.py).

#### 2C.3 Model Selection

- **Vision model(s) used:** CLIP (Zero-Shot), ViT-B/16 Transfer Learning, ViT-B/16 Fine-Tuning, GPT-4o Vision

- **Why these model(s) were chosen:** ViT-B/16 direkt aus Kursunterlagen (Slides_KI_Anwendung_img_2.pdf). CLIP für Zero-Shot Baseline (Woche 5+7). GPT-4o als State-of-the-Art Vergleich (aus Kurs-App DarrenOG).

#### 2C.4 Model Comparison and Iterations

| Iteration | Objective | Key changes | Model(s) used | Main metric | Change vs previous |
|---|---|---|---|---|---|
| 1 | Zero-Shot Baseline | Kein Training, candidate labels | CLIP ViT-L/14 | Accuracy: 0.00% | — |
| 2 | Transfer Learning | Base frozen, nur Classifier, LR=3e-4, 5 Epochs | ViT-B/16 | Accuracy: 60.82% | +60.82% |
| 3 | Fine-Tuning | Letzte 6 Layer + LayerNorm, LR=1e-5, 15 Epochs, fp16 | ViT-B/16 | Accuracy: **79.85%**, F1=0.85 | +19.03% |
| 4 | SotA Vergleich | Zero-Shot, detail=high | GPT-4o Vision | Accuracy: 58.00% | — (Vergleich) |

#### 2C.5 Evaluation and Error Analysis

- **Metrics and/or visual checks:** Accuracy, F1 Macro, Classification Report per Klasse, Konfusionsmatrix

- **EDA CV-Dataset** (See [`models/eda_cv.png`](https://github.com/sgDarren/InsuranceClaimIntelligence/blob/main/models/eda_cv.png)):
  - Nach GPT-4o Relabeling + WeightedSampler-Balancing: dent=671, scratch=200, crack=200, glass_shatter=800, no_damage=800
  - scratch und crack waren im Original-Dataset nicht vorhanden → Oversampling notwendig

- **Final results (Fine-Tuning, Test n=268):**

  | Klasse | Precision | Recall | F1 | Support |
  |---|---|---|---|---|
  | dent | 0.62 | 0.71 | 0.66 | 68 |
  | scratch | 0.95 | 1.00 | 0.98 | 20 |
  | crack | 0.95 | 1.00 | 0.98 | 20 |
  | glass_shatter | 0.92 | 0.75 | 0.83 | 80 |
  | no_damage | 0.79 | 0.82 | 0.80 | 80 |
  | **Accuracy** | | | **0.80** | **268** |
  | **F1 Macro** | | | **0.85** | |

- **Konfusionsmatrix-Analyse** (See [`models/eda_cv.png`](https://github.com/sgDarren/InsuranceClaimIntelligence/blob/main/models/eda_cv.png)):
  - `dent` → 15x als `no_damage` verwechselt, 5x als `glass_shatter`: Dellen bei Totalschäden sehen optisch wie kein Schaden aus oder ähneln Glasbruch-Konturen
  - `glass_shatter` → 16x als `dent` klassifiziert: Splitter und eingedrückte Karosserie sind bei Frontalschäden visuell ähnlich
  - `no_damage` → 13x als `dent` verwechselt: Reflexionen und Schatten auf unbeschädigten Karosserien werden als Dellen interpretiert
  - `scratch` und `crack`: perfekt klassifiziert (F1=0.98) weil Muster eindeutig

- **Error patterns and limitations:** Die Hauptverwechslung `dent`↔`glass_shatter`↔`no_damage` ist erklärbar: alle drei zeigen bei Frontalschäden ähnliche Konturen. Mit mehr Trainingsdaten oder höherer Auflösung wäre Verbesserung möglich. GPT-4o (58%) schlechter als Fine-Tuning (80%) bestätigt: domänenspezifisches Training schlägt generalistisches Zero-Shot.

See [`models/eda_cv.png`](https://github.com/sgDarren/InsuranceClaimIntelligence/blob/main/models/eda_cv.png), [`models/cv_results.json`](https://github.com/sgDarren/InsuranceClaimIntelligence/blob/main/models/cv_results.json).

#### 2C.6 Integration with Other Block(s)

- **Inputs received from other block(s):** Keine (CV ist erster Block in der Pipeline)

- **Outputs provided to other block(s):**
  - An ML: `damage_type_encoded`, `damage_severity`, `cv_confidence` als Features
  - An NLP: `damage_type` → `consistency_score = f(CV_label, NLP_description)` berechnen
  - An RAG: `damage_type` → Query "Deckung für [Schadenstyp] bei [Versicherungstyp]"

---

## 3. Deployment

- **Deployment URL:** https://huggingface.co/spaces/DarrenOG/InsuranceClaim

- **Main user flow:**
  1. Tab "Schadenanalyse": Bild hochladen + Beschreibung + Vertragsdaten → CV/NLP/ML Analyse → Schadenhöhe + Fraud-Score + Entscheidung
  2. Tab "Deckungsauskunft": Schadensart + Versicherungstyp → RAG über AXA AVB → Deckungsauskunft mit AVB-Klausel (3 Prompt-Varianten + 3 Rollen vergleichbar)
  3. Tab "Ergebnisse": Ablation Study + Modellvergleiche + SHAP Feature Importance

- **Screenshot or short demo:**

  Tab 1 — Schadenanalyse (CV + NLP + ML Analyse):
  ![App Tab 1](https://raw.githubusercontent.com/sgDarren/InsuranceClaimIntelligence/main/models/InsuranceClaimIntelligence_1.png)

  Tab 2 — Deckungsauskunft (RAG + AXA AVB):
  ![App Tab 2](https://raw.githubusercontent.com/sgDarren/InsuranceClaimIntelligence/main/models/InsuranceClaimIntelligence_2.png)

  Tab 3 — Ergebnisse & Ablation Study:
  ![App Tab 3](https://raw.githubusercontent.com/sgDarren/InsuranceClaimIntelligence/main/models/InsuranceClaimIntelligence_3.png)

See [`app.py`](https://github.com/sgDarren/InsuranceClaimIntelligence/blob/main/app.py) (HuggingFace-Version), [`src/app/app.py`](https://github.com/sgDarren/InsuranceClaimIntelligence/blob/main/src/app/app.py) (lokale Version).

---

## 4. Execution Instructions

- **Environment setup:**
```bash
git clone https://github.com/sgDarren/InsuranceClaimIntelligence
cd InsuranceClaimIntelligence
pip install -r requirements.txt
cp .env.example .env  # OPENAI_API_KEY eintragen
```

- **Data setup:**
```bash
# ML Dataset
kaggle datasets download mastmustu/insurance-claims-fraud-data -p data/raw/ --unzip

# CV Dataset (automatisch via HuggingFace)
# python src/cv/train.py lädt SaiVaibhavS/comprehensive-car-damage automatisch

# NLP: AXA AVB PDFs werden automatisch heruntergeladen (öffentlich)
```

- **Training command(s):**
```bash
# ML Block (kein GPU nötig, ~10 Min)
python src/ml/train.py

# CV Block (Google Colab T4 GPU erforderlich, ~35 Min)
# → Notebook: notebooks/insurance_claim_intelligence.ipynb (Block B)

# NLP RAG Block (~3 Min)
python src/nlp/rag_pipeline.py
```

- **Inference/run command(s):**
```bash
python src/app/app.py   # Gradio App → http://127.0.0.1:7860
```

- **Reproducibility notes:** SEED=42 in allen Modulen gesetzt. CV Block trainiert auf Google Colab T4 GPU (~35 Min). ML und NLP Blöcke laufen auf CPU (~10 Min resp. ~3 Min). Alle Modelle in `models/` gespeichert. `requirements.txt` mit geprüften Versionen.

---

## 5. Optional Bonus Evidence

- [x] Third selected block implemented with strong quality (NLP als 3. Block: RAG 4.70/5.0)
- [x] More than two data sources used with clear added value (4 Quellen: Kaggle Insurance + HF Car Damage + AXA AVB PDF x2)
- [x] A core section is done exceptionally well (CV Block: 4 Iterationen, GPT-4o Relabeling, Konfusionsmatrix-Analyse)
- [x] Extended evaluation (Ablation Study 4 Experimente, LM-as-Judge 10 Fragen, Classification Report per Klasse)
- [x] Ethics, bias, or fairness analysis
- [x] Creative or exceptional use case (LLM-gestütztes Dataset Relabeling mit GPT-4o für versicherungskonforme Klassen)

**Evidence:**

1. **Alle 3 Blöcke:** CV (79.85% F1=0.85) + ML (R²=0.721, AUC=0.931) + NLP (4.70/5.0) — vollständig implementiert und integriert

2. **Extended Evaluation:**
   - Ablation Study: 4 Experimente Structured→+NLP→+CV→Full Multimodal
   - LM-as-Judge: 10 Versicherungsfragen, 3 Prompt-Varianten bewertet
   - CV: Classification Report pro Klasse + Vergleich 4 Modelle

3. **Ethics/Bias:**
   - DSGVO: Schadenfotos temporär verarbeitet, nicht persistiert
   - Algorithmic Bias: Dataset westlich geprägt → als Limitation dokumentiert
   - Fairness: Demografische Features nicht genutzt (< 5% SHAP Importance)
   - False-Positive: Precision=1.00 priorisiert — kein legitimer Kunde falsch markiert
   - Human-in-the-Loop: Fraud-Score > 0.70 → Manuelle Prüfung
   - Halluzination: RAG mit AVB-Quellen + "Refuse when evidence is thin"

4. **Creative use case:** GPT-4o Vision als Dataset-Annotator — 700 Crushed-Bilder automatisch in versicherungskonforme Klassen relabelt (dent/scratch/crack). Wissenschaftlicher Beitrag: zeigt LLM-gestütztes Dataset-Labeling für Domänen ohne öffentliche Benchmark-Datasets.
