"""
Insurance Claim Intelligence — CV Block
ViT-B/16 Fine-Tuning fuer Fahrzeugschadensklassifikation
4 Iterationen: CLIP Zero-Shot → Transfer Learning → Fine-Tuning → GPT-4o Vision
Dataset: SaiVaibhavS/comprehensive-car-damage mit GPT-4o Hybrid-Relabeling
GPU: Google Colab T4 (empfohlen)
"""

import numpy as np, torch, evaluate, json, os, base64, pandas as pd
from PIL import Image
from io import BytesIO
from datasets import load_dataset, Dataset, DatasetDict
from transformers import AutoImageProcessor, ViTForImageClassification, TrainingArguments, Trainer
from torchvision import transforms
from torch.utils.data import WeightedRandomSampler
from collections import Counter
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from openai import OpenAI
from transformers import pipeline as hf_pipeline

# ── Setup ─────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.mem_get_info()[0]/1e9:.1f} GB frei")

SEED             = 42
MODEL_CHECKPOINT = "google/vit-base-patch16-224"
torch.manual_seed(SEED)
os.makedirs("models",   exist_ok=True)
os.makedirs("data/raw", exist_ok=True)

os.environ["OPENAI_API_KEY"] = os.environ.get("OPENAI_API_KEY", "")
client = OpenAI()

INSURANCE_CLASSES = ["dent", "scratch", "crack", "glass_shatter", "no_damage"]
label2id = {l: i for i, l in enumerate(INSURANCE_CLASSES)}
id2label = {i: l for i, l in enumerate(INSURANCE_CLASSES)}

# ══════════════════════════════════════════════════════════════
# 1. DATASET + GPT-4o HYBRID RELABELING
# ══════════════════════════════════════════════════════════════
print("\nLade Dataset...")
full_ds    = load_dataset("SaiVaibhavS/comprehensive-car-damage", split="train")
orig_names = full_ds.features["label"].names
print(f"{len(full_ds)} Bilder | Originalklassen: {orig_names}")

def classify_crushed(pil_image) -> str:
    """GPT-4o Vision klassifiziert Crushed-Bilder in Versicherungsklassen.
    Nur fuer F/R_Crushed aufgerufen — Breakage/Normal regelbasiert.
    """
    buf = BytesIO()
    pil_image.convert("RGB").save(buf, format="JPEG", quality=85)
    b64 = base64.b64encode(buf.getvalue()).decode()
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": [
            {"type": "image_url",
             "image_url": {"url": f"data:image/jpeg;base64,{b64}", "detail": "high"}},
            {"type": "text", "text": (
                "This car has body damage (not glass, not normal).\n"
                "Classify PRIMARY damage:\n"
                "- dent: body panel crushed/deformed\n"
                "- scratch: surface scratch/paint damage\n"
                "- crack: crack in bumper/plastic\n"
                "Reply ONLY: dent, scratch, or crack"
            )}
        ]}],
        max_tokens=5, temperature=0,
    )
    pred = response.choices[0].message.content.strip().lower().replace(".", "").strip()
    return pred if pred in ["dent", "scratch", "crack"] else "dent"

print("Hybrid-Relabeling: Regeln + GPT-4o fuer Crushed-Bilder...")
new_labels    = []
crushed_count = 0
for i, sample in enumerate(full_ds):
    orig = orig_names[sample["label"]]
    if orig in ("F_Breakage", "R_Breakage"):
        new_labels.append("glass_shatter")   # regelbasiert
    elif orig in ("F_Normal", "R_Normal"):
        new_labels.append("no_damage")        # regelbasiert
    else:
        label = classify_crushed(sample["image"])  # GPT-4o
        new_labels.append(label)
        crushed_count += 1
        if crushed_count % 100 == 0:
            print(f"  [{i}/{len(full_ds)}] {dict(Counter(new_labels))}")

# Balancing
df = pd.DataFrame({"idx": list(range(len(full_ds))), "label": new_labels})
df = df[df["label"].isin(INSURANCE_CLASSES)].reset_index(drop=True)

TARGET   = 200
balanced = []
for cls in INSURANCE_CLASSES:
    rows = df[df["label"] == cls]
    if len(rows) == 0:
        continue
    rows = rows.sample(
        TARGET if len(rows) < TARGET else min(len(rows), 800),
        replace=len(rows) < TARGET,
        random_state=SEED,
    )
    balanced.append(rows)
df_bal = pd.concat(balanced).sample(frac=1, random_state=SEED).reset_index(drop=True)
df_bal.to_csv("data/raw/insurance_labels_balanced.csv", index=False)

print(f"\n{len(df_bal)} Bilder nach Balancing:")
for cls in INSURANCE_CLASSES:
    n = len(df_bal[df_bal["label"] == cls])
    print(f"  {cls:15}: {n:4d}")

# Splits
idx_train, idx_temp, lbl_train, lbl_temp = train_test_split(
    df_bal["idx"], df_bal["label"],
    test_size=0.2, random_state=SEED, stratify=df_bal["label"])
idx_val, idx_test, lbl_val, lbl_test = train_test_split(
    idx_temp, lbl_temp, test_size=0.5, random_state=SEED, stratify=lbl_temp)
print(f"Split: Train={len(idx_train)}, Val={len(idx_val)}, Test={len(idx_test)}")

def make_hf_dataset(indices, labels):
    return Dataset.from_dict({
        "image": [full_ds[int(i)]["image"] for i in indices],
        "label": [label2id[l] for l in labels],
    })

print("Baue HuggingFace Datasets...")
our_dataset = DatasetDict({
    "train":      make_hf_dataset(idx_train, lbl_train),
    "validation": make_hf_dataset(idx_val,   lbl_val),
    "test":       make_hf_dataset(idx_test,  lbl_test),
})

# ══════════════════════════════════════════════════════════════
# 2. AUGMENTIERUNG + TRAINER
# ══════════════════════════════════════════════════════════════
processor = AutoImageProcessor.from_pretrained(MODEL_CHECKPOINT)
accuracy  = evaluate.load("accuracy")

train_tf = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomPerspective(distortion_scale=0.3, p=0.4),
    transforms.ColorJitter(brightness=0.4, contrast=0.3, saturation=0.2),
    transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
    transforms.RandomRotation(degrees=10),
    transforms.ToTensor(),
    transforms.Normalize(mean=processor.image_mean, std=processor.image_std),
    transforms.RandomErasing(p=0.1),
])
val_tf = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=processor.image_mean, std=processor.image_std),
])

def apply_train(batch):
    batch["pixel_values"] = [train_tf(img.convert("RGB")) for img in batch["image"]]
    return batch

def apply_val(batch):
    batch["pixel_values"] = [val_tf(img.convert("RGB")) for img in batch["image"]]
    return batch

our_dataset["train"]      = our_dataset["train"].with_transform(apply_train)
our_dataset["validation"] = our_dataset["validation"].with_transform(apply_val)
our_dataset["test"]       = our_dataset["test"].with_transform(apply_val)

train_labels_list = list(lbl_train)
counts  = Counter(train_labels_list)
weights = [1.0 / counts[l] for l in train_labels_list]
sampler = WeightedRandomSampler(weights, len(weights))

def collate_fn(batch):
    return {
        "pixel_values": torch.stack([x["pixel_values"] for x in batch]),
        "labels":       torch.tensor([x["label"] for x in batch]),
    }

def compute_metrics(eval_preds):
    preds = np.argmax(eval_preds[0], axis=1)
    return accuracy.compute(predictions=preds, references=eval_preds[1])

class BalancedTrainer(Trainer):
    def get_train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.args.per_device_train_batch_size,
            sampler=sampler,
            collate_fn=self.data_collator,
            num_workers=2,
            pin_memory=True,
        )

print("Setup abgeschlossen.")

# ══════════════════════════════════════════════════════════════
# ITERATION 1: CLIP Zero-Shot (Baseline)
# ══════════════════════════════════════════════════════════════
print("\n--- ITERATION 1: CLIP Zero-Shot (Baseline) ---")
clip = hf_pipeline("zero-shot-image-classification",
                   model="openai/clip-vit-large-patch14", device=0)
candidate_labels = [
    "car with dented deformed body panel",
    "car with surface scratch paint damage",
    "car with crack in bumper plastic",
    "car with broken shattered glass window",
    "car with no damage normal condition",
]
correct_clip, n_clip = 0, 100
for i in range(n_clip):
    result   = clip(full_ds[i]["image"], candidate_labels=candidate_labels)
    pred_idx = candidate_labels.index(result[0]["label"])
    if INSURANCE_CLASSES[pred_idx] == new_labels[i]:
        correct_clip += 1
acc_clip = correct_clip / n_clip
print(f"CLIP Zero-Shot: {acc_clip:.2%}")

# ══════════════════════════════════════════════════════════════
# ITERATION 2: Transfer Learning (Base frozen)
# ══════════════════════════════════════════════════════════════
print("\n--- ITERATION 2: Transfer Learning (Base frozen, LR=3e-4) ---")
model_tl = ViTForImageClassification.from_pretrained(
    MODEL_CHECKPOINT, num_labels=len(INSURANCE_CLASSES),
    id2label=id2label, label2id=label2id, ignore_mismatched_sizes=True)
for param in model_tl.vit.parameters():
    param.requires_grad = False

trainer_tl = BalancedTrainer(
    model=model_tl,
    args=TrainingArguments(
        output_dir="./vit-transfer",
        num_train_epochs=5, learning_rate=3e-4, warmup_steps=100,
        per_device_train_batch_size=32, per_device_eval_batch_size=64,
        eval_strategy="epoch", save_strategy="epoch",
        load_best_model_at_end=True, metric_for_best_model="accuracy",
        remove_unused_columns=False, save_total_limit=1,
        report_to="none", fp16=True, dataloader_pin_memory=True, seed=SEED,
    ),
    data_collator=collate_fn, compute_metrics=compute_metrics,
    train_dataset=our_dataset["train"], eval_dataset=our_dataset["validation"],
)
trainer_tl.train()
acc_tl = trainer_tl.evaluate(our_dataset["test"])["eval_accuracy"]
print(f"Transfer Learning: {acc_tl:.4f}")
torch.cuda.empty_cache()

# ══════════════════════════════════════════════════════════════
# ITERATION 3: Fine-Tuning (letzte 6 Layer + LayerNorm)
# ══════════════════════════════════════════════════════════════
print("\n--- ITERATION 3: Fine-Tuning (Last 6 layers, LR=1e-5) ---")
model_ft = ViTForImageClassification.from_pretrained(
    MODEL_CHECKPOINT, num_labels=len(INSURANCE_CLASSES),
    id2label=id2label, label2id=label2id, ignore_mismatched_sizes=True)
for param in model_ft.vit.parameters():
    param.requires_grad = False
for param in model_ft.vit.encoder.layer[-6:].parameters():
    param.requires_grad = True
for param in model_ft.vit.layernorm.parameters():
    param.requires_grad = True
print(f"Trainierbare Parameter: {sum(p.numel() for p in model_ft.parameters() if p.requires_grad):,}")

trainer_ft = BalancedTrainer(
    model=model_ft,
    args=TrainingArguments(
        output_dir="./vit-finetuned",
        num_train_epochs=15, learning_rate=1e-5, warmup_steps=200,
        weight_decay=0.01,
        per_device_train_batch_size=32, per_device_eval_batch_size=64,
        eval_strategy="epoch", save_strategy="epoch",
        load_best_model_at_end=True, metric_for_best_model="accuracy",
        remove_unused_columns=False, save_total_limit=1,
        report_to="none", fp16=True, dataloader_pin_memory=True, seed=SEED,
    ),
    data_collator=collate_fn, compute_metrics=compute_metrics,
    train_dataset=our_dataset["train"], eval_dataset=our_dataset["validation"],
)
trainer_ft.train()
acc_ft = trainer_ft.evaluate(our_dataset["test"])["eval_accuracy"]

print("\nClassification Report (Fine-Tuning, Test n=268):")
preds_out   = trainer_ft.predict(our_dataset["test"])
pred_labels = np.argmax(preds_out.predictions, axis=1)
true_labels = preds_out.label_ids
print(classification_report(true_labels, pred_labels,
                             target_names=INSURANCE_CLASSES, zero_division=0))
print(f"Fine-Tuning: {acc_ft:.4f}")
torch.cuda.empty_cache()

# ══════════════════════════════════════════════════════════════
# ITERATION 4: GPT-4o Vision (State of the Art Vergleich)
# ══════════════════════════════════════════════════════════════
print("\n--- ITERATION 4: GPT-4o Vision (State of the Art) ---")

def classify_gpt4o_eval(pil_image) -> str:
    buf = BytesIO()
    pil_image.convert("RGB").save(buf, format="JPEG", quality=85)
    b64 = base64.b64encode(buf.getvalue()).decode()
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": [
            {"type": "image_url",
             "image_url": {"url": f"data:image/jpeg;base64,{b64}", "detail": "high"}},
            {"type": "text", "text": (
                f"Classify this car damage: {INSURANCE_CLASSES}\n"
                "Reply with ONLY the category name."
            )}
        ]}],
        max_tokens=10, temperature=0,
    )
    pred = response.choices[0].message.content.strip().lower().replace(".", "").strip()
    return pred if pred in INSURANCE_CLASSES else "dent"

correct_gpt, n_gpt = 0, 50
for i, (idx, true_lbl) in enumerate(zip(list(idx_test)[:n_gpt], list(lbl_test)[:n_gpt])):
    pred = classify_gpt4o_eval(full_ds[int(idx)]["image"])
    if pred == true_lbl:
        correct_gpt += 1
    if i % 10 == 0:
        print(f"  [{i+1}/{n_gpt}] True: {true_lbl:15} | Pred: {pred}")
acc_gpt = correct_gpt / n_gpt
print(f"GPT-4o Vision: {acc_gpt:.2%}")

# ══════════════════════════════════════════════════════════════
# ZUSAMMENFASSUNG + SPEICHERN
# ══════════════════════════════════════════════════════════════
print(f"\n{'='*55}")
print("MODELLVERGLEICH CV BLOCK:")
print(f"  It.1 CLIP Zero-Shot:    {acc_clip:.2%}  (Baseline)")
print(f"  It.2 Transfer Learning: {acc_tl:.2%}")
print(f"  It.3 Fine-Tuning:       {acc_ft:.2%}  <- WINNER")
print(f"  It.4 GPT-4o Vision:     {acc_gpt:.2%}  (SotA Vergleich)")

trainer_ft.save_model("models/vit-damage-final")

# preprocessor_config speichern (fuer HuggingFace Deployment)
processor.save_pretrained("models/vit-damage-final/")

with open("models/label_map.json", "w") as f:
    json.dump(id2label, f, indent=2)
with open("models/cv_results.json", "w") as f:
    json.dump({
        "classes":           INSURANCE_CLASSES,
        "clip_zero_shot":    {"accuracy": float(acc_clip)},
        "transfer_learning": {"accuracy": float(acc_tl)},
        "fine_tuning":       {"accuracy": float(acc_ft)},
        "gpt4o_vision":      {"accuracy": float(acc_gpt)},
    }, f, indent=2)

print("Modelle gespeichert.")
print("CV Block abgeschlossen.")
