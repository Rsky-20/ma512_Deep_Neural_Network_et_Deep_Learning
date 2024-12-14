import json
import urllib.request
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, pipeline
from datasets import Dataset, DatasetDict, Features, ClassLabel, Value
import evaluate
import numpy as np
import torch
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk

# Télécharger les ressources nécessaires pour NLTK
nltk.download("stopwords")
nltk.download("punkt")

# --- Configuration ---
MODEL_NAME = "bert-base-uncased"  # Remplacez par "amandakonet/climatebert-fact-checking" si vous avez accès
LABELS = ["SUPPORTS", "NOT_ENOUGH_INFO", "REFUTES", "DISPUTED"]
ID2LABEL = {i: label for i, label in enumerate(LABELS)}
LABEL2ID = {label: i for i, label in enumerate(LABELS)}
EPOCHS = 1
BATCH_SIZE = 16
LEARNING_RATE = 5e-5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- Charger le dataset ---
def read_jsonl_from_url(url):
    data = []
    with urllib.request.urlopen(url) as response:
        for line in response:
            data.append(json.loads(line.decode("utf-8")))
    return data

url = "https://www.sustainablefinance.uzh.ch/dam/jcr:df02e448-baa1-4db8-921a-58507be4838e/climate-fever-dataset-r1.jsonl"
data = read_jsonl_from_url(url)
df = pd.DataFrame(data)

# --- Préparer les données ---
# Stop words personnalisés
custom_stop_words = set(["the", "in", "on", "to", "of", "that", "it", "by", "is", "a", "and", "for", "as"])
stop_words = set(stopwords.words("english")).union(custom_stop_words)

# Fonction pour nettoyer les affirmations
def clean_text(text):
    tokens = word_tokenize(text.lower())  # Convertir en minuscules et tokeniser
    filtered_tokens = [word for word in tokens if word not in stop_words and word not in string.punctuation]
    return " ".join(filtered_tokens)

# Appliquer le nettoyage
df["cleaned_claim"] = df["claim"].apply(clean_text)
df["cleaned_evidence"] = df["evidences"].apply(lambda evidences: clean_text(evidences[0]["evidence"]) if evidences else "")

# Préparer les labels
df["label"] = df["claim_label"].map(LABEL2ID)

# Fraction pour les ensembles de validation et test
val_frac = 0.15  # 15% pour validation
test_frac = 0.15  # 15% pour test

# Séparer les données
train_df = df.sample(frac=1 - val_frac - test_frac, random_state=42)
remaining_df = df.drop(train_df.index).reset_index(drop=True)
val_df = remaining_df.sample(frac=val_frac / (val_frac + test_frac), random_state=42)
test_df = remaining_df.drop(val_df.index).reset_index(drop=True)

# Vérification des proportions
print("Train size:", len(train_df))
print("Validation size:", len(val_df))
print("Test size:", len(test_df))

# Convertir les DataFrames en Datasets Hugging Face
def convert_to_dataset(df):
    df = df.reset_index(drop=True)
    return Dataset.from_pandas(df[["cleaned_claim", "cleaned_evidence", "label"]], features=Features({
        "cleaned_claim": Value("string"),
        "cleaned_evidence": Value("string"),
        "label": ClassLabel(names=LABELS)
    }))

datasets = DatasetDict({
    "train": convert_to_dataset(train_df),
    "validation": convert_to_dataset(val_df),
    "test": convert_to_dataset(test_df)
})

# --- Tokenisation ---
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def preprocess_function(examples):
    return tokenizer(examples["cleaned_claim"], examples["cleaned_evidence"], 
                     truncation=True, padding="max_length", max_length=128)

tokenized_datasets = datasets.map(preprocess_function, batched=True)

# --- Charger le modèle ---
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=len(LABELS),
    id2label=ID2LABEL,
    label2id=LABEL2ID
)

# --- Définir les arguments d'entraînement ---
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=LEARNING_RATE,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    num_train_epochs=EPOCHS,
    weight_decay=0.01,
    save_strategy="epoch",
    logging_dir="./logs",
    load_best_model_at_end=True,
    logging_steps=5,
    save_total_limit=2,
    report_to="none",  # Désactiver les rapports vers wandb ou autres par défaut
    fp16=True if DEVICE == "cuda" else False
)

# --- Définir la métrique ---
metric = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    accuracy = metric.compute(predictions=predictions, references=labels)
    return accuracy

# --- Entraîner le modèle ---
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

try:
    trainer.train()
except Exception as e:
    print("Erreur pendant l'entraînement :", e)

# --- Sauvegarder le modèle ---
model.save_pretrained("climate_classifier_model")
tokenizer.save_pretrained("climate_classifier_model")

# --- Tester et évaluer ---
try:
    test_results = trainer.evaluate(tokenized_datasets["test"])
    print("Résultats sur le jeu de test :", test_results)
except Exception as e:
    print("Erreur pendant l'évaluation :", e)


# Exemple de prédiction
def predict_claims(claims):
    model.to(DEVICE)  # Déplacer le modèle sur l'appareil spécifié
    inputs = []
    for claim in claims:
        inputs.append(
            tokenizer(
                claim,
                padding="max_length",
                truncation=True,
                max_length=128,
                return_tensors="pt"
            ).to(DEVICE)  # Déplacer les entrées sur l'appareil
        )
    model.eval()
    with torch.no_grad():
        for i, input_data in enumerate(inputs):
            logits = model(**input_data).logits
            predictions = torch.argmax(logits, dim=-1).item()
            print(f"Claim: {claims[i]}")
            print(f"Prédiction: {LABELS[predictions]}")

example_claims = [
    "Global warming is driving polar bears toward extinction.",
    "The sun has gone into lockdown.",
    "The polar bear population has been growing."
]

predict_claims(example_claims)


