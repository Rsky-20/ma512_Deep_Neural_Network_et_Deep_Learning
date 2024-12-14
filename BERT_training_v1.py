# Import necessary libraries
import pandas as pd
import json
import urllib.request
import numpy as np
import matplotlib.pyplot as plt
import random
import torch
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertForSequenceClassification, AdamW, get_linear_schedule_with_warmup
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from sklearn.metrics import f1_score
import argparse
import os

# Argument configuration
# Using argparse for flexibility in running the script with different configurations
# Example: python main.py --load_parquet --epoch 10 --batch_size 32
parser = argparse.ArgumentParser(description="Data loading configuration")
parser.add_argument("--load_jsonl", action="store_true", help="Load data from a JSONL file via a URL")
parser.add_argument("--load_parquet", action="store_true", help="Load data from a Parquet file")
parser.add_argument("--epoch", type=int, default=10, help="Set the number of training epochs (default: 5)")
parser.add_argument("--batch_size", type=int, default=32, help="Set the batch size (default: 32)")

args = parser.parse_args()

# Enable CUDA for faster computation with NVIDIA GeForce RTX 4060 Ti if available
# If no GPU is detected, fallback to CPU
print("CUDA Available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU Name:", torch.cuda.get_device_name(0))
    torch.backends.cudnn.benchmark = True
else:
    print("No GPU detected.")

# Ensure the output directory for saving models exists
output_directory = 'D:\\programmation\\python\\ma512_Deep_Learning'
model_save_directory = os.path.join(output_directory, "model_save")
os.makedirs(model_save_directory, exist_ok=True)  # Create directory if it doesn't exist

# Data loading
if args.load_jsonl:
    # Load dataset from a JSONL file via a URL
    url = "https://www.sustainablefinance.uzh.ch/dam/jcr:df02e448-baa1-4db8-921a-58507be4838e/climate-fever-dataset-r1.jsonl"
    def read_jsonl_from_url(url):
        data = []
        with urllib.request.urlopen(url) as response:
            for line in response:
                data.append(json.loads(line.decode('utf-8')))
        return data
    df = pd.DataFrame(read_jsonl_from_url(url))

elif args.load_parquet:
    # Load dataset from Parquet files
    splits = {'train': 'data/train-00000-of-00001.parquet', 'test': 'data/test-00000-of-00001.parquet', 'valid': 'data/valid-00000-of-00001.parquet'}
    df = pd.read_parquet("hf://datasets/amandakonet/climate_fever_adopted/" + splits["train"])
    # Preprocessing specific to Parquet files
    df = df.drop(columns=['evidence', 'label', 'category'])
    df = df.rename(columns={'evidence_label': 'claim_label'})
else:
    raise ValueError("Please specify a data loading method using --load_jsonl or --load_parquet.")

# Remove unnecessary words from claims
stop_words = ["the", "in", "on", "to", "of", "that", "it", "its", "by", "is", "a", "The", "and", "for",
             "there", "they", "it", "or", "than", "about", "as", "but", "just", "it", "an", "at", "also",
             "them", "their", "this", "so", "the", "if", "witch", "into", "from", "we", "while", "since",
             "with", "you", "too", "I"]

filtered_claims = []
for claim in df.claim:
    words = claim.split(" ")
    filtered_words = [word for word in words if word.lower() not in stop_words]
    filtered_claims.append(" ".join(filtered_words))

df["filtered_claim"] = filtered_claims

# Preprocessing data
possible_labels = df.claim_label.unique()
label_dict = {possible_label: index for index, possible_label in enumerate(possible_labels)}
df['claim_label'] = df.claim_label.replace(label_dict)

X_train, X_val, y_train, y_val = train_test_split(
    df.claim.values, df.claim_label.values,
    test_size=0.15, random_state=42, stratify=df.claim_label.values
)

train_indices = df.index[df["claim"].isin(X_train)]
val_indices = df.index[df["claim"].isin(X_val)]

df.loc[train_indices, 'data_type'] = 'train'
df.loc[val_indices, 'data_type'] = 'val'

# Initialize tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

# Encode data
encoded_data_train = tokenizer.batch_encode_plus(
    df[df.data_type == 'train'].claim.values,
    add_special_tokens=True,
    return_attention_mask=True,
    padding='max_length',
    max_length=256,
    truncation=True,
    return_tensors='pt'
)

encoded_data_val = tokenizer.batch_encode_plus(
    df[df.data_type == 'val'].claim.values,
    add_special_tokens=True,
    return_attention_mask=True,
    padding='max_length',
    max_length=256,
    truncation=True,
    return_tensors='pt'
)

input_ids_train = encoded_data_train['input_ids']
attention_masks_train = encoded_data_train['attention_mask']
labels_train = torch.tensor(df[df.data_type == 'train'].claim_label.values)

input_ids_val = encoded_data_val['input_ids']
attention_masks_val = encoded_data_val['attention_mask']
labels_val = torch.tensor(df[df.data_type == 'val'].claim_label.values)

dataset_train = TensorDataset(input_ids_train, attention_masks_train, labels_train)
dataset_val = TensorDataset(input_ids_val, attention_masks_val, labels_val)

# Load pre-trained BERT model
model = BertForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    num_labels=len(label_dict),
    output_attentions=False,
    output_hidden_states=False
)

# DataLoader configuration
batch_size = args.batch_size
dataloader_train = DataLoader(dataset_train, sampler=RandomSampler(dataset_train), batch_size=batch_size)
dataloader_validation = DataLoader(dataset_val, sampler=SequentialSampler(dataset_val), batch_size=batch_size)

# Optimizer and scheduler configuration
optimizer = AdamW(model.parameters(), lr=1e-5, eps=1e-8)
epochs = args.epoch

scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=0,
    num_training_steps=len(dataloader_train) * epochs
)

# Transfer model to GPU/CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Define evaluation metrics
def f1_score_func(preds, labels):
    preds_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return f1_score(labels_flat, preds_flat, average='weighted')

def evaluate(dataloader_val):
    model.eval()
    loss_val_total = 0
    predictions, true_vals = [], []

    for batch in dataloader_val:
        batch = tuple(b.to(device) for b in batch)

        inputs = {
            'input_ids': batch[0],
            'attention_mask': batch[1],
            'labels': batch[2],
        }

        with torch.no_grad():
            outputs = model(**inputs)
        loss = outputs.loss
        logits = outputs.logits
        loss_val_total += loss.item()

        logits = logits.detach().cpu().numpy()
        label_ids = inputs['labels'].cpu().numpy()

        predictions.append(logits)
        true_vals.append(label_ids)

    loss_val_avg = loss_val_total / len(dataloader_val)
    predictions = np.concatenate(predictions, axis=0)
    true_vals = np.concatenate(true_vals, axis=0)

    return loss_val_avg, predictions, true_vals

# Train the model
for epoch in tqdm(range(1, epochs + 1)):
    model.train()
    loss_train_total = 0

    progress_bar = tqdm(dataloader_train, desc=f'Epoch {epoch}', leave=False, disable=False)
    for batch in progress_bar:
        model.zero_grad()

        batch = tuple(b.to(device) for b in batch)

        inputs = {
            'input_ids': batch[0],
            'attention_mask': batch[1],
            'labels': batch[2],
        }

        outputs = model(**inputs)
        loss = outputs.loss
        loss_train_total += loss.item()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        progress_bar.set_postfix({'training_loss': f'{loss.item() / len(batch):.3f}'})
    
    # Save the model
    try:
        torch.save(model.state_dict(), f'{model_save_directory}/finetuned_BERT_epoch_{epoch}.model')
    except Exception as e:
        print(f"Error saving model for epoch {epoch}: {e}")

    # Evaluate the model
    loss_train_avg = loss_train_total / len(dataloader_train)
    val_loss, predictions, true_vals = evaluate(dataloader_validation)
    val_f1 = f1_score_func(predictions, true_vals)

    print(f"Epoch {

epoch}:")
    print(f"Training Loss: {loss_train_avg}")
    print(f"Validation Loss: {val_loss}")
    print(f"F1 Score (Weighted): {val_f1}")

# Plot training and validation accuracy
plt.plot(range(1, epochs + 1), [1 - loss for loss in train_accuracies], label="Training Accuracy")
plt.plot(range(1, epochs + 1), val_accuracies, label="Validation Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.title("Accuracy vs Epochs")
plt.show()
