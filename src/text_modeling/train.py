import os
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AdamW, get_scheduler
from model import CrossEncoderClassifier  # your CrossEncoderClassifier definition
from tqdm import tqdm


# -----------------------
# Dataset wrapper
# -----------------------

LABEL_MAP = {
    "Trustworthy": 0,
    "Advertisement": 1,
    "Irrelevant Content": 2,
    "Rant without visit": 3
}

class ReviewDataset(Dataset):
    def __init__(self, data, tokenizer, max_len=256):
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        review = item["review"]
        biz_meta = item["biz_meta"]
        sentiment = item.get("sentiment", "Neutral")

        # append sentiment into review (so model can use it)
        review_text = f"{review} (Sentiment: {sentiment})"

        enc = self.tokenizer(
            review_text,
            biz_meta,
            padding="max_length",
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt",
        )

        # Label: 0=Ad, 1=Irrelevant, 2=RantNV, 3=Trustworthy
        label_str = item["label"]                       # e.g. "Ad"
        label = torch.tensor(LABEL_MAP[label_str], dtype=torch.long)
        label = torch.tensor(label, dtype=torch.long)

        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels": label,
        }


# -----------------------
# Training function
# -----------------------

def evaluate(model, dataloader, device):
    model.eval()
    total_loss = 0
    loss_fct = torch.nn.CrossEntropyLoss()
    with torch.no_grad():
        for batch in dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = loss_fct(outputs["logits"], batch["labels"])
            total_loss += loss.item()
    return total_loss / len(dataloader)

def train_model(
    json_file,
    model_name="microsoft/deberta-v3-base",
    num_labels=4,
    batch_size=16,
    lr=2e-5,
    epochs=3,
    device="cuda" if torch.cuda.is_available() else "cpu",
    save_dir="./checkpoints",
):
    # load data
    with open(json_file, "r") as f:
        data = json.load(f)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    dataset = ReviewDataset(data[:10], tokenizer)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # model
    model = CrossEncoderClassifier(model_name=model_name, num_labels=num_labels)
    model.to(device)

    optimizer = AdamW(model.parameters(), lr=lr)
    num_training_steps = epochs * len(dataloader)
    scheduler = get_scheduler(
        "linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
    )

    best_val_loss = float("inf")
    best_model_path = None

    # training loop
    model.train()
    for epoch in range(epochs):
        loop = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
        total_loss = 0
        for batch in loop:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)  # our CrossEncoderClassifier returns dict
            loss = outputs["loss"]

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            loop.set_postfix(loss=loss.item())

        avg_loss = total_loss / len(dataloader)
        val_loss = evaluate(model, dataloader, device)
        print(f"Epoch {epoch+1} finished. Avg loss = {avg_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_path = os.path.join(save_dir, f"best_model_epoch{epoch+1}")
            print(f"  ✅ New best model, saving to {best_model_path}")
            tokenizer.save_pretrained(best_model_path)
            model.save_pretrained(best_model_path)
    return model, tokenizer


# # -----------------------
# # Evaluation (simple)
# # -----------------------
# def evaluate(model, tokenizer, json_file, device="cuda" if torch.cuda.is_available() else "cpu"):
#     with open(json_file, "r") as f:
#         data = json.load(f)

#     dataset = ReviewDataset(data[:10], tokenizer)
#     dataloader = DataLoader(dataset, batch_size=16)

#     model.eval()
#     preds, labels = [], []
#     with torch.no_grad():
#         for batch in dataloader:
#             batch = {k: v.to(device) for k, v in batch.items()}
#             outputs = model(**batch)
#             logits = outputs["logits"]
#             pred = torch.argmax(logits, dim=1).cpu().numpy()
#             true = batch["labels"].cpu().numpy()
#             preds.extend(pred)
#             labels.extend(true)

#     from sklearn.metrics import classification_report
#     print(classification_report(labels, preds, target_names=["Ad", "Irrelevant", "RantNV", "Trustworthy"]))


# -----------------------
# Main
# -----------------------
if __name__ == "__main__":
    json_file = "../../datasets/data_for_transformer.json"

    model, tokenizer = train_model(json_file, epochs=10)

    # print("Evaluating...")
    # evaluate(model, tokenizer, json_file)
