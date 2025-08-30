import json
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
from model import CrossEncoderClassifier  # same model you trained

# -----------------------
# Config
# -----------------------
LABEL_MAP = {
    "Trustworthy": 0,
    "Advertisement": 1,
    "Irrelevant Content": 2,
    "Rant without visit": 3
}
ID2LABEL = {v: k for k, v in LABEL_MAP.items()}

# -----------------------
# Prediction function
# -----------------------
def load_model(model_path, num_labels=4, device="cuda" if torch.cuda.is_available() else "cpu"):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = CrossEncoderClassifier.from_pretrained(model_path, num_labels=num_labels)
    model.to(device)
    model.eval()
    return model, tokenizer, device


def predict(model, tokenizer, device, review, biz_meta, sentiment="Neutral"):
    # same preprocessing as training
    review_text = f"{review} (Sentiment: {sentiment})"
    enc = tokenizer(
        review_text,
        biz_meta,
        padding="max_length",
        truncation=True,
        max_length=256,
        return_tensors="pt"
    ).to(device)
    encoded_input = {
        "input_ids": enc["input_ids"],
        "attention_mask": enc["attention_mask"],
    }
    with torch.no_grad():
        outputs = model(**encoded_input)
        logits = outputs["logits"]
        probs = F.softmax(logits, dim=1).cpu().numpy()[0]

    pred_id = int(torch.argmax(logits, dim=1).cpu().item())
    pred_label = ID2LABEL[pred_id]

    return {
        "prediction": pred_label,
        "probabilities": {ID2LABEL[i]: float(p) for i, p in enumerate(probs)}
    }

# -----------------------
# Batch prediction on JSON
# -----------------------
def predict_from_file(model_path, input_file, output_file=None):
    model, tokenizer, device = load_model(model_path)

    with open(input_file, "r") as f:
        data = json.load(f)

    results = []
    for item in data[:10]:
        review = item["review"]
        biz_meta = item["biz_meta"]
        sentiment = item.get("sentiment", "Neutral")
        result = predict(model, tokenizer, device, review, biz_meta, sentiment)
        results.append({
            "review": review,
            "biz_meta": biz_meta,
            "sentiment": sentiment,
            "result": result
        })

    # save results
    if output_file:
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2)

    return results

# -----------------------
# Example usage
# -----------------------
if __name__ == "__main__":
    model_path = "./checkpoints/best_model_epoch10"  # change to your saved checkpoint
    input_file = "../../datasets/data_for_transformer.json"

    preds = predict_from_file(model_path, input_file, output_file="predictions.json")
    print(json.dumps(preds, indent=2))
