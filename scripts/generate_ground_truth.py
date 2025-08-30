import json
from transformers import pipeline
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

device = "mps" if torch.backends.mps.is_available() else "cpu"

model_id = "mistralai/Mistral-7B-Instruct-v0.3"
tokenizer = AutoTokenizer.from_pretrained(model_id)

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16, 
    device_map={"": device}
)

pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=100
)

with open("../datasets/data_for_gpt-oss.json", "r") as f:
    dataset = json.load(f)

results = []

def classify_one(entry):
    prompt = f"""
You classify Google reviews into one of four exclusive categories:
- Trustworthy: Genuine experience at the business.
- Advertisement: Promo/marketing, owner self-post, coupon links.
- Irrelevant Content: Off-topic, spam, wrong place, meaningless.
- Rant without visit: Complaints/opinions with no evidence of visiting.

Input: business metadata JSON
{json.dumps(entry['meta'], indent=2)}

Review JSON
{json.dumps(entry['review'], indent=2)}

Pick one label only and provide a short rationale:
"""
    outputs = pipe(prompt)
    text = outputs[0]["generated_text"]

    # ✅ Extract the actual label
    label = None
    rationale_lines = []
    capture_rationale = False

    for line in text.split("\n"):
        line = line.strip()
        if line in ["Trustworthy", "Advertisement", "Irrelevant Content", "Rant without visit"] and label is None:
            label = line
            capture_rationale = True  # start capturing rationale from next lines
            continue
        if capture_rationale:
            if line:  # non-empty line
                rationale_lines.append(line)

    rationale = " ".join(rationale_lines).strip() if rationale_lines else ""

    if label is None:
        label = "Unknown"

    return {
        "gmap_id": entry["review"].get("gmap_id"),
        "user_id": entry["review"].get("user_id"),
        "label": label,
        "raw_output": rationale
    }


for i, entry in enumerate(dataset[:100]):
    try:
        res = classify_one(entry)
        results.append(res)
    except Exception as e:
        print(f"Error at {i}: {e}")
        continue

    if (i + 1) % 10 == 0:
        with open("../datasets/classified_reviews.json", "w") as f:
            json.dump(results, f, indent=2)
        print(f"Processed and saved {i + 1} reviews")

with open("../datasets/classified_reviews.json", "w") as f:
    json.dump(results, f, indent=2)
print("Done.")
