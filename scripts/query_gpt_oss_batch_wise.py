import json
from transformers import pipeline
import torch

# Initialize local pipeline using GPU (MPS or CPU)
pipe = pipeline(
    "text-generation",
    model="openai/gpt-oss-20b",
    torch_dtype="auto",
    device_map="auto"
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

Pick one label only:
"""
    outputs = pipe([{"role": "user", "content": prompt}], max_new_tokens=50)
    text = outputs[0]["generated_text"]
    # Here you might need to parse label from the generated text
    # For quite formatted output, assume text has label structure
    parsed_label = text.strip().split("\n")[0]  # simplistic parsing

    return {
        "gmap_id": entry["review"].get("gmap_id"),
        "user_id": entry["review"].get("user_id"),
        "label": parsed_label,
        "raw_output": text
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
