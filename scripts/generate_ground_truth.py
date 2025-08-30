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

with open("../datasets/data_for_ground_truth.json", "r") as f:
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
    outputs = pipe(prompt)
    text = outputs[0]["generated_text"]

    # Store full output
    full_output = text.strip()

    # ✅ Split at marker to extract label and rationale
    try:
        _, after_label_marker = full_output.split("Pick one label only:", 1)
    except ValueError:
        after_label_marker = full_output

    lines = [line.strip() for line in after_label_marker.strip().split("\n") if line.strip()]

    # first line = label
    label = lines[0] if lines else "Unknown"

    # everything after "Explanation" = rationale
    rationale = ""
    remaining_text = "\n".join(lines[1:]) if len(lines) > 1 else ""
    if "Explanation" in remaining_text:
        _, rationale_text = remaining_text.split("Explanation", 1)
        rationale = rationale_text.strip()
    else:
        rationale = remaining_text.strip()

    return {
        "gmap_id": entry["review"].get("gmap_id"),
        "user_id": entry["review"].get("user_id"),
        "label": label,
        "raw_output": rationale,
        "full_output": full_output
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
