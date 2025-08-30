import json
import re
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import torch

def parse_model_output(full_output):
    """
    Parses a string containing model output to extract the last JSON object with 'label'.
    
    Args:
        full_output (str): The full string output from the model.
        
    Returns:
        tuple: A tuple containing the extracted label and rationale.
               Returns ("Unknown", full_output) as a fallback on parsing failure.
    """
    # The regex is now non-greedy (`*?`) to find individual JSON objects
    # and properly captures everything from the opening to the closing brace.
    matches = re.findall(r'(\{[\s\S]*?"label"[\s\S]*?\})', full_output)
    
    if matches:
        try:
            # Get the last JSON object from the list of all matches
            last_match = matches[-1]
            parsed = json.loads(last_match)
            return parsed.get("label", "Unknown"), parsed.get("rationale", "")
        except json.JSONDecodeError:
            # Handle cases where the last match isn't valid JSON
            return "Unknown", full_output
    else:
        # Fallback if no matching JSON object is found
        return "Unknown", full_output

# --- Begin unchanged script setup ---
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
    max_new_tokens=200
)

with open("../datasets/data_for_ground_truth.json", "r") as f:
    dataset = json.load(f)

results = []
# --- End unchanged script setup ---

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

    Output only a JSON object like this:
    {{
    "label": "<one of the four categories>",
    "rationale": "<brief 1-2 sentence explanation>"
    }}
    """
    outputs = pipe(prompt)
    text = outputs[0]["generated_text"].strip()
    
    # Use the new, robust parsing function
    label, rationale = parse_model_output(text)

    # Return the complete dictionary
    return {
        "gmap_id": entry["review"].get("gmap_id"),
        "user_id": entry["review"].get("user_id"),
        "label": label,
        "rationale": rationale,
        "full_output": text
    }


for i, entry in enumerate(dataset):
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