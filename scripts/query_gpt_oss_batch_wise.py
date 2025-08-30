import json

with open("../datasets/data_for_gpt-oss.json", "r") as f:
    dataset = json.load(f)

print(len(dataset))  # e.g. 20000
print(dataset[0].keys())  # dict_keys(['meta', 'review'])

from openai import OpenAI

client = OpenAI(
    base_url="https://router.huggingface.co/v1",
    api_key="",
)

def classify_one(entry):
    meta_json = json.dumps(entry["meta"], indent=2)
    review_json = json.dumps(entry["review"], indent=2)

    prompt = f"""
You classify Google reviews into one of four exclusive categories:
- Trustworthy: Genuine experience at the business.
- Advertisement: Promo/marketing, owner self-post, coupon links.
- Irrelevant Content: Off-topic, spam, wrong place, meaningless.
- Rant without visit: Complaints/opinions with no evidence of visiting/using the business.
If in doubt, choose the stricter non-Trustworthy label.

Input: business metadata JSON
{meta_json}

Review JSON
{review_json}

Pick one label only to classify the review.
Output in the below JSON format only:

{{
"role": "assistant",
"content": {{
    "gmap_id": "<string>",
    "label": "Trustworthy | Advertisement | Irrelevant Content | Rant without visit",
    "confidence": 0.0,
    "rationale": "<short reason>"
    }}
}}
"""
    completion = client.chat.completions.create(
        model="openai/gpt-oss-120b:cerebras",
        messages=[{"role": "user", "content": prompt}],
    )
    msg = completion.choices[0].message

    try:
        parsed = json.loads(msg.content)
    except Exception:
        return {
            "gmap_id": entry["review"].get("gmap_id"),
            "user_id": entry["review"].get("user_id"),
            "label": "ERROR",
            "confidence": 0.0,
            "rationale": msg.content
        }

    # ✅ flatten structure: pull from "content" + inject user_id + gmap_id
    content = parsed.get("content", {})
    flattened = {
        "gmap_id": entry["review"].get("gmap_id"),
        "user_id": entry["review"].get("user_id"),
        "label": content.get("label"),
        "confidence": content.get("confidence"),
        "rationale": content.get("rationale"),
    }

    return flattened


results = []
for i, entry in enumerate(dataset[:1000]):
    try:
        res = classify_one(entry)
        results.append(res)
    except Exception as e:
        print(f"Error on review {i}: {e}")
        continue

    # Save every 10 results
    if (i+1) % 10 == 0:
        with open("../datasets/classified_reviews.json", "w") as f:
            json.dump(results, f, indent=2)
        print(f"Processed and saved {i+1} reviews")


with open("../datasets/classified_reviews.json", "w") as f:
    json.dump(results, f, indent=2)
