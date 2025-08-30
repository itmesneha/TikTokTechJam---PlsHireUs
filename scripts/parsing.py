import re
import json

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

# Example test outputs from the model
test_outputs = {
    "gmap_id": "0x56c8992b5dee7225:0x9f7f4bf151868cf7",
    "user_id": "103655755742322578001",
    "label": "Unknown",
    "raw_output": "You classify Google reviews into one of four exclusive categories:\n    - Trustworthy: Genuine experience at the business.\n    - Advertisement: Promo/marketing, owner self-post, coupon links.\n    - Irrelevant Content: Off-topic, spam, wrong place, meaningless.\n    - Rant without visit: Complaints/opinions with no evidence of visiting.\n\n    Input: business metadata JSON\n    {\n  \"name\": \"Anchorage Market\",\n  \"address\": \"Anchorage Market, 88th Ave, Anchorage, AK 99515\",\n  \"gmap_id\": \"0x56c8992b5dee7225:0x9f7f4bf151868cf7\",\n  \"description\": null,\n  \"latitude\": 61.1414349,\n  \"longitude\": -149.8684816,\n  \"category\": [\n    \"Farmers' market\"\n  ],\n  \"avg_rating\": 4.2,\n  \"num_of_reviews\": 18,\n  \"price\": null,\n  \"hours\": [\n    [\n      \"Thursday\",\n      \"Closed\"\n    ],\n    [\n      \"Friday\",\n      \"10AM\\u20135PM\"\n    ],\n    [\n      \"Saturday\",\n      \"10AM\\u20135PM\"\n    ],\n    [\n      \"Sunday\",\n      \"10AM\\u20135PM\"\n    ],\n    [\n      \"Monday\",\n      \"Closed\"\n    ],\n    [\n      \"Tuesday\",\n      \"Closed\"\n    ],\n    [\n      \"Wednesday\",\n      \"Closed\"\n    ]\n  ],\n  \"MISC\": {\n    \"Service options\": [\n      \"In-store shopping\"\n    ],\n    \"Accessibility\": [\n      \"Wheelchair accessible entrance\"\n    ]\n  },\n  \"state\": \"Closed \\u22c5 Opens 10AM Fri\",\n  \"relative_results\": null,\n  \"url\": \"https://www.google.com/maps/place//data=!4m2!3m1!1s0x56c8992b5dee7225:0x9f7f4bf151868cf7?authuser=-1&hl=en&gl=us\"\n}\n\n    Review JSON\n    {\n  \"user_id\": \"103655755742322578001\",\n  \"name\": \"John Deal\",\n  \"time\": 1631131711388,\n  \"rating\": 3,\n  \"text\": \"It's a market\",\n  \"pics\": null,\n  \"resp\": null,\n  \"gmap_id\": \"0x56c8992b5dee7225:0x9f7f4bf151868cf7\"\n}\n\n    Output only a JSON object like this:\n    {\n    \"label\": \"<one of the four categories>\",\n    \"rationale\": \"<brief 1-2 sentence explanation>\"\n    }\n     {\n    \"label\": \"Irrelevant Content\",\n    \"rationale\": \"The review does not provide a specific experience or opinion about the business.\"\n    }",
    "full_output": "You classify Google reviews into one of four exclusive categories:\n    - Trustworthy: Genuine experience at the business.\n    - Advertisement: Promo/marketing, owner self-post, coupon links.\n    - Irrelevant Content: Off-topic, spam, wrong place, meaningless.\n    - Rant without visit: Complaints/opinions with no evidence of visiting.\n\n    Input: business metadata JSON\n    {\n  \"name\": \"Anchorage Market\",\n  \"address\": \"Anchorage Market, 88th Ave, Anchorage, AK 99515\",\n  \"gmap_id\": \"0x56c8992b5dee7225:0x9f7f4bf151868cf7\",\n  \"description\": null,\n  \"latitude\": 61.1414349,\n  \"longitude\": -149.8684816,\n  \"category\": [\n    \"Farmers' market\"\n  ],\n  \"avg_rating\": 4.2,\n  \"num_of_reviews\": 18,\n  \"price\": null,\n  \"hours\": [\n    [\n      \"Thursday\",\n      \"Closed\"\n    ],\n    [\n      \"Friday\",\n      \"10AM\\u20135PM\"\n    ],\n    [\n      \"Saturday\",\n      \"10AM\\u20135PM\"\n    ],\n    [\n      \"Sunday\",\n      \"10AM\\u20135PM\"\n    ],\n    [\n      \"Monday\",\n      \"Closed\"\n    ],\n    [\n      \"Tuesday\",\n      \"Closed\"\n    ],\n    [\n      \"Wednesday\",\n      \"Closed\"\n    ]\n  ],\n  \"MISC\": {\n    \"Service options\": [\n      \"In-store shopping\"\n    ],\n    \"Accessibility\": [\n      \"Wheelchair accessible entrance\"\n    ]\n  },\n  \"state\": \"Closed \\u22c5 Opens 10AM Fri\",\n  \"relative_results\": null,\n  \"url\": \"https://www.google.com/maps/place//data=!4m2!3m1!1s0x56c8992b5dee7225:0x9f7f4bf151868cf7?authuser=-1&hl=en&gl=us\"\n}\n\n    Review JSON\n    {\n  \"user_id\": \"103655755742322578001\",\n  \"name\": \"John Deal\",\n  \"time\": 1631131711388,\n  \"rating\": 3,\n  \"text\": \"It's a market\",\n  \"pics\": null,\n  \"resp\": null,\n  \"gmap_id\": \"0x56c8992b5dee7225:0x9f7f4bf151868cf7\"\n}\n\n    Output only a JSON object like this:\n    {\n    \"label\": \"<one of the four categories>\",\n    \"rationale\": \"<brief 1-2 sentence explanation>\"\n    }\n     {\n    \"label\": \"Irrelevant Content\",\n    \"rationale\": \"The review does not provide a specific experience or opinion about the business.\"\n    }"
}

# Correctly call the function on the 'full_output' value of the dictionary
label, rationale = parse_model_output(test_outputs["full_output"])
print("-" * 40)
print("Label:", label)
print("Rationale:", rationale)
print("-" * 40)