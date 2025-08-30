## features
# - Description, MISC and Category
# - Sentiment analysis of user reviews
import pandas as pd
import json
from transformers import pipeline

# assuming your JSON is loaded into a variable called `data`
data = json.load(open("datasets\data_for_gpt-oss.json"))

df = pd.json_normalize(data, sep='_')

# Convert review_time (ms) to readable datetime
df['review_time'] = pd.to_datetime(df['review_time'], unit='ms')

print(df.head())
print(df.columns)

df = df[['meta_name', 'meta_address', 'meta_gmap_id', 'meta_description', 'meta_category', 'meta_avg_rating',
       'meta_num_of_reviews','review_user_id',
       'review_name', 'review_rating', 'review_text',
       'review_pics', 'review_resp', 'review_gmap_id',
       'meta_MISC_Service options', 'meta_MISC_Accessibility',
       'meta_MISC_Amenities', 'meta_MISC_Planning', 'review_resp_time',
       'review_resp_text', 'meta_MISC_Offerings', 'meta_MISC_Payments',
       'meta_MISC_From the business', 'meta_MISC_Health & safety',
       'meta_MISC_Highlights', 'meta_MISC_Popular for',
       'meta_MISC_Dining options', 'meta_MISC_Atmosphere', 'meta_MISC_Crowd',
       'meta_MISC_Lodging options', 'meta_MISC_Health and safety',
       'meta_MISC_Recycling']]

print(df.head())
print(df.columns)
print(df.info())

# # Select all MISC columns
# misc_cols = [col for col in df.columns if col.startswith("meta_MISC_")]

# # Replace NaN with "No" and flatten lists to comma-separated strings
# for col in misc_cols:
#     df[col] = df[col].fillna("No").apply(
#         lambda x: ", ".join(x) if isinstance(x, list) else x
#     )


misc_cols = [col for col in df.columns if col.startswith("meta_MISC_")]
# for col in misc_cols:
#     df[col] = df[col].fillna("").apply(
#         lambda x: ", ".join([c.strip() for c in x]) if isinstance(x, list) else str(x).strip()
#     )

# # When combining, also strip the final result
# df['business_features'] = (
#     df['meta_description'].fillna("").str.strip() + ", " +
#     df['meta_category'] + ", " +
#     df[misc_cols].apply(lambda row: " ".join(row.values.astype(str)), axis=1)
# ).str.strip()

# print(df['business_features'].unique())

def build_features(row):
    parts = []
    
    # Description
    if pd.notna(row["meta_description"]) and str(row["meta_description"]).strip() != "":
        parts.append(str(row["meta_description"]).strip())
    
    # Category (can be list)
    cat = row["meta_category"]
    if isinstance(cat, list):
        cat_str = ", ".join([str(x).strip() for x in cat if str(x).strip() not in ("", "nan")])
        if cat_str:
            parts.append(cat_str)
    elif pd.notna(cat) and str(cat).strip() not in ("", "nan"):
        parts.append(str(cat).strip())
    
    # MISC fields
    misc_texts = []
    for v in row[misc_cols]:
        if isinstance(v, list):
            misc_texts.append(", ".join([str(x).strip() for x in v if str(x).strip() not in ("", "nan")]))
        elif pd.notna(v) and str(v).strip() not in ("", "nan"):
            misc_texts.append(str(v).strip())
    
    if misc_texts:
        parts.append(" ".join(misc_texts))
    
    # Join all non-empty parts with ", "
    return ", ".join(parts)

# Apply to build business_features
df["business_features"] = df.apply(build_features, axis=1)




# Initialize sentiment analysis pipeline
sentiment_pipeline = pipeline("sentiment-analysis")

def get_sentiment_label(text):
    if not isinstance(text, str) or text.strip() == "":
        return "NEUTRAL"
    result = sentiment_pipeline(text[:512])[0]  # truncate to 512 chars for efficiency
    # result['label'] can be e.g., 'POSITIVE', 'NEGATIVE', 'NEUTRAL' (depends on model)
    return result['label'].upper()

df['review_sentiment'] = df['review_text'].apply(get_sentiment_label)
# print(df[['review_text', 'review_sentiment']].head(10))

def rating_to_sentiment(rating):
    if rating <= 2:
        return "NEGATIVE"
    elif rating == 3:
        return "NEUTRAL"
    elif rating >= 4:
        return "POSITIVE"
    else:
        return "NEUTRAL"

df['rating_sentiment'] = df['review_rating'].apply(rating_to_sentiment)

# Example: flag matches / mismatches
df['sentiment_match'] = df['review_sentiment'] == df['rating_sentiment']

# Summary statistics
match_rate = df['sentiment_match'].mean()
print(f"Percentage of review sentiments matching user ratings: {match_rate*100:.2f}%")
