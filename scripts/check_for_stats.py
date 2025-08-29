import json

def get_stats(review_data):
    # Load review data from JSON file
    with open(review_data, "r") as f:
        data = json.load(f)

    # Get stats
    num_reviews = len(data)
    num_unique_gmap_ids = len(set(review["meta"]["gmap_id"] for review in data))
    avg_rating = sum(review["review"]["rating"] for review in data) / num_reviews
    max_rating = max(review["review"]["rating"] for review in data)
    min_rating = min(review["review"]["rating"] for review in data)

    # Get top 5 most common categories
    categories = [review["meta"]["category"][0] for review in data if review["meta"]["category"]]
    category_counts = {}
    for category in categories:
        if category in category_counts:
            category_counts[category] += 1
        else:
            category_counts[category] = 1
    top_categories = sorted(category_counts.items(), key=lambda x: x[1], reverse=True)[:5]


    # Print stats
    print(f"Number of reviews: {num_reviews}")
    print(f"Number of unique gmap ids: {num_unique_gmap_ids}")
    print(f"Average rating: {avg_rating:.2f}")
    print(f"Max rating: {max_rating}")
    print(f"Min rating: {min_rating}")
    print("Top 5 most common categories:")
    for category, count in top_categories:
        print(f"{category}: {count}")

if __name__ == "__main__":
    get_stats("../datasets/data_for_gpt-oss.json")