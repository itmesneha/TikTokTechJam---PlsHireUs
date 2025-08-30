import json

def transform_data(input_file, output_file):
    with open(input_file, 'r') as f:
        data = json.load(f)

    new_data = []
    for item in data:
        meta = item['meta']
        if meta["MISC"] is None:
            misc_str = "No MISC info"
        else:
            misc_str = ""
            for key, value in meta["MISC"].items():
                misc_str += f"{key}: {', '.join(value)}; "
        if meta["category"] is None or len(meta["category"]) == 0:
            category_str = "No category info"
        else:
            category_str = ', '.join(meta["category"])

        if meta["description"] is None:
            meta["description"] = "No description"
        biz_meta = f"desc: {meta['description']}; MISC: {misc_str}; category: {category_str};"
        user_id = item.get("review", {})["user_id"]
        review = item.get("review", {}).get("text", "No review text")
        new_item = {
            "gmap_id": meta["gmap_id"],
            "biz_meta": biz_meta,
            "review": review,
            "user_id": user_id
        }
        new_data.append(new_item)

    with open(output_file, 'w') as f:
        json.dump(new_data, f, indent=4)

def add_labels(input_file, classified_reviews_file, output_file):
    # Load the classified_reviews data
    with open(classified_reviews_file, 'r') as f:
        classified_reviews_data = json.load(f)

    # Load the transformer data
    with open(input_file, 'r') as f:
        transformer_data = json.load(f)

    # Create a dictionary to map gmap_id and user_id to transformer data
    transformer_dict = {}
    for item in transformer_data:
        gmap_id = item['gmap_id']
        user_id = item['user_id']
        transformer_dict[(gmap_id, user_id)] = item

    # Iterate over the classified_reviews data and add the label to the transformer data
    for item in classified_reviews_data:
        gmap_id = item['gmap_id']
        user_id = item['user_id']
        label = item['label']
        confidence = item['confidence']
        rationale = item['rationale']

        # Check if the gmap_id and user_id exist in the transformer data
        if (gmap_id, user_id) in transformer_dict:
            transformer_dict[(gmap_id, user_id)]['label'] = label
            transformer_dict[(gmap_id, user_id)]['confidence'] = confidence
            transformer_dict[(gmap_id, user_id)]['rationale'] = rationale

    # Save the updated transformer data to a new file
    with open(output_file, 'w') as f:
        json.dump(list(transformer_dict.values()), f, indent=4)



if __name__ == "__main__":
    # Call the function
    transform_data('../../datasets/data_for_gpt-oss.json', '../../datasets/data_for_transformer.json')
    add_labels('../../datasets/data_for_transformer.json', '../../datasets/classified_reviews.json', '../../datasets/data_for_transformer.json')





