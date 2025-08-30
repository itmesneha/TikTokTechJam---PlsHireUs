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
        review = item.get("review", {}).get("text", "No review text")
        new_item = {
            "gmap_id": meta["gmap_id"],
            "biz_meta": biz_meta,
            "review": review
        }
        new_data.append(new_item)

    with open(output_file, 'w') as f:
        json.dump(new_data, f, indent=4)





if __name__ == "__main__":
    # Call the function
    transform_data('../../datasets/data_for_gpt-oss.json', '../../datasets/data_for_transformer.json')