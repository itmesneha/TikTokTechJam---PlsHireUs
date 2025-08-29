# process.py
import json, gzip
from download_data import download

def process_review_data(paths):
    # Load metadata into memory
    meta_data = {}
    for meta_path in paths["metas"]:
        path_meta = download(meta_path)
        with gzip.open(path_meta, "rt", encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line)
                meta_data[obj["gmap_id"]] = obj

    # Process review data
    review_data = []
    for review_path in paths["reviews"]:
        path_review = download(review_path)
        with gzip.open(path_review, "rt", encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line)
                gmap_id = obj["gmap_id"]
                meta = meta_data.get(gmap_id)
                if meta:
                    review_data.append({"meta": meta, "review": obj})

    # Save review data to JSON file
    with open("../datasets/data_for_gpt-oss.json", "w") as f:
        json.dump(review_data, f)

if __name__ == "__main__":
    # Load paths from JSON file
    with open("data_for_download.json", "r") as f:
        paths = json.load(f)

    process_review_data(paths)