import requests, json, gzip
from pathlib import Path

def download(url: str, out_dir: str = "../datasets/") -> Path:
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    dest = out_dir / Path(url).name
    tmp  = dest.with_name(dest.name + ".part")

    # Resume if partial exists
    headers = {}
    if tmp.exists():
        headers["Range"] = f"bytes={tmp.stat().st_size}-"
        mode = "ab"
    else:
        mode = "wb"

    with requests.get(url, stream=True, allow_redirects=True, headers=headers, timeout=60) as r:
        r.raise_for_status()
        with open(tmp, mode) as f:
            for chunk in r.iter_content(chunk_size=1024*1024):
                if chunk: f.write(chunk)

    tmp.rename(dest)
    print(f"Saved to {dest}")
    return dest


if __name__ == "__main__":
    path_review = download("https://mcauleylab.ucsd.edu/public_datasets/gdrive/googlelocal/review-Alaska_10.json.gz")
    with gzip.open(path_review, "rt", encoding="utf-8") as f:
        for i, line in enumerate(f):
            obj = json.loads(line)  # one JSON object per line
            print(obj)
            if i == 2: break


    path_meta = download("https://mcauleylab.ucsd.edu/public_datasets/gdrive/googlelocal/meta-Alaska.json.gz")
    with gzip.open(path_meta, "rt", encoding="utf-8") as f:
        for i, line in enumerate(f):
            obj = json.loads(line)  # one JSON object per line
            print(obj)
            if i == 2: break