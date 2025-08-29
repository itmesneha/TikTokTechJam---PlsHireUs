# download.py
import requests
from pathlib import Path

def download(url: str, out_dir: str = "../datasets/") -> Path:
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    dest = out_dir / Path(url).name
    tmp  = dest.with_name(dest.name + ".part")

    if dest.exists():
        print(f"Skipping {url}, file already exists at {dest}")
        return dest

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