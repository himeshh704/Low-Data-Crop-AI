"""
Downloads real PlantVillage images from GitHub for training.
Maps Tomato___healthy -> Healthy
     Tomato___Early_blight -> Diseased
     Tomato___Tomato_Yellow_Leaf_Curl_Virus -> Stressed
"""
import urllib.request
import json
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from config import cfg

# Map our 3 classes to PlantVillage folder names on GitHub
CLASS_MAP = {
    "Healthy":  "Tomato___healthy",
    "Diseased": "Tomato___Early_blight",
    "Stressed": "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
}

BASE_API = "https://api.github.com/repos/spMohanty/PlantVillage-Dataset/contents/raw/color/{folder}"
SAMPLES = 50

def download_images(class_name, folder_name, target_dir, n=SAMPLES):
    url = BASE_API.format(folder=folder_name)
    headers = {
        "User-Agent": "Mozilla/5.0",
        "Accept": "application/vnd.github.v3+json"
    }
    print(f"\n[{class_name}] Fetching file list from GitHub...")
    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=15) as r:
            files = json.loads(r.read().decode())
    except Exception as e:
        print(f"  ERROR fetching list: {e}")
        return 0

    if not isinstance(files, list):
        print(f"  ERROR: unexpected response: {files}")
        return 0

    os.makedirs(target_dir, exist_ok=True)
    downloaded = 0
    for f in files[:n]:
        dl_url = f.get("download_url")
        if not dl_url:
            continue
        dest = os.path.join(target_dir, f["name"])
        if os.path.exists(dest):
            downloaded += 1
            continue
        try:
            urllib.request.urlretrieve(dl_url, dest)
            downloaded += 1
            print(f"  [{downloaded}/{n}] {f['name']}", end="\r")
        except Exception as e:
            print(f"  WARN: Failed {f['name']}: {e}")
    print(f"  Done. {downloaded} images saved.")
    return downloaded

if __name__ == "__main__":
    print("=== Downloading Real PlantVillage Images ===")
    total = 0
    for cls_name, folder in CLASS_MAP.items():
        target = os.path.join(cfg.FEW_SHOT_DIR, cls_name)
        # Wipe old synthetic data
        import shutil
        if os.path.exists(target):
            shutil.rmtree(target)
        n = download_images(cls_name, folder, target)
        total += n
    print(f"\n=== Download complete! {total} real images ready. ===")
    print("Now run: python train.py")
