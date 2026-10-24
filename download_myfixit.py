"""
download_myfixit.py

Downloads the selected MyFixit category JSON files directly from the
rub-ksv/MyFixit-Dataset GitHub repo (raw.githubusercontent.com) into
data/myfixit_jsons/. No scraping/ToS concerns here - this is a public
dataset repo meant to be downloaded and used (CC-licensed, cite the paper
in your README).
"""

import requests

from config import MYFIXIT_CATEGORIES, MYFIXIT_BASE_URL, MYFIXIT_JSONS_DIR


def main():
    MYFIXIT_JSONS_DIR.mkdir(parents=True, exist_ok=True)

    for category in MYFIXIT_CATEGORIES:
        file_name = f"{category}.json"
        dest_path = MYFIXIT_JSONS_DIR / file_name

        if dest_path.exists():
            print(f"[skip] already have {file_name}")
            continue

        url = f"{MYFIXIT_BASE_URL}/{file_name.replace(' ', '%20')}"
        print(f"Downloading {category} from {url} ...")
        resp = requests.get(url, timeout=60)
        resp.raise_for_status()
        dest_path.write_bytes(resp.content)
        print(f"  saved {file_name} ({len(resp.content) / 1024:.0f} KB)")

    print("\nDone.")


if __name__ == "__main__":
    main()
