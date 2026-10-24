"""
inspect_schema.py

Run this once, right after download_myfixit.py, before you trust
myfixit_processing.py to parse everything correctly. Prints the raw keys
of the first manual (and its first step) in each downloaded category
file, so you can eyeball them against EXPECTED_MANUAL_KEYS /
EXPECTED_STEP_KEYS in myfixit_processing.py.

Usage:
    python inspect_schema.py
"""

import json

from config import MYFIXIT_CATEGORIES, MYFIXIT_JSONS_DIR


def main():
    for category in MYFIXIT_CATEGORIES:
        json_path = MYFIXIT_JSONS_DIR / f"{category}.json"
        if not json_path.exists():
            print(f"[skip] {json_path} not found - run download_myfixit.py first")
            continue

        with open(json_path, encoding="utf-8") as f:
            first_line = f.readline().strip()

        manual = json.loads(first_line)
        print(f"\n=== {category} ===")
        print(f"Manual keys: {sorted(manual.keys())}")
        steps = manual.get("Steps") or manual.get("steps") or []
        if steps:
            print(f"Step keys:   {sorted(steps[0].keys())}")
            print(f"Sample step: {json.dumps(steps[0], indent=2)[:400]}")
        else:
            print("No 'Steps' key found - check for a differently-cased or "
                  "named field in the manual keys printed above.")


if __name__ == "__main__":
    main()
