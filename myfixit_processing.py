import json
from dataclasses import dataclass
from pathlib import Path

EXPECTED_MANUAL_KEYS = {"Title", "Guidid", "Category", "Url", "Steps"}
EXPECTED_STEP_KEYS = {"Text_raw", "Order", "StepId"}

_schema_checked = False  # only warn once per process, not once per manual


@dataclass
class Chunk:
    text: str
    guide_title: str
    guide_id: int
    category: str
    subject: str
    source_category: str
    step_order: int
    step_id: int
    url: str


def load_manuals(json_path: Path):
    """Yields one manual dict per line from a MyFixit category JSON file."""
    with open(json_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _check_schema(manual: dict, json_path: Path):
    """One-time sanity check that the keys we rely on actually exist."""
    global _schema_checked
    if _schema_checked:
        return
    _schema_checked = True

    missing_manual_keys = EXPECTED_MANUAL_KEYS - manual.keys()
    if missing_manual_keys:
        print(
            f"[schema warning] {json_path.name}: manual is missing expected "
            f"key(s) {sorted(missing_manual_keys)}. Actual keys present: "
            f"{sorted(manual.keys())}. Chunks built from this file may be "
            f"incomplete - check the dataset's field names."
        )

    steps = manual.get("Steps") or []
    if steps:
        missing_step_keys = EXPECTED_STEP_KEYS - steps[0].keys()
        if missing_step_keys:
            print(
                f"[schema warning] {json_path.name}: a step is missing "
                f"expected key(s) {sorted(missing_step_keys)}. Actual step "
                f"keys: {sorted(steps[0].keys())}."
            )


def process_category_file(json_path: Path, source_category: str) -> list[Chunk]:
    """Turns every step of every manual in a category file into a Chunk."""
    chunks = []
    manuals_seen = 0
    manuals_with_zero_chunks = 0

    for manual in load_manuals(json_path):
        manuals_seen += 1
        _check_schema(manual, json_path)

        title = manual.get("Title", "")
        guide_id = manual.get("Guidid")
        category = manual.get("Category", "")
        subject = manual.get("Subject", "")  # not all manuals have this field
        url = manual.get("Url", "")

        chunks_before = len(chunks)
        for step in manual.get("Steps", []):
            text = step.get("Text_raw", "").strip()
            if not text:
                continue  # skip steps with no usable text
            chunks.append(
                Chunk(
                    text=text,
                    guide_title=title,
                    guide_id=guide_id,
                    category=category,
                    subject=subject,
                    source_category=source_category,
                    step_order=step.get("Order", -1),
                    step_id=step.get("StepId", -1),
                    url=url,
                )
            )
        if len(chunks) == chunks_before:
            manuals_with_zero_chunks += 1

    if manuals_with_zero_chunks:
        print(
            f"[warning] {json_path.name}: {manuals_with_zero_chunks}/"
            f"{manuals_seen} manuals produced zero chunks (no usable step "
            f"text found)."
        )

    return chunks
