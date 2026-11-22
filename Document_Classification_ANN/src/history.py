"""Local prediction history. Stores metadata only - never document contents."""
from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path
from typing import Iterable

import pandas as pd

COLUMNS = ["timestamp", "source", "document", "predicted_category", "confidence"]


def add_records(path: Path | str, records: Iterable[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    new_file = not path.exists()
    with open(path, "a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=COLUMNS)
        if new_file:
            writer.writeheader()
        for rec in records:
            writer.writerow({
                "timestamp": rec.get("timestamp") or datetime.now().isoformat(timespec="seconds"),
                "source": rec.get("source", ""),
                "document": rec.get("document", ""),
                "predicted_category": rec.get("predicted_category", ""),
                "confidence": round(float(rec.get("confidence", 0.0)), 4),
            })


def load_history(path: Path | str) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        return pd.DataFrame(columns=COLUMNS)
    return pd.read_csv(path)


def clear_history(path: Path | str) -> None:
    path = Path(path)
    if path.exists():
        path.unlink()
