"""Data loading for LexGLUE (SCOTUS, LEDGAR).

We read the parquet files from the Hugging Face Hub directly (no `datasets`
dependency, no loading script) and cache them under data/raw/<dataset>/.
"""
from __future__ import annotations

import re
import urllib.request
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
HUB_URL = "https://huggingface.co/datasets/lighteval/lexglue/resolve/main/{ds}/{split}-00000-of-00001.parquet"
SPLITS = ("train", "validation", "test")
DATASETS = ("scotus", "ledgar")
SEED = 42


def _path(dataset: str, split: str) -> Path:
    return RAW_DIR / dataset / f"{split}.parquet"


def download(dataset: str) -> None:
    """Download the three parquet splits if not already cached."""
    for split in SPLITS:
        dest = _path(dataset, split)
        if dest.exists():
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        url = HUB_URL.format(ds=dataset, split=split)
        print(f"downloading {url}")
        tmp = dest.with_suffix(".part")
        urllib.request.urlretrieve(url, tmp)
        tmp.rename(dest)


def clean_text(text: str) -> str:
    """Light cleaning: collapse whitespace. Keep case/punctuation (models decide)."""
    return re.sub(r"\s+", " ", text).strip()


def load_split(dataset: str, split: str, clean: bool = True) -> pd.DataFrame:
    """Return a DataFrame with columns `text` and `label` (int)."""
    if split not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS}")
    path = _path(dataset, split)
    if not path.exists():
        download(dataset)
    df = pd.read_parquet(path)
    df = df[["text", "label"]].copy()
    if clean:
        df["text"] = df["text"].map(clean_text)
    return df.reset_index(drop=True)


def load_all(dataset: str, clean: bool = True) -> dict[str, pd.DataFrame]:
    return {s: load_split(dataset, s, clean) for s in SPLITS}


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Download LexGLUE parquet files")
    p.add_argument("--dataset", choices=DATASETS, default="scotus")
    download(p.parse_args().dataset)
