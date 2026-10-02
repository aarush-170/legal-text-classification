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


def label_names(dataset: str) -> list[str]:
    """Class names in a fixed order (numeric order if all names are digits, else alphabetical).

    The lighteval parquet stores each row's label as a string in `gold` and the full
    class list in `references`; we derive the id mapping from the train split.
    """
    df = pd.read_parquet(_path(dataset, "train"), columns=["references"])
    names = [str(x) for x in df["references"].iloc[0]]
    return sorted(names, key=int) if all(n.isdigit() for n in names) else sorted(names)


def load_split(dataset: str, split: str, clean: bool = True) -> pd.DataFrame:
    """Return a DataFrame with columns `text`, `label` (int id) and `label_name`."""
    if split not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS}")
    path = _path(dataset, split)
    if not path.exists():
        download(dataset)
    raw = pd.read_parquet(path, columns=["input", "gold"])
    names = label_names(dataset)
    ids = {n: i for i, n in enumerate(names)}
    df = pd.DataFrame({
        "text": raw["input"],
        "label_name": [str(g[0]) for g in raw["gold"]],
    })
    df["label"] = df["label_name"].map(ids).astype(int)
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
