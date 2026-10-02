"""EDA: split sizes, label distribution, text-length stats.

Usage: python -m src.eda --dataset scotus
Writes results/eda_<dataset>.json and results/eda_<dataset>_*.png
"""
from __future__ import annotations

import argparse
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from src.data import DATASETS, ROOT, SPLITS, load_all

RESULTS = ROOT / "results"


def main(dataset: str) -> None:
    RESULTS.mkdir(exist_ok=True)
    data = load_all(dataset)
    out: dict = {"dataset": dataset}

    out["split_sizes"] = {s: len(df) for s, df in data.items()}
    labels = sorted(set().union(*[set(df.label) for df in data.values()]))
    out["n_classes_observed"] = len(labels)
    out["labels_observed"] = [int(x) for x in labels]

    dist = pd.DataFrame({s: data[s].label.value_counts().reindex(labels, fill_value=0) for s in SPLITS})
    out["label_distribution"] = {s: {int(k): int(v) for k, v in dist[s].items()} for s in SPLITS}
    out["majority_class_share_test"] = float(dist["test"].max() / dist["test"].sum())

    stats = {}
    for s, df in data.items():
        words = df.text.str.split().map(len)
        chars = df.text.str.len()
        stats[s] = {
            "words_mean": float(words.mean()), "words_median": float(words.median()),
            "words_p95": float(words.quantile(0.95)), "words_max": int(words.max()),
            "words_min": int(words.min()), "chars_mean": float(chars.mean()),
        }
        df["n_words"] = words
    out["length_stats"] = stats
    out["empty_texts"] = {s: int((df.text.str.len() == 0).sum()) for s, df in data.items()}
    out["train_dev_test_overlap"] = {
        "train&test": len(set(data["train"].text) & set(data["test"].text)),
        "train&validation": len(set(data["train"].text) & set(data["validation"].text)),
    }

    (RESULTS / f"eda_{dataset}.json").write_text(json.dumps(out, indent=2))

    ax = dist.plot.bar(figsize=(10, 4), title=f"{dataset}: label distribution")
    ax.set_xlabel("label id"); ax.set_ylabel("documents")
    plt.tight_layout(); plt.savefig(RESULTS / f"eda_{dataset}_labels.png", dpi=120); plt.close()

    plt.figure(figsize=(8, 4))
    data["train"].n_words.clip(upper=data["train"].n_words.quantile(0.99)).hist(bins=50)
    plt.title(f"{dataset}: words per document (train, clipped at p99)")
    plt.xlabel("words"); plt.tight_layout()
    plt.savefig(RESULTS / f"eda_{dataset}_lengths.png", dpi=120); plt.close()

    print(json.dumps({k: v for k, v in out.items() if k != "labels_observed"}, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=DATASETS, default="scotus")
    main(ap.parse_args().dataset)
