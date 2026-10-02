"""Shared evaluation: accuracy, macro-F1, micro-F1, per-class report, confusion matrix.

Note: for single-label multi-class classification, micro-F1 == accuracy.
We report both because LexGLUE reports micro-F1 / macro-F1.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix, f1_score)

from src.data import ROOT

RESULTS = ROOT / "results"


def compute_metrics(y_true, y_pred) -> dict:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "micro_f1": float(f1_score(y_true, y_pred, average="micro", zero_division=0)),
    }


def per_class_report(y_true, y_pred) -> dict:
    return classification_report(y_true, y_pred, output_dict=True, zero_division=0)


def plot_confusion_matrix(y_true, y_pred, path: Path, title: str = "") -> None:
    labels = sorted(set(y_true) | set(y_pred))
    cm = confusion_matrix(y_true, y_pred, labels=labels, normalize="true")
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(labels)), labels); ax.set_yticks(range(len(labels)), labels)
    ax.set_xlabel("predicted"); ax.set_ylabel("true"); ax.set_title(title)
    for i in range(len(labels)):
        for j in range(len(labels)):
            if cm[i, j] >= 0.05:
                ax.text(j, i, f"{cm[i, j]:.2f}", ha="center", va="center",
                        color="white" if cm[i, j] > 0.5 else "black", fontsize=7)
    fig.colorbar(im); fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)


def save_run(dataset: str, model: str, dev_metrics: dict, test_metrics: dict,
             y_test, pred_test, extra: dict | None = None) -> None:
    """Write results/<dataset>_<model>.json, a confusion-matrix PNG, and refresh the comparison table."""
    RESULTS.mkdir(exist_ok=True)
    payload = {"dataset": dataset, "model": model, "dev": dev_metrics, "test": test_metrics,
               "per_class_test": per_class_report(y_test, pred_test), **(extra or {})}
    (RESULTS / f"{dataset}_{model}.json").write_text(json.dumps(payload, indent=2))
    plot_confusion_matrix(np.asarray(y_test), np.asarray(pred_test),
                          RESULTS / f"{dataset}_{model}_confusion.png", f"{dataset} / {model} (test)")
    build_comparison_table(dataset)


def build_comparison_table(dataset: str) -> None:
    """results/<dataset>_comparison.csv: one row per model, test metrics."""
    import pandas as pd

    rows = []
    for f in sorted(RESULTS.glob(f"{dataset}_*.json")):
        d = json.loads(f.read_text())
        if "test" in d and "model" in d:
            rows.append({"model": d["model"], **{f"test_{k}": round(v, 4) for k, v in d["test"].items()},
                         **{f"dev_{k}": round(v, 4) for k, v in d["dev"].items()}})
    if rows:
        df = pd.DataFrame(rows)
        df.to_csv(RESULTS / f"{dataset}_comparison.csv", index=False)
        print(df.to_string(index=False))
