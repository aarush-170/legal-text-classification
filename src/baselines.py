"""Classical baselines. Hyper-parameters are tuned on dev only; test is scored once.

Usage:
    python -m src.baselines --dataset scotus --model tfidf
(LDA + Doc2Vec are added in the next step.)
"""
from __future__ import annotations

import argparse
import itertools
import time

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from src.data import DATASETS, ROOT, SEED, load_all
from src.evaluate import compute_metrics, save_run

MODELS_DIR = ROOT / "models"


def run_tfidf(dataset: str, quick: bool = False) -> None:
    data = load_all(dataset)
    tr, dv, te = data["train"], data["validation"], data["test"]
    if quick:  # tiny smoke test
        tr, dv, te = tr.head(300), dv.head(100), te.head(100)

    ngrams = [(1, 1)] if quick else [(1, 1), (1, 2)]
    Cs = [10] if quick else [1, 10, 100]
    class_weights = [None] if quick else [None, "balanced"]

    best = {"macro_f1": -1.0}
    for ngram in ngrams:
        t0 = time.time()
        vec = TfidfVectorizer(ngram_range=ngram, sublinear_tf=True, min_df=3, max_df=0.9,
                              max_features=300_000, lowercase=True, dtype=np.float32)
        Xtr, Xdv = vec.fit_transform(tr.text), vec.transform(dv.text)
        print(f"[tfidf] ngram={ngram} features={Xtr.shape[1]} vectorised in {time.time()-t0:.0f}s")
        for C, cw in itertools.product(Cs, class_weights):
            clf = LogisticRegression(C=C, class_weight=cw, max_iter=2000, random_state=SEED)
            clf.fit(Xtr, tr.label)
            m = compute_metrics(dv.label, clf.predict(Xdv))
            print(f"  C={C:<4} class_weight={cw!s:<9} dev macro-F1={m['macro_f1']:.4f} acc={m['accuracy']:.4f}")
            if m["macro_f1"] > best["macro_f1"]:
                best = {**m, "ngram": ngram, "C": C, "class_weight": cw, "vec": vec, "clf": clf}

    vec, clf = best["vec"], best["clf"]
    pred = clf.predict(vec.transform(te.text))
    test_m = compute_metrics(te.label, pred)
    params = {k: (list(v) if isinstance(v, tuple) else v) for k, v in best.items()
              if k in ("ngram", "C", "class_weight")}
    print(f"[tfidf] best on dev: {params}")
    print(f"[tfidf] TEST: {test_m}")
    if not quick:
        MODELS_DIR.mkdir(exist_ok=True)
        joblib.dump({"vec": vec, "clf": clf}, MODELS_DIR / f"{dataset}_tfidf_lr.joblib")
        dev_m = {k: best[k] for k in ("accuracy", "macro_f1", "micro_f1")}
        save_run(dataset, "tfidf_lr", dev_m, test_m, te.label.values, pred,
                 extra={"best_params": params, "seed": SEED})


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=DATASETS, default="scotus")
    ap.add_argument("--model", choices=["tfidf"], default="tfidf")
    ap.add_argument("--quick", action="store_true", help="tiny CPU smoke test; saves nothing")
    a = ap.parse_args()
    run_tfidf(a.dataset, a.quick)
