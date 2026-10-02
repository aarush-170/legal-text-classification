"""LDA + LR and Doc2Vec + LR baselines (reproducing the CS229 paper). Tuned on dev only.

Usage:
    python -m src.topic_baselines --dataset scotus --model lda
    python -m src.topic_baselines --dataset scotus --model doc2vec
Add --quick for a tiny CPU smoke test (nothing saved).
"""
from __future__ import annotations

import argparse
import itertools
import time

import joblib
import numpy as np
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from gensim.utils import simple_preprocess
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression

from src.data import DATASETS, ROOT, SEED, load_all
from src.evaluate import compute_metrics, save_run

MODELS_DIR = ROOT / "models"
CS = [1, 10, 100]
DOC2VEC_DIMS = [100, 200]
DOC2VEC_DMS = [0, 1]

def _fit_lr_grid(Xtr, ytr, Xdv, ydv, Cs):
    best = {"macro_f1": -1.0}
    for C in Cs:
        clf = LogisticRegression(C=C, max_iter=2000, random_state=SEED).fit(Xtr, ytr)
        m = compute_metrics(ydv, clf.predict(Xdv))
        print(f"    C={C:<4} dev macro-F1={m['macro_f1']:.4f} acc={m['accuracy']:.4f}")
        if m["macro_f1"] > best["macro_f1"]:
            best = {**m, "C": C, "clf": clf}
    return best


def _splits(dataset, quick):
    d = load_all(dataset)
    tr, dv, te = d["train"], d["validation"], d["test"]
    return (tr.head(300), dv.head(100), te.head(100)) if quick else (tr, dv, te)


def run_lda(dataset: str, quick: bool = False, n_jobs: int = -1) -> None:
    tr, dv, te = _splits(dataset, quick)
    vec = CountVectorizer(stop_words="english", min_df=5, max_df=0.5, max_features=20_000)
    Ctr, Cdv, Cte = vec.fit_transform(tr.text), vec.transform(dv.text), vec.transform(te.text)
    best = {"macro_f1": -1.0}
    for k in ([10] if quick else [25, 50, 100]):
        t0 = time.time()
        lda = LatentDirichletAllocation(n_components=k, learning_method="online", random_state=SEED,
                                        n_jobs=n_jobs, max_iter=10 if not quick else 3)
        Ttr, Tdv = lda.fit_transform(Ctr), lda.transform(Cdv)
        print(f"[lda] topics={k} fitted in {time.time()-t0:.0f}s")
        r = _fit_lr_grid(Ttr, tr.label, Tdv, dv.label, [10] if quick else CS)
        if r["macro_f1"] > best["macro_f1"]:
            best = {**r, "k": k, "lda": lda}
    pred = best["clf"].predict(best["lda"].transform(Cte))
    test_m = compute_metrics(te.label, pred)
    print(f"[lda] best on dev: topics={best['k']} C={best['C']}\n[lda] TEST: {test_m}")
    if not quick:
        MODELS_DIR.mkdir(exist_ok=True)
        joblib.dump({"vec": vec, "lda": best["lda"], "clf": best["clf"]}, MODELS_DIR / f"{dataset}_lda_lr.joblib")
        dev_m = {k: best[k] for k in ("accuracy", "macro_f1", "micro_f1")}
        save_run(dataset, "lda_lr", dev_m, test_m, te.label.values, pred,
                 extra={"best_params": {"topics": best["k"], "C": best["C"]}, "seed": SEED})


def _tok(texts):
    return [simple_preprocess(t, min_len=2, max_len=30) for t in texts]


def run_doc2vec(dataset: str, quick: bool = False, workers: int = 4, dim: int = 100, epochs: int = 20) -> None:
    tr, dv, te = _splits(dataset, quick)
    if quick:
        epochs = 3

    ttr, tdv, tte = _tok(tr.text), _tok(dv.text), _tok(te.text)

    best_overall = {"macro_f1": -1.0}

    dims = [dim] if quick else DOC2VEC_DIMS
    dms = [1] if quick else DOC2VEC_DMS

    for current_dim in dims:
        for current_dm in dms:
            print(f"[doc2vec] trying dim={current_dim}, dm={current_dm}")

            t0 = time.time()
            model = Doc2Vec(
                vector_size=current_dim,
                min_count=5,
                epochs=epochs,
                workers=workers,
                seed=SEED,
                dm=current_dm,
            )

            tagged_train = [
                TaggedDocument(w, [i])
                for i, w in enumerate(ttr)
            ]

            model.build_vocab(tagged_train)
            model.train(
                tagged_train,
                total_examples=model.corpus_count,
                epochs=model.epochs,
            )

            print(
                f"[doc2vec] trained dim={current_dim} "
                f"dm={current_dm} epochs={epochs} "
                f"in {time.time() - t0:.0f}s"
            )

            Xtr = np.vstack([
                model.dv[i] for i in range(len(ttr))
            ])

            t0 = time.time()

            Xdv = np.vstack([
                model.infer_vector(w) for w in tdv
            ])

            Xte = np.vstack([
                model.infer_vector(w) for w in tte
            ])

            print(
                f"[doc2vec] inferred dev/test vectors "
                f"in {time.time() - t0:.0f}s"
            )

            lr_best = _fit_lr_grid(
                Xtr,
                tr.label,
                Xdv,
                dv.label,
                [10] if quick else CS,
            )

            print(
                f"[doc2vec] dim={current_dim} "
                f"dm={current_dm} "
                f"best C={lr_best['C']} "
                f"dev macro-F1={lr_best['macro_f1']:.4f}"
            )

            if lr_best["macro_f1"] > best_overall["macro_f1"]:
                best_overall = {
                    **lr_best,
                    "model": model,
                    "dim": current_dim,
                    "dm": current_dm,
                }

    model = best_overall["model"]

    Xte = np.vstack([
        model.infer_vector(w) for w in tte
    ])

    pred = best_overall["clf"].predict(Xte)
    test_m = compute_metrics(te.label, pred)

    print(
        f"[doc2vec] best on dev: "
        f"dim={best_overall['dim']} "
        f"dm={best_overall['dm']} "
        f"C={best_overall['C']}"
    )
    print(f"[doc2vec] TEST: {test_m}")

    if not quick:
        MODELS_DIR.mkdir(exist_ok=True)

        model.save(
            str(MODELS_DIR / f"{dataset}_doc2vec.model")
        )

        joblib.dump(
            best_overall["clf"],
            MODELS_DIR / f"{dataset}_doc2vec_lr.joblib"
        )

        dev_m = {
            k: best_overall[k]
            for k in ("accuracy", "macro_f1", "micro_f1")
        }

        save_run(
            dataset,
            "doc2vec_lr",
            dev_m,
            test_m,
            te.label.values,
            pred,
            extra={
                "best_params": {
                    "C": best_overall["C"],
                    "dim": best_overall["dim"],
                    "dm": best_overall["dm"],
                    "epochs": epochs,
                },
                "seed": SEED,
                "workers": workers,
            },
        )

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=DATASETS, default="scotus")
    ap.add_argument("--model", choices=["lda", "doc2vec"], required=True)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--workers", type=int, default=4, help="doc2vec threads (workers=1 for exact reproducibility)")
    a = ap.parse_args()
    run_lda(a.dataset, a.quick) if a.model == "lda" else run_doc2vec(a.dataset, a.quick, a.workers)
