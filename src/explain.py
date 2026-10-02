"""Explanations + similar-case retrieval for the demo.

Prediction/explanation use the TF-IDF + LR model (models/<dataset>_tfidf_lr.joblib). Because the model is
linear, the logit of class c is  sum_t tfidf(t) * coef[c, t] + bias[c], so every token occurrence has an
exact additive contribution and a sentence's score is the sum over its tokens. The sentences with the highest
score for the predicted class are the passages that most drove the prediction.

Retrieval returns the top-k most similar TRAIN opinions by cosine similarity, with their true labels:
  - backend "legalbert": mean-of-chunks Legal-BERT embeddings (needs models/<dataset>_legalbert_embeddings.npz,
    produced by `python -m src.transformer --mode embed`)
  - backend "tfidf": TF-IDF cosine (always available; used as fallback)

CLI:  python -m src.explain --dataset scotus --test-index 0
"""
from __future__ import annotations

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")  # Anaconda + pip torch OpenMP clash on Windows

import argparse
import re
from collections import Counter

import joblib
import numpy as np
import scipy.sparse as sp

from src.data import DATASETS, ROOT, clean_text, display_names, load_split

MODELS_DIR = ROOT / "models"
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"(\[])")


def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENT_SPLIT.split(text) if s.strip()]


class Retriever:
    """Top-k nearest training opinions by cosine similarity."""

    def __init__(self, dataset: str, vec, backend: str = "auto"):
        self.dataset, self.vec = dataset, vec
        emb_path = MODELS_DIR / f"{dataset}_legalbert_embeddings.npz"
        if backend == "auto":
            backend = "legalbert" if emb_path.exists() else "tfidf"
        if backend == "legalbert" and not emb_path.exists():
            raise FileNotFoundError(f"{emb_path} missing; run `python -m src.transformer --mode embed` first")
        self.backend = backend
        train = load_split(dataset, "train")
        self.texts, self.labels = train.text.tolist(), train.label.values
        if backend == "tfidf":
            cache = MODELS_DIR / f"{dataset}_tfidf_train.npz"
            if cache.exists():
                self.matrix = sp.load_npz(cache)
            else:
                self.matrix = vec.transform(self.texts).tocsr()
                sp.save_npz(cache, self.matrix)
        else:
            e = np.load(emb_path)["train"]
            self.matrix = e / np.maximum(np.linalg.norm(e, axis=1, keepdims=True), 1e-9)
            self._tok = self._model = self._device = None

    def _embed_query(self, text: str) -> np.ndarray:
        import torch
        from transformers import AutoModel, AutoTokenizer

        from src.transformer import MODEL_NAME, chunk_documents, encode, mean_by_doc

        if self._model is None:
            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self._tok = AutoTokenizer.from_pretrained(MODEL_NAME)
            self._tok.model_max_length = int(1e9)
            self._model = AutoModel.from_pretrained(MODEL_NAME).to(self._device)
        ch, di = chunk_documents([text], self._tok)  # same chunking defaults as the stored train embeddings
        q = mean_by_doc(encode(ch, self._model, self._tok, self._device, 8, "embed"), di, 1)[0]
        return q / max(np.linalg.norm(q), 1e-9)

    def top_k(self, text: str, k: int = 5) -> list[dict]:
        if self.backend == "tfidf":
            sims = (self.matrix @ self.vec.transform([text]).T).toarray().ravel()  # rows are L2-normalised
        else:
            sims = self.matrix @ self._embed_query(text)
        names = display_names(self.dataset)
        return [{"rank": r + 1, "train_index": int(i), "similarity": float(sims[i]),
                 "true_label": names[self.labels[i]], "snippet": self.texts[i][:300]}
                for r, i in enumerate(np.argsort(-sims)[:k])]


class Explainer:
    def __init__(self, dataset: str = "scotus", retrieval: str = "auto"):
        bundle = joblib.load(MODELS_DIR / f"{dataset}_tfidf_lr.joblib")  # {"vec", "clf"}
        self.vec, self.clf, self.dataset = bundle["vec"], bundle["clf"], dataset
        self.names = display_names(dataset)
        self.retriever = Retriever(dataset, self.vec, retrieval)

    def predict(self, text: str) -> dict[str, float]:
        probs = self.clf.predict_proba(self.vec.transform([text]))[0]
        return {self.names[c]: float(p) for c, p in zip(self.clf.classes_, probs)}

    def explain(self, text: str, n_passages: int = 5, n_terms: int = 10) -> dict:
        """Exact additive attribution of the predicted class's logit to sentences and terms."""
        X = self.vec.transform([text])
        probs = self.clf.predict_proba(X)[0]
        pred_pos = int(np.argmax(probs))
        coef = self.clf.coef_[pred_pos]
        row = X.tocoo()
        weight = {int(j): float(v) for j, v in zip(row.col, row.data)}  # doc-level tfidf
        analyzer, vocab = self.vec.build_analyzer(), self.vec.vocabulary_
        sentences = split_sentences(text)
        toks = [analyzer(s) for s in sentences]
        counts = Counter(t for ts in toks for t in ts if t in vocab)
        sent_scores, term_scores = [], Counter()
        for ts in toks:
            score = 0.0
            for t in ts:
                j = vocab.get(t)
                if j is None or j not in weight:
                    continue
                c = (weight[j] / counts[t]) * coef[j]  # per-occurrence share of tfidf(t)*coef
                score += c
                term_scores[t] += c
            sent_scores.append(score)
        order = [i for i in np.argsort(sent_scores)[::-1] if len(sentences[i]) >= 40][:n_passages]
        return {
            "predicted": self.names[int(self.clf.classes_[pred_pos])],
            "confidence": float(probs[pred_pos]),
            "sentences": sentences,
            "sentence_scores": sent_scores,
            "top_passages": [{"sentence": sentences[i], "score": float(sent_scores[i]), "position": int(i)}
                             for i in order],
            "top_terms": [(t, float(s)) for t, s in term_scores.most_common(n_terms)],
        }

    def similar_cases(self, text: str, k: int = 5) -> list[dict]:
        return self.retriever.top_k(text, k)

    def analyse(self, text: str) -> dict:
        text = clean_text(text)
        return {"probabilities": self.predict(text), **self.explain(text), "similar": self.similar_cases(text)}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=DATASETS, default="scotus")
    ap.add_argument("--test-index", type=int, default=0)
    ap.add_argument("--retrieval", choices=["auto", "tfidf", "legalbert"], default="auto")
    a = ap.parse_args()
    ex = Explainer(a.dataset, a.retrieval)
    row = load_split(a.dataset, "test").iloc[a.test_index]
    out = ex.analyse(row.text)
    print(f"true: {ex.names[row.label]} | predicted: {out['predicted']} ({out['confidence']:.1%}) "
          f"| retrieval: {ex.retriever.backend}")
    print("top terms:", [t for t, _ in out["top_terms"]])
    for p in out["top_passages"][:3]:
        print(f"  [{p['score']:+.3f}] {p['sentence'][:200]}")
    for s in out["similar"]:
        print(f"  #{s['rank']} sim={s['similarity']:.3f} true={s['true_label']} | {s['snippet'][:80]}")
