"""Explanations + similar-case retrieval for the demo.

Prediction/explanation use the TF-IDF + LR model
(models/<dataset>_tfidf_lr.joblib).

Because the model is linear, the logit of class c is:
sum_t tfidf(t) * coef[c, t] + bias[c]

Therefore, every token occurrence has an exact additive contribution
and a sentence's score is the sum over its tokens.

Retrieval returns the top-k most similar TRAIN opinions by cosine
similarity, with their true labels:
  - backend "legalbert": mean-of-chunks Legal-BERT embeddings
  - backend "tfidf": TF-IDF cosine similarity

CLI:
    python -m src.explain --dataset scotus --test-index 0
"""

from __future__ import annotations

import os

os.environ.setdefault(
    "KMP_DUPLICATE_LIB_OK",
    "TRUE",
)

import argparse
import re
from collections import Counter

import joblib
import numpy as np
import scipy.sparse as sp

from src.data import (
    DATASETS,
    ROOT,
    clean_text,
    display_names,
    load_split,
)


MODELS_DIR = ROOT / "models"


# Sentence splitter that avoids splitting after common legal abbreviations.
_SENT_SPLIT = re.compile(
    r"(?<=[.!?])\s+(?=[A-Z0-9])"
)

_LEGAL_ABBREVIATIONS = {
    "U.S.",
    "S. Ct.",
    "v.",
    "No.",
    "Nos.",
    "Mr.",
    "Mrs.",
    "Ms.",
    "Dr.",
    "Inc.",
    "Ltd.",
    "Corp.",
    "Jan.",
    "Feb.",
    "Mar.",
    "Apr.",
    "Jun.",
    "Jul.",
    "Aug.",
    "Sep.",
    "Sept.",
    "Oct.",
    "Nov.",
    "Dec.",
}


def split_sentences(text: str) -> list[str]:
    """Split text into sentences while preserving common legal abbreviations."""

    if not text:
        return []

    # Temporarily protect periods in known legal abbreviations.
    protected = text

    replacements = {}

    for i, abbreviation in enumerate(
        sorted(_LEGAL_ABBREVIATIONS, key=len, reverse=True)
    ):
        token = f"__LEGAL_ABBR_{i}__"
        replacements[token] = abbreviation
        protected = protected.replace(abbreviation, token)

    sentences = [
        s.strip()
        for s in _SENT_SPLIT.split(protected)
        if s.strip()
    ]

    # Restore abbreviations.
    restored = []

    for sentence in sentences:
        for token, abbreviation in replacements.items():
            sentence = sentence.replace(token, abbreviation)

        restored.append(sentence)

    return restored


class Retriever:
    """Top-k nearest training opinions by cosine similarity."""

    def __init__(
        self,
        dataset: str,
        vec,
        backend: str = "auto",
    ):
        self.dataset = dataset
        self.vec = vec

        emb_path = (
            MODELS_DIR
            / f"{dataset}_legalbert_embeddings.npz"
        )

        if backend == "auto":
            backend = (
                "legalbert"
                if emb_path.exists()
                else "tfidf"
            )

        if (
            backend == "legalbert"
            and not emb_path.exists()
        ):
            raise FileNotFoundError(
                f"{emb_path} missing; run "
                "`python -m src.transformer --mode embed` first"
            )

        self.backend = backend

        train = load_split(
            dataset,
            "train",
        )

        self.texts = train.text.tolist()
        self.labels = train.label.values

        if backend == "tfidf":
            cache = (
                MODELS_DIR
                / f"{dataset}_tfidf_train.npz"
            )

            if cache.exists():
                self.matrix = sp.load_npz(cache)
            else:
                self.matrix = (
                    vec.transform(self.texts)
                    .tocsr()
                )
                sp.save_npz(
                    cache,
                    self.matrix,
                )

        else:
            e = np.load(emb_path)["train"]

            # Mean-pooled BERT embeddings share a large
            # common component, so centre on the train mean.
            self.mean = e.mean(0)

            e = e - self.mean

            self.matrix = (
                e
                / np.maximum(
                    np.linalg.norm(
                        e,
                        axis=1,
                        keepdims=True,
                    ),
                    1e-9,
                )
            )

            self._tok = None
            self._model = None
            self._device = None

    def _embed_query(
        self,
        text: str,
    ) -> np.ndarray:
        import torch
        from transformers import (
            AutoModel,
            AutoTokenizer,
        )

        from src.transformer import (
            MODEL_NAME,
            chunk_documents,
            encode,
            mean_by_doc,
        )

        if self._model is None:
            self._device = torch.device(
                "cuda"
                if torch.cuda.is_available()
                else "cpu"
            )

            self._tok = AutoTokenizer.from_pretrained(
                MODEL_NAME
            )

            self._tok.model_max_length = int(1e9)

            self._model = AutoModel.from_pretrained(
                MODEL_NAME
            ).to(self._device)

        ch, di = chunk_documents(
            [text],
            self._tok,
        )

        q = (
            mean_by_doc(
                encode(
                    ch,
                    self._model,
                    self._tok,
                    self._device,
                    8,
                    "embed",
                ),
                di,
                1,
            )[0]
            - self.mean
        )

        return q / max(
            np.linalg.norm(q),
            1e-9,
        )

    def top_k(
        self,
        text: str,
        k: int = 5,
    ) -> list[dict]:

        if self.backend == "tfidf":
            sims = (
                self.matrix
                @ self.vec.transform([text]).T
            ).toarray().ravel()

        else:
            sims = (
                self.matrix
                @ self._embed_query(text)
            )

        names = display_names(
            self.dataset
        )

        return [
            {
                "rank": r + 1,
                "train_index": int(i),
                "similarity": float(
                    sims[i]
                ),
                "true_label": names[
                    self.labels[i]
                ],
                "snippet": self.texts[i][:300],
            }
            for r, i in enumerate(
                np.argsort(-sims)[:k]
            )
        ]


def retrieval_precision_at_k(
    dataset: str = "scotus",
    split: str = "validation",
    k: int = 5,
    backend: str = "tfidf",
) -> float:
    """Calculate retrieval precision@k.

    A retrieved training opinion is considered relevant when its
    true label matches the query opinion's true label.

    Precision@k = relevant retrieved cases / k.
    """

    if k <= 0:
        raise ValueError(
            "k must be greater than 0"
        )

    query_split = load_split(
        dataset,
        split,
    )

    model_path = (
        MODELS_DIR
        / f"{dataset}_tfidf_lr.joblib"
    )

    if not model_path.exists():
        raise FileNotFoundError(
            f"{model_path} not found. Create it first with: "
            f"python -m src.baselines --dataset {dataset} --model tfidf"
        )

    bundle = joblib.load(model_path)

    vec = bundle["vec"]

    retriever = Retriever(
        dataset,
        vec,
        backend=backend,
    )

    total_precision = 0.0
    count = 0

    for _, row in query_split.iterrows():
        retrieved = retriever.top_k(
            clean_text(row.text),
            k=k,
        )

        relevant = sum(
            item["true_label"]
            == display_names(dataset)[row.label]
            for item in retrieved
        )

        total_precision += relevant / k
        count += 1

    return (
        total_precision / count
        if count
        else 0.0
    )


class Explainer:
    def __init__(
        self,
        dataset: str = "scotus",
        retrieval: str = "auto",
    ):
        model_path = (
            MODELS_DIR
            / f"{dataset}_tfidf_lr.joblib"
        )

        if not model_path.exists():
            raise FileNotFoundError(
                f"{model_path} not found. Create it first with: "
                f"python -m src.baselines --dataset {dataset} --model tfidf"
            )

        bundle = joblib.load(
            model_path
        )

        self.vec = bundle["vec"]
        self.clf = bundle["clf"]
        self.dataset = dataset

        self.names = display_names(
            dataset
        )

        self.retriever = Retriever(
            dataset,
            self.vec,
            retrieval,
        )

    def predict(
        self,
        text: str,
    ) -> dict[str, float]:

        probs = self.clf.predict_proba(
            self.vec.transform([text])
        )[0]

        return {
            self.names[c]: float(p)
            for c, p in zip(
                self.clf.classes_,
                probs,
            )
        }

    def explain(
        self,
        text: str,
        n_passages: int = 5,
        n_terms: int = 10,
    ) -> dict:
        """Exact additive attribution of the predicted class."""

        X = self.vec.transform([text])

        probs = self.clf.predict_proba(X)[0]

        pred_pos = int(
            np.argmax(probs)
        )

        coef = self.clf.coef_[pred_pos]

        row = X.tocoo()

        weight = {
            int(j): float(v)
            for j, v in zip(
                row.col,
                row.data,
            )
        }

        analyzer = self.vec.build_analyzer()
        vocab = self.vec.vocabulary_

        sentences = split_sentences(
            text
        )

        toks = [
            analyzer(sentence)
            for sentence in sentences
        ]

        counts = Counter(
            token
            for token_list in toks
            for token in token_list
            if token in vocab
        )

        sent_scores = []
        term_scores = Counter()

        for token_list in toks:
            score = 0.0

            for token in token_list:
                j = vocab.get(token)

                if (
                    j is None
                    or j not in weight
                ):
                    continue

                c = (
                    weight[j]
                    / counts[token]
                ) * coef[j]

                score += c
                term_scores[token] += c

            sent_scores.append(score)

        order = [
            i
            for i in np.argsort(
                sent_scores
            )[::-1]
            if len(sentences[i]) >= 40
        ][:n_passages]

        return {
            "predicted": self.names[
                int(
                    self.clf.classes_[
                        pred_pos
                    ]
                )
            ],
            "confidence": float(
                probs[pred_pos]
            ),
            "sentences": sentences,
            "sentence_scores": sent_scores,
            "top_passages": [
                {
                    "sentence": sentences[i],
                    "score": float(
                        sent_scores[i]
                    ),
                    "position": int(i),
                }
                for i in order
            ],
            "top_terms": [
                (token, float(score))
                for token, score in
                term_scores.most_common(
                    n_terms
                )
            ],
        }

    def similar_cases(
        self,
        text: str,
        k: int = 5,
    ) -> list[dict]:
        return self.retriever.top_k(
            text,
            k,
        )

    def analyse(
        self,
        text: str,
    ) -> dict:
        text = clean_text(text)

        return {
            "probabilities": self.predict(text),
            **self.explain(text),
            "similar": self.similar_cases(text),
        }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--dataset",
        choices=DATASETS,
        default="scotus",
    )

    ap.add_argument(
        "--test-index",
        type=int,
        default=0,
    )

    ap.add_argument(
        "--retrieval",
        choices=[
            "auto",
            "tfidf",
            "legalbert",
        ],
        default="auto",
    )

    a = ap.parse_args()

    ex = Explainer(
        a.dataset,
        a.retrieval,
    )

    row = load_split(
        a.dataset,
        "test",
    ).iloc[a.test_index]

    out = ex.analyse(
        row.text
    )

    print(
        f"true: {ex.names[row.label]} "
        f"| predicted: {out['predicted']} "
        f"({out['confidence']:.1%}) "
        f"| retrieval: {ex.retriever.backend}"
    )

    print(
        "top terms:",
        [
            t
            for t, _ in out["top_terms"]
        ],
    )

    for p in out["top_passages"][:3]:
        print(
            f"  [{p['score']:+.3f}] "
            f"{p['sentence'][:200]}"
        )

    for s in out["similar"]:
        print(
            f"  #{s['rank']} "
            f"sim={s['similarity']:.3f} "
            f"true={s['true_label']} "
            f"| {s['snippet'][:80]}"
        )