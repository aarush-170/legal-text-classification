"""Gradio demo: paste a Supreme Court opinion -> issue area + driving passages + similar past cases.

Run from the project root:   python -m app.app          (opens http://127.0.0.1:7860)
Needs models/scotus_tfidf_lr.joblib (python -m src.baselines --dataset scotus --model tfidf).
Retrieval uses Legal-BERT embeddings if models/scotus_legalbert_embeddings.npz exists, otherwise TF-IDF.
"""
from __future__ import annotations

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import html
import random

import gradio as gr

from src.data import display_names, load_split
from src.explain import Explainer

DATASET = "scotus"
explainer = Explainer(DATASET)
test = load_split(DATASET, "test")
NAMES = display_names(DATASET)
MAX_CHARS = 400_000  # guard against pasting something enormous


def analyse(text: str):
    text = (text or "").strip()
    if len(text) < 200:
        raise gr.Error("Please paste a longer opinion (at least a few paragraphs).")
    out = explainer.analyse(text[:MAX_CHARS])
    top5 = dict(sorted(out["probabilities"].items(), key=lambda kv: -kv[1])[:5])

    top = max((p["score"] for p in out["top_passages"]), default=1.0) or 1.0
    cards = []
    for p in out["top_passages"]:
        alpha = max(0.15, min(1.0, p["score"] / top)) * 0.6
        cards.append(
            f"<div style='margin:6px 0;padding:8px 10px;border-radius:6px;background:rgba(255,193,7,{alpha:.2f})'>"
            f"<b>sentence #{p['position'] + 1}</b> &middot; contribution {p['score']:+.3f}<br>"
            f"{html.escape(p['sentence'][:600])}</div>")
    terms = ", ".join(html.escape(t) for t, _ in out["top_terms"])
    passages = (f"<p><b>Predicted: {html.escape(out['predicted'])}</b> ({out['confidence']:.1%})</p>"
                f"<p>Words pushing the prediction: <i>{terms}</i></p>" + "".join(cards))

    rows = [[s["rank"], f"{s['similarity']:.3f}", s["true_label"], s["snippet"][:160]] for s in out["similar"]]
    return top5, passages, rows


def load_example():
    i = random.randrange(len(test))
    row = test.iloc[i]
    return row.text, f"Loaded test opinion #{i}. True issue area: **{NAMES[row.label]}**"


with gr.Blocks(title="Legal issue-area classifier") as demo:
    gr.Markdown("# Supreme Court opinion issue-area classifier\n"
                f"Model: TF-IDF + logistic regression (13 SCDB issue areas). "
                f"Similar-case retrieval: **{explainer.retriever.backend}** embeddings over the training set.")
    text = gr.Textbox(label="Opinion text", lines=12, placeholder="Paste a US Supreme Court opinion here...")
    with gr.Row():
        go = gr.Button("Classify", variant="primary")
        ex = gr.Button("Load a random test opinion")
    note = gr.Markdown()
    with gr.Row():
        label = gr.Label(label="Predicted issue area (top 5)", num_top_classes=5)
        passages = gr.HTML(label="Passages that most drove the prediction")
    gr.Markdown("### Most similar training cases (with their true labels)")
    table = gr.Dataframe(headers=["#", "similarity", "true issue area", "opinion start"], interactive=False,
                         wrap=True)
    go.click(analyse, text, [label, passages, table])
    ex.click(load_example, None, [text, note])

if __name__ == "__main__":
    demo.launch()
