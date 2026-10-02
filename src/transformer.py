"""Chunked Legal-BERT for long opinions.

Each opinion is tokenised without truncation and split into <=510-token chunks (+[CLS]/[SEP]).
If an opinion has more than --max-chunks chunks, chunks are sampled evenly across the document
(--sampling spread) or the first ones are kept (--sampling first).

Two modes:
  embed     frozen encoder; mean-pool tokens per chunk, mean over chunks -> doc embedding -> LR (dev-tuned C).
            Cheap (inference only); the embeddings are reused for similar-case retrieval.
  finetune  every chunk inherits its document's label; at inference the doc prediction is the
            MEAN OF CHUNK LOGITS. Best epoch is chosen on dev macro-F1.

Usage (GPU strongly recommended; see notebooks/colab_transformer.ipynb for Colab):
    python -m src.transformer --dataset scotus --mode embed
    python -m src.transformer --dataset scotus --mode finetune --epochs 3
    python -m src.transformer --dataset scotus --mode finetune --quick   # tiny smoke test, nothing saved
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from tqdm import tqdm
from transformers import (AutoModel, AutoModelForSequenceClassification, AutoTokenizer,
                          get_linear_schedule_with_warmup)

from src.data import DATASETS, ROOT, SEED, label_names, load_all
from src.evaluate import compute_metrics, save_run

MODEL_NAME = "nlpaueb/legal-bert-base-uncased"
MODELS_DIR = ROOT / "models"


def chunk_documents(texts, tok, max_len: int = 512, max_chunks: int = 16, sampling: str = "spread"):
    """Return (chunks, doc_idx): token-id lists (no special tokens) and the doc index of each."""
    size = max_len - 2  # room for [CLS] and [SEP]
    chunks, doc_idx = [], []
    for d, text in enumerate(tqdm(texts, desc="tokenising", leave=False)):
        ids = tok(text, add_special_tokens=False, truncation=False)["input_ids"]
        parts = [ids[i:i + size] for i in range(0, len(ids), size)] or [[]]
        if len(parts) > max_chunks:
            if sampling == "spread":
                keep = np.linspace(0, len(parts) - 1, max_chunks).round().astype(int)
                parts = [parts[i] for i in keep]
            else:
                parts = parts[:max_chunks]
        chunks += parts
        doc_idx += [d] * len(parts)
    return chunks, np.array(doc_idx)


def _batches(chunks, order, bs, tok, device):
    for i in range(0, len(order), bs):
        idx = order[i:i + bs]
        feats = [tok.build_inputs_with_special_tokens(chunks[j]) for j in idx]
        batch = tok.pad({"input_ids": feats}, return_tensors="pt")
        yield idx, {k: v.to(device) for k, v in batch.items()}


@torch.no_grad()
def encode(chunks, model, tok, device, bs, kind):
    """kind='logits' -> (n_chunks, n_labels); kind='embed' -> (n_chunks, hidden) masked mean pooling."""
    model.eval()
    order = np.argsort([len(c) for c in chunks])  # length-sorted -> less padding
    out = None
    for idx, batch in tqdm(_batches(chunks, order, bs, tok, device), total=-(-len(order) // bs),
                           desc=kind, leave=False):
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            if kind == "logits":
                res = model(**batch).logits
            else:
                h = model(**batch).last_hidden_state
                m = batch["attention_mask"].unsqueeze(-1).to(h.dtype)
                res = (h * m).sum(1) / m.sum(1)
        res = res.float().cpu().numpy()
        if out is None:
            out = np.zeros((len(chunks), res.shape[1]), dtype=np.float32)
        out[idx] = res
    return out


def mean_by_doc(x, doc_idx, n_docs):
    s = np.zeros((n_docs, x.shape[1]), dtype=np.float32)
    np.add.at(s, doc_idx, x)
    return s / np.bincount(doc_idx, minlength=n_docs)[:, None]


def run_embed(args, data, tok, device):
    model = AutoModel.from_pretrained(MODEL_NAME).to(device)
    feats = {}
    for split, df in data.items():
        t0 = time.time()
        ch, di = chunk_documents(df.text.tolist(), tok, args.max_len, args.max_chunks, args.sampling)
        feats[split] = mean_by_doc(encode(ch, model, tok, device, args.batch_size, "embed"), di, len(df))
        print(f"[embed] {split}: {len(df)} docs, {len(ch)} chunks, {time.time()-t0:.0f}s")
    best = {"macro_f1": -1.0}
    for C in ([10] if args.quick else [1, 10, 100]):
        clf = LogisticRegression(C=C, class_weight="balanced", max_iter=3000, random_state=SEED)
        clf.fit(feats["train"], data["train"].label)
        m = compute_metrics(data["validation"].label, clf.predict(feats["validation"]))
        print(f"  C={C:<4} dev macro-F1={m['macro_f1']:.4f} acc={m['accuracy']:.4f}")
        if m["macro_f1"] > best["macro_f1"]:
            best = {**m, "C": C, "clf": clf}
    pred = best["clf"].predict(feats["test"])
    test_m = compute_metrics(data["test"].label, pred)
    print(f"[embed] TEST: {test_m}")
    if not args.quick:
        MODELS_DIR.mkdir(exist_ok=True)
        np.savez_compressed(MODELS_DIR / f"{args.dataset}_legalbert_embeddings.npz", **feats)
        dev_m = {k: best[k] for k in ("accuracy", "macro_f1", "micro_f1")}
        save_run(args.dataset, "legalbert_embed_lr", dev_m, test_m, data["test"].label.values, pred,
                 extra={"best_params": {"C": best["C"]}, "max_chunks": args.max_chunks, "seed": SEED})


def run_finetune(args, data, tok, device):
    n_labels = len(label_names(args.dataset))
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=n_labels).to(device)
    ch = {}
    for split, df in data.items():
        ch[split] = chunk_documents(df.text.tolist(), tok, args.max_len, args.max_chunks, args.sampling)
        print(f"[finetune] {split}: {len(df)} docs -> {len(ch[split][0])} chunks")
    tr_chunks, tr_doc = ch["train"]
    tr_y = torch.tensor(data["train"].label.values[tr_doc])
    n_steps_ep = -(-len(tr_chunks) // args.batch_size)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    sched = get_linear_schedule_with_warmup(opt, int(0.06 * n_steps_ep * args.epochs), n_steps_ep * args.epochs)
    scaler = torch.amp.GradScaler(enabled=device.type == "cuda")
    counts = np.bincount(data["train"].label, minlength=n_labels).astype(np.float32)
    w = torch.tensor((counts.sum() / (n_labels * np.maximum(counts, 1))) ** 0.5, device=device)  # sqrt-balanced
    loss_fn = torch.nn.CrossEntropyLoss(weight=w)
    rng = np.random.default_rng(SEED)

    def doc_logits(split):
        c, d = ch[split]
        return mean_by_doc(encode(c, model, tok, device, args.batch_size * 2, "logits"), d, len(data[split]))

    best_f1, best_state, best_dev = -1.0, None, None
    for ep in range(args.epochs):
        model.train()
        order = rng.permutation(len(tr_chunks))
        t0, total = time.time(), 0.0
        for idx, batch in tqdm(_batches(tr_chunks, order, args.batch_size, tok, device),
                               total=n_steps_ep, desc=f"epoch {ep+1}"):
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                logits = model(**batch).logits
            loss = loss_fn(logits.float(), tr_y[idx].to(device))
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            total += loss.item()
        dev_m = compute_metrics(data["validation"].label, doc_logits("validation").argmax(1))
        print(f"[finetune] epoch {ep+1}: train loss/step={total / n_steps_ep:.4f} "
              f"dev macro-F1={dev_m['macro_f1']:.4f} acc={dev_m['accuracy']:.4f} ({time.time()-t0:.0f}s)")
        if dev_m["macro_f1"] > best_f1:
            best_f1, best_dev = dev_m["macro_f1"], dev_m
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    pred = doc_logits("test").argmax(1)
    test_m = compute_metrics(data["test"].label, pred)
    print(f"[finetune] TEST: {test_m}")
    if not args.quick:
        MODELS_DIR.mkdir(exist_ok=True)
        torch.save(best_state, MODELS_DIR / f"{args.dataset}_legalbert_finetuned.pt")
        save_run(args.dataset, "legalbert_finetune", best_dev, test_m, data["test"].label.values, pred,
                 extra={"best_params": {"lr": args.lr, "epochs": args.epochs, "batch_size": args.batch_size,
                                        "max_chunks": args.max_chunks, "sampling": args.sampling}, "seed": SEED})


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=DATASETS, default="scotus")
    ap.add_argument("--mode", choices=["embed", "finetune"], required=True)
    ap.add_argument("--max-len", type=int, default=512)
    ap.add_argument("--max-chunks", type=int, default=16)
    ap.add_argument("--sampling", choices=["spread", "first"], default="spread")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--device", default="auto", help="auto | cuda | cpu")
    ap.add_argument("--quick", action="store_true", help="tiny smoke test; saves nothing")
    args = ap.parse_args()

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"device: {device}")
    data = load_all(args.dataset)
    if args.quick:
        args.max_chunks, args.epochs = 2, 1
        data = {"train": data["train"].groupby("label").head(3).reset_index(drop=True),
                "validation": data["validation"].head(20), "test": data["test"].head(20)}
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    tok.model_max_length = int(1e9)  # we chunk ourselves; silence the length warning
    (run_embed if args.mode == "embed" else run_finetune)(args, data, tok, device)
