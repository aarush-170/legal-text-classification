# Work split (2 people, separable modules)

Each person works on their own branch (`feat/<name>-<topic>`), commits under their own git identity, and opens a PR to `main`.
Shared files (`src/data.py`, `src/evaluate.py`) are stable; change them only via a small PR agreed by both.
All models write results through `evaluate.save_run(...)` so the comparison table fills itself in.

## Person A (repo owner)
| Module | Task |
|---|---|
| `src/data.py`, `src/eda.py`, `src/evaluate.py` | done: loading, EDA, metrics |
| `src/baselines.py` | TF-IDF + LR (done, tuning on dev) |
| `src/transformer.py` | chunked Legal-BERT (~512-token chunks, mean logits), GPU / Colab flag |
| `notebooks/colab_transformer.ipynb` | Colab runner for the transformer |

## Person B (teammate, @aarohijaiswal)
| Module | Task |
|---|---|
| `src/topic_baselines.py` | LDA + LR and Doc2Vec (gensim) + LR, reproducing the CS229 paper; tune on dev only; seeds fixed; `python -m src.topic_baselines --dataset scotus --model doc2vec` |
| `src/explain.py` | top-driving sentences from TF-IDF LR coefficients; top-5 similar training cases by embedding cosine similarity |
| `app/app.py` | Gradio demo: prediction + confidence, highlighted passages, similar cases |
| `notebooks/eda.ipynb` | EDA notebook using `src/eda.py` outputs and plots |

## Shared / at the end
README results section, LexGLUE paper comparison (cite the exact table), final report, optional LEDGAR run.

## Interfaces (so work doesn't block)
- `from src.data import load_all, label_names` -> dict of DataFrames with `text`, `label`, `label_name`.
- TF-IDF model is saved at `models/scotus_tfidf_lr.joblib` as `{"vec": TfidfVectorizer, "clf": LogisticRegression}` (A produces it; B's `explain.py` consumes it).
- Embeddings for retrieval: B picks (Doc2Vec vectors or TF-IDF SVD) and documents the choice; A's Legal-BERT embeddings can be swapped in later.
