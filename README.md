# Legal Text Classification (LexGLUE SCOTUS)

Classify US Supreme Court opinions by issue area, following the Stanford CS229 report
"Classification of Legal Text" (Iyer, 2020) with stronger baselines, a chunked Legal-BERT,
and a demo app that explains predictions and retrieves similar past cases.

## Setup
```bash
python -m venv .venv && source .venv/Scripts/activate   # Windows Git Bash; use .venv\Scripts\Activate.ps1 in PowerShell
pip install -r requirements.txt
```

## Run (one command per step)
```bash
python -m src.data --dataset scotus          # download parquet files to data/raw/ (~173 MB)
python -m src.eda --dataset scotus           # split sizes, label dist, length stats -> results/
python -m src.baselines --dataset scotus --model tfidf
```
Results land in `results/` (metrics JSON, comparison CSV, confusion matrices).

## Layout
`src/data.py` loading/cleaning · `src/baselines.py` TF-IDF/LDA/Doc2Vec + LR · `src/transformer.py` chunked Legal-BERT ·
`src/evaluate.py` metrics · `src/explain.py` explanations + retrieval · `app/` demo · `notebooks/` EDA · `results/` outputs

## Status
See git log. Data: LexGLUE (Chalkidis et al., ACL 2022), parquet from `lighteval/lexglue`.
