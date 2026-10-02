# Published LexGLUE SCOTUS baselines

Source: Chalkidis, I. et al. (2022). *LexGLUE: A Benchmark Dataset for Legal Language Understanding in English.*
ACL 2022. arXiv:2110.00976, **Table 3** (test micro-F1 / macro-F1, in percent; SCOTUS is a starred dataset, so
the paper uses the hierarchical variant of each model except Longformer and BigBird). Values in
`lexglue_published_scotus.csv` were copied from that table.

Our numbers (test, 0-1 scale) are in `scotus_comparison.csv`. Differences to keep in mind: the paper's hierarchical
models and its TFIDF-SVM may use different preprocessing, hyper-parameters and seeds (the paper reports means over
several seeds for neural models), so the comparison is indicative, not like-for-like.
