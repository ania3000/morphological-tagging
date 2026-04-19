# The code used in thesis (2026) on morphological tagging for Ossetic
* BERT-based:
  * Joint tags prediction (POS+feats) -- .py, with old compute_metrics
  * Multi-task prediction (a separate classifier for each feature) -- .ipynb
* FST-based:
  * foma analyzer w/o dictionary w/closed classes list
  * foma analyzer w/dictionary
