# AFOP Core

- `fit_nca_and_dscan.m` fits NCA on `source_train` and selects Top-D using
  5-way 5-shot source-validation episodes.
- `evaluate_afop.m` initializes class prototypes and adapts the cosine-softmax
  head on each support set.

The minimal end-to-end entry point is `run_afop_closedset.m` in the repository
root.
