# Analysis

- `prepare_representations.m` prepares AFOP, parent-pool, and modality-specific
  embeddings.
- `run_representation_analysis.m` computes pairwise CKA, CKA-to-reference, and
  t-SNE coordinates.
- `compute_paired_statistics.m` performs paired bootstrap confidence intervals,
  sign-flip permutation tests, and within-task Holm correction.

Inputs are supplied through `AFOP_REPRESENTATION_INPUT` or
`AFOP_STATISTICS_INPUT`.
