# Experiments

Experiments are grouped by scientific question. Each runner is self-contained:
it locates the repository root, adds `src/` and `configs/`, and writes outputs
outside the source tree.

| Folder | Purpose |
|---|---|
| [`preprocessing/`](preprocessing/) | Raw TXT segmentation and target generation |
| [`closed_set/`](closed_set/) | Main AFOP table and baseline training |
| [`generalization/`](generalization/) | Cross-shape, cross-material, and real-contact OOD |
| [`robustness/`](robustness/) | Physical perturbation and phase-shift sweeps |
| [`ablations/`](ablations/) | Feature optimization and wavelet sensitivity |
| [`analysis/`](analysis/) | CKA, t-SNE, bootstrap, permutation, and Holm tests |

Run scripts from the repository root with `run('experiments/.../<script>.m')`.
Detailed ordering and environment variables are listed in
[`docs/REPRODUCIBILITY.md`](../docs/REPRODUCIBILITY.md).
