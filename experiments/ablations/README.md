# Ablations

- `run_feature_optimization.m` compares NCA, PCA, ReliefF, and mRMR; fixed and
  adaptive dimensions; and full-pool, Top-D prototype, and AFOP variants.
- `run_wavelet_sensitivity.m` rebuilds the complete feature pool for each
  wavelet kernel or decomposition level before NCA and D-scan.

Both studies use source-validation episodes for model selection and the shared
closed-set test manifest for reporting.
