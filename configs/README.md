# Configurations

- `afop_default_config.m` defines the compact AFOP pipeline, including the
  split seeds, center-jitter protocol, D-scan, episodic head, and evaluation
  combinations.
- `benchmark_config.m` defines shared paths and hyperparameters for the full
  benchmark and raw-signal baselines.

Experiment runners read temporary overrides from environment variables. The
most common are `AFOP_MODE`, `AFOP_STAGE`, `AFOP_SPLIT`, and
`AFOP_COMBO_INDEX`; each experiment folder documents its own entry points.
