# Robustness

- `run_force_speed.m` evaluates the fixed force/speed/noise target domain.
- `run_phase_sweep.m` evaluates one value of `AFOP_SHIFT_SIGMA` while retaining
  paired support/query identities across severity levels.

Generate targets with the scripts in `experiments/preprocessing/` before
running model stages selected by `AFOP_STAGE`.
