# Source Modules

`src/` contains reusable functions; experiment orchestration remains under
`experiments/`.

| Module | Contents |
|---|---|
| `afop/` | NCA ranking, source-validation D-scan, and episodic AFOP evaluation |
| `features/` | 386-D PVDF/SG feature-pool construction and label attachment |
| `preprocessing/` | Periodic segmentation, center estimation, and jitter cropping |
| `protocol/` | Stratified splits and deterministic episode manifests |
| `baselines/` | FEAT, MAML++, CWT-ResNet, Tactile Transformer, and Channel-GAT |
| `evaluation/` | Shared embedding and raw-signal evaluators |
| `ablations/` | Feature frontends, fixed-D scan, and prototype-only evaluation |
| `analysis/` | Linear CKA utilities |
| `data/` | Data validation, normalization, and model-specific adapters |
| `io/` | Run-directory and artifact helpers |

Call `setup_afop_path()` to add the source tree and configurations to the
MATLAB path. Experiment directories are deliberately not added to avoid name
collisions with reusable functions.
