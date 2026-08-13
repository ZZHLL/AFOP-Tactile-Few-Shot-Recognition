# AFOP: Few-Shot Tactile Recognition

MATLAB implementation of AFOP and the experiments used for closed-set
recognition, cross-domain generalization, perturbation robustness, feature
ablation, and representation analysis.

## Repository layout

```text
.
|-- configs/                 Experiment and AFOP configurations
|-- data/                    Local datasets (not tracked by Git)
|-- docs/                    Full reproduction guide
|-- experiments/
|   |-- preprocessing/       Segmentation and target-domain generation
|   |-- closed_set/          Main benchmark and baseline training
|   |-- generalization/      Cross-shape, cross-material, and real-contact OOD
|   |-- robustness/          Force/speed/noise and phase-shift studies
|   |-- ablations/           Feature optimization and wavelet sensitivity
|   `-- analysis/            CKA, t-SNE, and paired statistics
|-- src/
|   |-- afop/                NCA, D-scan, and episodic AFOP adaptation
|   |-- features/            PVDF/SG feature-pool construction
|   |-- preprocessing/       Signal segmentation and center-jitter cropping
|   |-- protocol/            Splits and shared episode manifests
|   |-- baselines/           FEAT, MAML++, CWT-ResNet, Transformer, and GAT
|   |-- evaluation/          Shared evaluators
|   |-- analysis/            CKA utilities
|   |-- ablations/           Feature-ranking and fixed-D utilities
|   |-- data/                Data loading and raw-signal adapters
|   `-- io/                  Run and artifact management
`-- run_afop_closedset.m     Minimal AFOP entry point
```

## Quick start

1. Place the prepared files described in [`data/README.md`](data/README.md)
   under `data/`.
2. Start MATLAB in the repository root.
3. Run the minimal AFOP pipeline:

```matlab
output = run_afop_closedset();
```

For the complete benchmark workflow:

```matlab
setenv('AFOP_MODE','full')
run('experiments/closed_set/prepare_protocol.m')
run('experiments/closed_set/run_afop_evaluation.m')
```

`AFOP_MODE` accepts `smoke`, `pilot`, or `full`. These modes use 3, 50, and
500 evaluation episodes, respectively.

## Documentation

- [Complete reproduction guide](docs/REPRODUCIBILITY.md)
- [Experiment entry points](experiments/README.md)
- [Source-code modules](src/README.md)
- [Configuration reference](configs/README.md)
- [Dataset contract](data/README.md)

## Data availability

The raw tactile recordings and real-contact OOD dataset are not included in
this repository because of their size. They are currently being organized and
will be released through a cloud-storage link. The download information will
be added here and in [`data/README.md`](data/README.md) when available.

## Requirements

The code was developed with MATLAB R2023b. AFOP requires Statistics and
Machine Learning Toolbox, Signal Processing Toolbox, DSP System Toolbox,
Wavelet Toolbox, and Deep Learning Toolbox. Individual baselines and automatic
segmentation have additional requirements listed in the reproduction guide.

## Outputs

Generated protocol files are written to `artifacts/`; checkpoints and
evaluation results are written to `runs/`; progress files are written to
`logs/` or `state/`. These directories are excluded from version control.
