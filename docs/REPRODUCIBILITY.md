# AFOP MATLAB Reproducibility Code

This package contains the core MATLAB code for AFOP and the principal
experiments added during revision. It intentionally excludes datasets,
checkpoints, generated results, plotting utilities, cluster launchers, and
legacy auxiliary baselines.

## Included experiments

- preprocessing: automatic trial segmentation, nominal-center estimation,
  center-jitter cropping, and construction of the 386-D feature pool;
- AFOP: stratified splitting, source-only NCA ranking, episodic D-scan, and
  prototype-initialized cosine-softmax adaptation;
- closed-set evaluation with one shared episodic manifest;
- 12 balanced repeated cross-shape splits and three directed cross-material
  splits;
- force/speed/noise perturbation and the sigma sweep;
- the four added raw-signal baselines: FEAT, MAML++, Tactile Transformer, and
  Channel-GAT;
- the submitted CWT-ResNet50 training and episodic evaluation pipeline;
- the AFO-MLP episode-time adaptation baseline;
- the 18-class real-contact OOD protocol and all principal model evaluators;
- shared backbone evaluation scripts and checkpoint-safe result output.
- source-validation feature-ranking, fixed-D, component-isolation, CKA,
  t-SNE, paired-bootstrap, sign-flip permutation, and Holm-analysis scripts.

FEAT uses supervised source pretraining followed by shot-specific episodic
meta-training. MAML++ uses raw-signal episodic meta-training. Tactile
Transformer and Channel-GAT have no source-pretraining script in this package:
the reported protocol fits a fresh network from each episode's support set,
which is implemented inside `evaluate_support_only_raw_model_checkpointed.m`.
All four baselines consume raw 4-channel signals, not the handcrafted AFOP
feature pool.

CWT-ResNet converts each four-channel trial into a vertically tiled CWT image,
trains a source-domain ResNet-50 with the early layers frozen, and adapts a
prototype-initialized episodic head. AFO-MLP consumes the same source-selected
Top-D feature subspace as AFOP, but fits a two-layer MLP and cosine head in each
episode.

## Required data files

Create a `data` subdirectory and place the following files in it:

- `dataset2.mat`, variable `dataset3_jittered`, size 36 x 60 cells; each cell
  is a 4 x 4000 trial ordered as PVDF1, PVDF2, SG1, SG2;
- `labels_table.mat`, variable `labels_table`;
- `features_with_labels.mat`, variable `features_with_labels`;
- `aligned_8s.mat`, variable `dataset_8s`, size 36 x 60 cells with 4 x 8001
  aligned trials, required only to regenerate perturbation targets.

For the real-contact OOD experiment, additionally create `data/ood18` with:

- `dataset3.mat`, variable `dataset3`, size 18 x 60 cells with 4 x 4000 trials;
- `labels_table.mat`, variable `labels_table`, with rows in the same
  class-major/trial-minor order as `dataset3`.

The package never writes into `data`. Protocol artifacts are placed in
`artifacts`, trained models and evaluations in `runs`, and status files in
`logs` or `state`. Existing artifacts are verified or retained rather than
silently overwritten.

## Closed-set workflow

Run MATLAB from the package root. Set `AFOP_MODE` to `smoke`, `pilot`, or
`full`; these correspond to 3, 50, and 500 evaluation episodes.

```matlab
run('experiments/closed_set/prepare_protocol.m')
run('experiments/closed_set/train_feat_encoder.m')
setenv('AFOP_FEAT_SHOT','1'); run('experiments/closed_set/train_feat_meta.m')
setenv('AFOP_FEAT_SHOT','3'); run('experiments/closed_set/train_feat_meta.m')
setenv('AFOP_FEAT_SHOT','5'); run('experiments/closed_set/train_feat_meta.m')
run('experiments/closed_set/train_mamlpp.m')
run('experiments/closed_set/run_afop_evaluation.m')
run('experiments/closed_set/evaluate_raw_baselines.m')
```

Generate CWT images once, train the source CWT-ResNet, and evaluate one table
cell selected by `AFOP_COMBO_INDEX`:

```matlab
run('experiments/closed_set/prepare_cwt_images.m')
run('experiments/closed_set/train_cwt_resnet.m')
setenv('AFOP_COMBO_INDEX','1')
run('experiments/closed_set/evaluate_cwt_resnet.m')
```

The same shared closed-set manifest and Top-D columns are used by AFO-MLP:

```matlab
setenv('AFOP_COMBO_INDEX','1')
run('experiments/closed_set/evaluate_afo_mlp.m')
```

The two support-only families can also be evaluated separately:

```matlab
setenv('AFOP_MODEL_FAMILY','tactile_transformer')
run('experiments/closed_set/evaluate_support_only.m')
setenv('AFOP_MODEL_FAMILY','channel_gat')
run('experiments/closed_set/evaluate_support_only.m')
```

The support-only evaluator uses the complete support set as one minibatch.
This fixes the optimization budget per epoch across N-way settings and matches
the corrected formal evaluation.

## Cross-shape and cross-material

Both runners use `AFOP_STAGE` to select one stage. Cross-shape has 12 splits;
cross-material has three directed splits. The supported stages are
`prepare`, `afop_eval`, `feat_pretrain`, `feat_meta`, `feat_eval`,
`mamlpp_train`, `mamlpp_eval`, `transformer_eval`, and `gat_eval`.

```matlab
setenv('AFOP_MODE','full')
setenv('AFOP_STAGE','prepare'); run('experiments/generalization/run_cross_shape.m')
setenv('AFOP_SPLIT','1'); setenv('AFOP_STAGE','afop_eval');
run('experiments/generalization/run_cross_shape.m')
```

For FEAT, run `feat_pretrain`, `feat_meta`, and `feat_eval` in that order for
each split. For MAML++, run `mamlpp_train` then `mamlpp_eval`. Transformer and
Channel-GAT need only their evaluation stage; set `AFOP_COMBO_INDEX` from 1 to
4. Replace `run_cross_shape.m` with `run_cross_material.m` for the
three material splits.

## Force/speed/noise perturbation

First create the fixed target domain, then create the shared manifest and run
the desired model stage.

```matlab
run('experiments/preprocessing/prepare_force_speed_target.m')
setenv('AFOP_STAGE','prepare'); run('experiments/robustness/run_force_speed.m')
setenv('AFOP_STAGE','afop_eval'); run('experiments/robustness/run_force_speed.m')
```

The other supported stages are `feat_eval`, `mamlpp_eval`,
`transformer_eval`, and `gat_eval`. FEAT and MAML++ use the frozen nominal
closed-set checkpoints. Transformer and Channel-GAT are fitted only on target
support samples. AFOP re-extracts the target feature pool but fits NCA and
D-scan exclusively on the nominal source partition.

## End-to-end preprocessing

Run automatic segmentation first. The continuation script directly consumes
the baseline-corrected `fullData` produced by segmentation, then estimates or
reuses each class's nominal event
center, creates the aligned 8001-sample source and default center-jittered
4000-sample trials,
builds the labels, and extracts the 386-D pool:

```matlab
setenv('AFOP_SEGMENTATION_ARTIFACT','D:\path\segmentation_yyyymmdd_hhmmss.mat')
setenv('AFOP_PREPARED_OUTPUT','D:\path\prepared_data')
setenv('AFOP_CENTER_CHANNEL','3')
setenv('AFOP_SHIFT_SIGMA','200')
setenv('AFOP_SHIFT_CLIP','600')
run('experiments/preprocessing/prepare_dataset_from_segmentation.m')
```

If the event edges occupy only part of a segmented trial, set
`AFOP_CENTER_ROI` to `first,last` in samples. A manually verified
`grooveCenterIdx` stored in `shapeDataSet` takes precedence over automatic
center estimation. The output files are `aligned_8s.mat`, `dataset2.mat`,
`labels_table.mat`, and `features_with_labels.mat`.

## Sigma sweep

Set `AFOP_SHIFT_SIGMA` in samples, generate the target, and evaluate all model
families on the same fixed episodes. The target generator uses the common
amplitude range [0.7, 1.3], speed range [0.85, 1.15], and relative noise level
0.05 while varying sigma; the shift is clipped at three sigma.

```matlab
setenv('AFOP_SHIFT_SIGMA','200')
run('experiments/preprocessing/prepare_phase_target.m')
setenv('AFOP_STAGE','prepare'); run('experiments/robustness/run_phase_sweep.m')
setenv('AFOP_STAGE','afop_eval'); run('experiments/robustness/run_phase_sweep.m')
```

Repeat for the required sigma values and model stages. The sigma runner uses
the force/speed protocol manifest as the reference so support/query identities
remain paired across severity levels.

## Real-contact OOD evaluation

The OOD runner evaluates 5/10/12/18-way 1-shot and 3-shot recognition using a
single shared target-domain manifest. `AFOP_MODE=smoke`, `pilot`, and `full`
run 3, 50, and 500 episodes, respectively. Prepare the target features and
manifest first:

```matlab
setenv('AFOP_MODE','full')
setenv('AFOP_STAGE','prepare')
run('experiments/generalization/run_real_contact_ood.m')
```

Set `AFOP_COMBO_INDEX` from 1 to 8 and choose one of `afop_eval`,
`afo_mlp_eval`, `feat_eval`, `mamlpp_eval`, `transformer_eval`, or `gat_eval`.
AFOP and AFO-MLP reuse the source-only NCA/D-scan frontend; FEAT and MAML++
reuse their frozen source checkpoints; Transformer and Channel-GAT are fitted
from target support samples only.

For CWT-ResNet, create target CWT embeddings once before episodic evaluation:

```matlab
setenv('AFOP_STAGE','cwt_prepare'); run('experiments/generalization/run_real_contact_ood.m')
setenv('AFOP_STAGE','cwt_eval'); run('experiments/generalization/run_real_contact_ood.m')
```

The source protocol and source checkpoints must already exist in `artifacts`
and `runs`; no target query samples are used for fitting or model selection.

## Automatic trial segmentation

`run_auto_segmentation.m` reads every TXT recording in a user-specified folder,
applies the same four-channel Savitzky-Golay filtering, estimates the trial
period, and extracts 60 baseline-corrected trials. Its default fast mode fixes
the template start from the second reliable cycle landmark and scans candidate
lengths. Setting `AFOP_SEGMENT_JOINT_SEARCH=true` enables the complete
Algorithm-1 search over both the onset set $\mathcal{N}_0$ and length set
$\mathcal{L}$. The output contains `dataset0`,
`dataset0_filtered_sg`, and `shapeDataSet` and is timestamped to prevent an
existing file from being overwritten.

```matlab
setenv('AFOP_SEGMENT_INPUT','D:\path\to\raw_txt')
setenv('AFOP_SEGMENT_OUTPUT','D:\path\to\segmentation_output')
setenv('AFOP_SEGMENT_CHANNEL','3')
setenv('AFOP_SEGMENT_TRIALS','60')
setenv('AFOP_SEGMENT_PLOTS','false')
setenv('AFOP_SEGMENT_JOINT_SEARCH','false') % default fast mode
run('experiments/preprocessing/run_auto_segmentation.m')
```

For the full joint search, the defaults are
`AFOP_SEGMENT_ONSET_RANGE=0.10`, `AFOP_SEGMENT_LENGTH_RANGE=0.10`,
`AFOP_SEGMENT_ONSET_STEP=20`, and `AFOP_SEGMENT_LENGTH_STEP=20`. The range
values are the $\alpha_N$ and $\alpha_L$ fractions of the estimated period;
the step values are $\Delta_N$ and $\Delta_L$ in samples. These effective
settings are stored in the timestamped output artifact as `config.outlierConfig`.

Each TXT file must contain either four signal columns or a time column followed
by four signal columns. The expected channel order is PVDF1, PVDF2, SG1, SG2.

## Feature-optimization ablations

The ranking, fixed-D, and component-isolation experiments share the formal
closed-set split and test episode manifest:

```matlab
setenv('AFOP_MODE','full')
run('experiments/ablations/run_feature_optimization.m')
```

NCA, ReliefF, mRMR, and PCA are fitted on `source_train`; every candidate D is
evaluated on the same 5-way 5-shot `source_val` episodes with one query per
class, and D* is the candidate with the highest mean validation accuracy. The component outputs compare the
full 386-D pool with prototype initialization, Top-D with prototype
initialization and no fine-tuning, and full AFOP. Fixed D values are
4, 8, 16, and 32.

## Wavelet sensitivity

The wavelet-kernel and decomposition-level studies rebuild the complete
candidate pool for every setting. They use the same closed-set split and
5-way 5-shot source-validation D-scan as the main pipeline. No additional
wavelet-specific cap is placed on D; the formal candidate set is `1:20`.

```matlab
setenv('AFOP_WAVELET_STUDY','kernel') % nine kernels at three levels
setenv('AFOP_WAVELET_INDEX','1')
run('experiments/ablations/run_wavelet_sensitivity.m')

setenv('AFOP_WAVELET_STUDY','level')  % rbio2.2 at levels 1--8
setenv('AFOP_WAVELET_INDEX','1')
run('experiments/ablations/run_wavelet_sensitivity.m')
```

Kernel indices follow `db5`, `coif5`, `sym5`, `fk6`, `dmey`, `bior2.2`,
`rbio2.2`, `bior2.6`, and `rbio2.6`. Each run evaluates 5-, 12-, and 36-way
1-shot recognition over 500 shared episodes and saves the full D-scan curve.

## Representation and statistical analysis

`experiments/analysis/run_representation_analysis.m` expects
`AFOP_REPRESENTATION_INPUT` to point
to a MAT file containing `representations`, a structure whose fields are
sample-aligned model embedding matrices. Optional `referenceRepresentations`
fields provide condition-specific matrices for CKA-to-reference. The script
exports pairwise linear CKA, per-model t-SNE coordinates, and stability values.
`experiments/analysis/prepare_representations.m` constructs the AFOP, Direct-Proto,
SG-only, and PVDF-only matrices directly from the released feature pool using
the formal source split. Embeddings from learned baselines can be appended as
additional fields when reproducing the full cross-encoder CKA panel.

`experiments/analysis/compute_paired_statistics.m` expects
`AFOP_STATISTICS_INPUT` to contain a
`comparisons` structure with `task`, `baseline`, `referenceAccuracy`, and
`baselineAccuracy`. Accuracy vectors must come from the same episode manifest.
It reports paired bootstrap confidence intervals, two-sided sign-flip
permutation p-values, and Holm-adjusted p-values within each task.

## Software requirements

The code was developed for MATLAB R2023b and requires the Statistics and
Machine Learning, Signal Processing, DSP System, Wavelet, and Deep Learning
Toolboxes. A CUDA-capable GPU is expected for the four raw-signal baselines;
AFOP and feature construction can run on CPU.

CWT-ResNet also requires the Deep Learning Toolbox Model for ResNet-50 Network
support package. Automatic segmentation uses parallel search and normalized
template correlation and therefore requires Parallel Computing Toolbox and
Image Processing Toolbox in addition to Signal Processing Toolbox.

## Reproducibility safeguards

- all splits and episode manifests use recorded deterministic seeds;
- support and query rows are checked for disjointness;
- NCA, D-scan, raw normalization, and checkpoint selection never use target
  query samples;
- target-domain feature engineering is recomputed from target raw signals;
- result and checkpoint paths are append-only or explicitly resumable.
