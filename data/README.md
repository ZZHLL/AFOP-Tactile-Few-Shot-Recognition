# Data

The raw tactile recordings and real-contact OOD dataset are currently being
organized. Because of their size, they will be distributed through a
cloud-storage link rather than committed to this repository. Download details
will be added to this file when the release is ready.

This directory is excluded from version control. After downloading or
preparing the data, the closed-set pipeline expects:

| File | MATLAB variable | Contract |
|---|---|---|
| `dataset2.mat` | `dataset3_jittered` | 36 x 60 cell array; each trial is 4 x 4000 |
| `labels_table.mat` | `labels_table` | One row per class/trial pair |
| `features_with_labels.mat` | `features_with_labels` | Prepared 386-D feature pool |
| `aligned_8s.mat` | `dataset_8s` | 36 x 60 cells of 4 x 8001 aligned trials |

Channel order is `PVDF1, PVDF2, SG1, SG2`.

The real-contact experiment additionally expects `data/ood18/dataset3.mat`
and `data/ood18/labels_table.mat`. See
[`docs/REPRODUCIBILITY.md`](../docs/REPRODUCIBILITY.md) for preprocessing and
target-domain generation.
