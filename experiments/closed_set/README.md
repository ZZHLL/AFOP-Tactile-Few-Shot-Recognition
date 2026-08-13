# Closed-Set Benchmark

Start with `prepare_protocol.m`, which creates the source split, NCA ranking,
D-scan result, and shared evaluation manifest. `run_afop_evaluation.m` reproduces the
AFOP row.

Raw-signal baselines use `train_feat_encoder.m`, `train_feat_meta.m`, and
`train_mamlpp.m`. CWT-ResNet uses `prepare_cwt_images.m` followed by
`train_cwt_resnet.m`. Evaluation scripts share the stored episode manifest;
`AFOP_COMBO_INDEX` selects one N-way/K-shot setting where required.
