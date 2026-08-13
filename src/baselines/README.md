# Baselines

| Folder | Input | Training protocol |
|---|---|---|
| `feat/` | Raw four-channel trials | Source pretraining and shot-specific episodic training |
| `mamlpp/` | Raw four-channel trials | Source episodic meta-training |
| `cwt_resnet/` | CWT images | Source ResNet-50 training and episodic head adaptation |
| `tactile_transformer/` | Raw temporal sequences | Fresh support-set fit per episode |
| `channel_gat/` | Four sensor-channel nodes | Fresh support-set fit per episode |

AFO-MLP is implemented in `src/evaluation/` because it reuses the AFOP Top-D
feature subspace rather than a standalone representation backbone.
