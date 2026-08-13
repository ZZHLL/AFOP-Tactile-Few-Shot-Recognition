# Preprocessing

| Entry point | Purpose |
|---|---|
| `run_auto_segmentation.m` | Segment periodic four-channel TXT recordings |
| `prepare_dataset_from_segmentation.m` | Build aligned, jittered, labeled, and feature-pool artifacts |
| `prepare_force_speed_target.m` | Generate the force/speed/noise target domain |
| `prepare_phase_target.m` | Generate one phase-shift severity |

Segmentation supports the fast fixed-onset mode and the optional joint
onset/length search through `AFOP_SEGMENT_JOINT_SEARCH`.
