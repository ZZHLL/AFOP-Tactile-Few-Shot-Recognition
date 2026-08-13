# Generalization

- `run_cross_shape.m`: 12 balanced repeated group-holdout shape splits.
- `run_cross_material.m`: three directed leave-one-material-out splits.
- `run_real_contact_ood.m`: 18-class evaluation under changed contact
  conditions.

The first two runners use `AFOP_SPLIT` and `AFOP_STAGE`; the OOD runner uses
`AFOP_STAGE` and `AFOP_COMBO_INDEX`. Model fitting and evaluation stages are
kept separate so long-running jobs can be resumed by stage.
