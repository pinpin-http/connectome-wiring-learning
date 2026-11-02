# Documents

Three kinds of documents live here. Only the first kind is needed to check the preprint.

## Protocols of the reported experiments

Each was fixed before the data it governs; frozen protocols are never edited afterwards (a deviation is recorded in a
separate file). Experiment numbers follow [ARCHITECTURE.md](../ARCHITECTURE.md).

| experiment | protocol | note |
|---|---|---|
| 1. Visual tasks (Study S) | [STUDY_S.md](STUDY_S.md) | declared before the main runs; reading criteria and their validation |
| Statistical tools of experiment 1 | [STATISTICAL_AUDIT_20261019.md](STATISTICAL_AUDIT_20261019.md) | SHA-256 recorded in `results/` |
| 2-3. Imposed actions, probe, equalized activity | [experience_commune/PROTOCOL_FROZEN.md](experience_commune/PROTOCOL_FROZEN.md), [DEVIATIONS.md](experience_commune/DEVIATIONS.md) | |
| 4. Equal-tuning controls | [revision48/PROTOCOL_FROZEN.md](revision48/PROTOCOL_FROZEN.md) | SHA-256 recorded in `results/` |
| 4, post hoc ceiling sensitivity | [REVISION_UNSATURATED.md](REVISION_UNSATURATED.md) | fixed before reading the measures it uses; labelled post hoc |
| 5. Eye-map permutation | [retino_align/PROTOCOL_FROZEN.md](retino_align/PROTOCOL_FROZEN.md) | |
| 6. Equalized activity, frozen network | [activity_match/PROTOCOL_FROZEN.md](activity_match/PROTOCOL_FROZEN.md), [V2](activity_match/PROTOCOL_FROZEN_V2.md) (family E), [V2B](activity_match/PROTOCOL_FROZEN_V2B.md) (family S) | |
| 7. *Super Mario Bros.* | [RL_STUDY.md](RL_STUDY.md) §17-§30, [PERCEPTION_CREDIT_STUDY.md](PERCEPTION_CREDIT_STUDY.md), [GENERALISATION_STUDY.md](GENERALISATION_STUDY.md), [MARIO_PILOT_PLAN.md](MARIO_PILOT_PLAN.md) | declarations appended section by section as the study went |

## Guides

- [REPRODUCIBILITY.md](REPRODUCIBILITY.md): every table and figure of the preprint, the script and the file it comes from.
- [ROM.md](ROM.md): supplying your own copy of the game (never distributed).

## Development record

- [PROTOCOL.md](PROTOCOL.md): the founding protocol and the early gates A-G (closed loop, operating point, interface,
  learning-rule validation) that set up the model; their scripts are in `scripts/gates/`.
- [CALIBRATION.md](CALIBRATION.md): audit of which model parameters are constrained by data and which are free.

These two are kept as written at the time, dates and open questions included: they show how the model was set up,
not results of the preprint. Where they mention files that are no longer in the repository (working notes, journals),
those files were development material and are not needed to reproduce anything.
