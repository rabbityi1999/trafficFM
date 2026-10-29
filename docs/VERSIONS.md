# Implementation and checkpoint guide

| Model module | Purpose / objective | Checkpoint handling |
|---|---|---|
| `models/default.py` | Archived latent flow implementation; the external training engine also supports a point objective | Keep the original objective and architecture with the checkpoint |
| `models/default_direct_mae.py` | Direct-prediction variant | Not interchangeable with a flow-model checkpoint |
| `models/default_direct_mse.py` | Direct-prediction variant | Record the actual external training objective |
| `models/data_space_fixed.py` | Data-space conditional flow matching; embedding and decoding participate in each velocity evaluation | Requires `--train_objective model`; includes a version buffer |
| `models/history_calendar_fixed.py` | Data-space model with explicit fractional time-of-day lookup handling | Record time-of-day encoding; semantics differ even when parameter shapes match |

The legacy implementation is retained for provenance. Its latent velocity-regression objective does not train the final token decoder; the regression suite explicitly checks this difference. The data-space implementation connects that decoder to the training objective and solver. A repaired implementation is not evidence that an earlier table row used that implementation.

For fractional time-of-day features, the calendar variant converts `slot/288` to an integer embedding index while leaving the encoder inputs unchanged. Integer slot features require an explicit format choice.

An experiment record should contain model module, objective, dataset and split hashes, feature encoding, mask convention, architecture arguments, seed, optimizer, training budget, checkpoint hash, solver and step count K, sample count M, calibration rule, and validation-selection rule. Keep point-only predictions distinct from sample-based probability evaluation.

This source release does not assign new checkpoints to historical paper rows. A row-level configuration/checkpoint archive is a separate release artifact still to be supplied.
