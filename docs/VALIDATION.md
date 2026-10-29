# Release validation

Packaging validation on 2026-10-29:

- 28 synthetic CPU tests passed via `python -m unittest discover -s tests -v`.
- Linear/VAR checks cover temporal axes, per-sensor projections, gradients, explicit VAR design-matrix equivalence, lag/source/target orientation, recursive forecasts, training-only row selection, singular systems, checkpoint roundtrips, causal filling, masked metrics, split boundaries and both end-to-end CLI runs on synthetic data.
- The tests exercise decoder/encoder gradients, optimizer updates, missing-target masks, sample axes, independent samples, Euler and alternative solvers, conditional guidance, checkpoint-version isolation, fractional time-of-day indexing, empirical CRPS, and STID output shape/gradient flow.
- Python syntax compilation passed for the initial distributed source set.
- `methods/TrafficFM/train_pems.py --help` passed.
- `methods/baselines/STID/run_unified_144.py --help` passed. CSDI and DiffSTG CLI imports could not be tested in the core environment because `linear_attention_transformer` and `easydict`, respectively, are absent; these require separate baseline dependencies.

These checks do not constitute a traffic-data training run or numerical reproduction of every paper table. No new training, GPU allocation or test-set tuning was performed during packaging. Optional baseline dependencies are not part of the core CPU test environment.
