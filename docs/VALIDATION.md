# Release validation

Packaging validation on 2026-10-29:

- 34 CPU/offline tests passed via `python -m unittest discover -s tests -v`.
- Seven locked upstream repositories were fetched successfully, covering ten baseline names and 613 selected source/configuration/documentation files. `python tools/fetch_baselines.py --all --check` passed exact commit, tree, origin, entry-point and file-content checks.
- Downloader tests cover model coverage, shared-repository deduplication, immutable revisions, path validation, source-only checkout, idempotent reuse and preservation of local edits. They use local fixture repositories and require no network or training dependencies.
- Linear/VAR checks cover temporal axes, per-sensor projections, gradients, explicit VAR design-matrix equivalence, lag/source/target orientation, recursive forecasts, training-only row selection, singular systems, checkpoint roundtrips, causal filling, masked metrics, split boundaries and both end-to-end CLI runs on synthetic data.
- The tests exercise decoder/encoder gradients, optimizer updates, missing-target masks, sample axes, independent samples, Euler and alternative solvers, conditional guidance, checkpoint-version isolation, fractional time-of-day indexing, empirical CRPS, and STID output shape/gradient flow.
- Python syntax compilation passed for the initial distributed source set.
- `methods/TrafficFM/train_pems.py --help` passed.
- `methods/baselines/STID/run_unified_144.py --help` passed. CSDI and DiffSTG CLI import checks remain untested in the core environment; they require `linear_attention_transformer` and `easydict`, respectively.

## Validation scope

Completed checks cover CPU/offline regression tests, source retrieval and the CLI checks listed above. Full traffic-data training and numerical reproduction of the paper tables have not been validated in this release. Optional baseline dependencies are configured separately.
