# TrafficFM

Research code for **TrafficFM: Heterogeneity-Aware Conditional Flow Matching for Long-Horizon Traffic Forecasting**.

This repository brings together TrafficFM and baseline implementations for traffic forecasting. The principal task is **144 historical steps → 144 future steps** at five-minute resolution on **METR-LA (207 sensors)** and **PEMS-BAY (325 sensors)**.

## Release contents

- `methods/TrafficFM/`: model variants, training engines, and evaluation entry points.
- `methods/baselines/Time-Series-Library/`: Autoformer, DLinear, FEDformer, Informer, iTransformer, LightTS, PatchTST, Transformer, and supporting library modules.
- `methods/baselines/{CSDI,DiffSTG,STID,PatchSTG}/`: baseline implementations and available traffic adapters.
- [Linear](methods/baselines/Linear/README.md) and [VAR](methods/baselines/VAR/README.md): direct temporal projection and multivariate autoregression, with training/fitting, checkpoint saving and fixed-index evaluation.
- `tests/`: synthetic CPU regression tests; no traffic data or GPU is required.
- [Baseline inventory](docs/BASELINES.md): all 24 baseline names in the current main comparison, with code availability and upstream references.
- [Data and execution](docs/RUNNING.md), [version guide](docs/VERSIONS.md), and [third-party notices](THIRD_PARTY_NOTICES.md).

This is an **initial source release**, not a regenerated leaderboard. Source availability, executable tests, and reproduction of a published table row are distinct statuses. The inventory records these separately. Datasets, trained weights, and historical run logs are not bundled.

## Quick start

Use an isolated environment; the baseline libraries have different optional dependencies.

```bash
git clone https://github.com/rabbityi1999/trafficFM.git
cd trafficFM
python -m venv .venv
# Linux/macOS: source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -r requirements-core.txt
python -m unittest discover -s tests -v
```

The core regression suite was exercised on CPU with PyTorch 2.8.0. This is a packaging-test environment, not a claim about the environment used for every paper experiment. Per-baseline requirements and optional packages must be installed separately; do not combine every baseline into one shared environment without resolving dependencies.

For training and data paths, see [RUNNING.md](docs/RUNNING.md). CLI defaults are examples or archived development settings, not a universal paper configuration.

## Model selection

`default`, `default_direct_mae`, `data_space_fixed`, and `history_calendar_fixed` are different implementations/objectives. Select one explicitly and retain its configuration with its checkpoint. Do not interchange their checkpoints or infer their paper-table mapping from a filename. See [VERSIONS.md](docs/VERSIONS.md).

## Attribution and licensing

Baseline copyrights and licenses remain with their original authors. Included license texts and upstream links are listed in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). Upstream-only entries are references, not vendored implementations. A repository-wide license for the original TrafficFM contributions has not yet been selected by the authors.
