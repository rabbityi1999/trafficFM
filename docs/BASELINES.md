# Baseline inventory

All 24 baseline names below occur in the current main comparison. **Bundled** means source files are included; **Fixed-version fetch** means the implementation is retrieved from a commit locked in `third_party/baselines.lock.json`. The release provides 14 bundled implementations and download entries for the remaining 10. See [upstream download and model locations](UPSTREAM_BASELINES.md).

| Baseline | Distribution | Location / upstream |
|---|---|---|
| Autoformer | Bundled library | `methods/baselines/Time-Series-Library/models/Autoformer.py` |
| CSDI | Bundled + traffic runner | `methods/baselines/CSDI/run_unified_144.py` |
| DCRNN | Fixed-version fetch | `python tools/fetch_baselines.py --models DCRNN` |
| DGCRN | Fixed-version fetch | `python tools/fetch_baselines.py --models DGCRN` |
| DiffSTG | Bundled + traffic runner | `methods/baselines/DiffSTG/run_unified_144.py` |
| DLinear | Bundled library | `methods/baselines/Time-Series-Library/models/DLinear.py` |
| FEDformer | Bundled library | `methods/baselines/Time-Series-Library/models/FEDformer.py` |
| FreqFlow | Fixed-version fetch | `python tools/fetch_baselines.py --models FreqFlow` |
| GWNET | Fixed-version fetch | `python tools/fetch_baselines.py --models GWNET` |
| HimNet | Fixed-version fetch | `python tools/fetch_baselines.py --models HimNet` |
| Informer | Bundled library | `methods/baselines/Time-Series-Library/models/Informer.py` |
| iTransformer | Bundled library | `methods/baselines/Time-Series-Library/models/iTransformer.py` |
| LightTS | Bundled library | `methods/baselines/Time-Series-Library/models/LightTS.py` |
| Linear | Bundled + traffic runner | [Direct temporal projection](../methods/baselines/Linear/README.md) |
| PatchSTG | Bundled source + development configs | `methods/baselines/PatchSTG/`; spatial metadata not bundled |
| PatchTST | Bundled library | `methods/baselines/Time-Series-Library/models/PatchTST.py` |
| STD-MAE | Fixed-version fetch | `python tools/fetch_baselines.py --models STD-MAE` |
| STGCN | Fixed-version fetch | `python tools/fetch_baselines.py --models STGCN` |
| STID | Bundled + traffic runner | `methods/baselines/STID/run_unified_144.py` |
| STUM | Fixed-version fetch | `python tools/fetch_baselines.py --models STUM` |
| Transformer | Bundled library | `methods/baselines/Time-Series-Library/models/Transformer.py` |
| TSFlow | Fixed-version fetch | `python tools/fetch_baselines.py --models TSFlow` |
| UniST | Fixed-version fetch | `python tools/fetch_baselines.py --models UniST` |
| VAR | Bundled + traffic runner | [Multivariate autoregression](../methods/baselines/VAR/README.md) |

STG4Traffic upstream: https://github.com/trainingl/STG4Traffic. The benchmark adapters should not be conflated with an unmodified upstream release.

## Recorded upstream revisions

These revisions identify upstream bases from the research checkout, not guarantees that local adapters were unmodified:

| Source | Recorded commit |
|---|---|
| FreqFlow / Freq_Spect | `ecf0e354c9e8ec39d93392d0a8dadd9c8e2446d0` |
| PatchSTG | `8b298b230227a304a082e05241c69dca3740a73d` |
| STUM | `3884fb50f3052f1be5d0d3a3b1aa95fb3e3e5af4` |
| TSFlow | `ff893176a41a1037f63efae674aab71480eee7d0` |
| UniST | `6ee69db00d8f67e550f9bc3c06f004f9f7885703` |

STMPNet and SpecSTG also appeared in supplementary probability investigations. Their sources are https://github.com/yang-1201/STMPNet (`b06f454c1604b7b722cb907160e8b79c99821185`) and https://github.com/lequanlin/SpecSTG (`c434bcd27507851ad9f40aa92e7147d3e91e5b20`). They are not counted among the 24 rows above and their experiment adapters are not included in this initial source release.

Use `python tools/fetch_baselines.py --all` to retrieve the ten upstream entries and `--all --check` to verify the downloaded source against the lockfile. Their local benchmark adapters and configuration records are tracked separately.
