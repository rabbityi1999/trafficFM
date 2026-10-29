# Baseline inventory

All 24 baseline names below occur in the current main comparison. **Bundled** means source files are included. **Reference** means the source is linked upstream. The source package includes 14 baseline implementations. Linear and VAR have versioned model definitions, runnable traffic entry points and usage documentation in their respective directories.

| Baseline | Distribution | Location / upstream |
|---|---|---|
| Autoformer | Bundled library | `methods/baselines/Time-Series-Library/models/Autoformer.py` |
| CSDI | Bundled + traffic runner | `methods/baselines/CSDI/run_unified_144.py` |
| DCRNN | Reference | STG4Traffic source collection; see below |
| DGCRN | Reference | STG4Traffic source collection |
| DiffSTG | Bundled + traffic runner | `methods/baselines/DiffSTG/run_unified_144.py` |
| DLinear | Bundled library | `methods/baselines/Time-Series-Library/models/DLinear.py` |
| FEDformer | Bundled library | `methods/baselines/Time-Series-Library/models/FEDformer.py` |
| FreqFlow | Reference | https://github.com/moghadas76/Freq_Spect |
| GWNET | Reference | STG4Traffic benchmark adaptation; https://github.com/nnzhan/Graph-WaveNet |
| HimNet | Reference | https://github.com/XDZhelheim/HimNet |
| Informer | Bundled library | `methods/baselines/Time-Series-Library/models/Informer.py` |
| iTransformer | Bundled library | `methods/baselines/Time-Series-Library/models/iTransformer.py` |
| LightTS | Bundled library | `methods/baselines/Time-Series-Library/models/LightTS.py` |
| Linear | Bundled + traffic runner | [Direct temporal projection](../methods/baselines/Linear/README.md) |
| PatchSTG | Bundled source + development configs | `methods/baselines/PatchSTG/`; spatial metadata not bundled |
| PatchTST | Bundled library | `methods/baselines/Time-Series-Library/models/PatchTST.py` |
| STD-MAE | Reference | https://github.com/Jimmy-7664/STD-MAE |
| STGCN | Reference | STG4Traffic source collection |
| STID | Bundled + traffic runner | `methods/baselines/STID/run_unified_144.py` |
| STUM | Reference | https://github.com/RWLinno/STUM |
| Transformer | Bundled library | `methods/baselines/Time-Series-Library/models/Transformer.py` |
| TSFlow | Reference | https://github.com/marcelkollovieh/TSFlow |
| UniST | Reference | https://github.com/tsinghua-fib-lab/UniST |
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

For the remaining upstream-only entries, fixed-revision retrieval is the selected distribution approach. Their local benchmark adapters and configuration records are tracked separately.
