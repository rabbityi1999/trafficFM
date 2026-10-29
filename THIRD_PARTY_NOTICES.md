# Third-party notices

The sources below retain their original ownership and licenses. TrafficFM integration changes include fixed-index traffic loaders/runners, model adapters and import/path adjustments. Numerical implementation provenance is tracked in `SOURCE_MANIFEST.json` by source-file and distributed-file hashes. Upstream revisions are given where known; no unknown revision is fabricated.

| Component | Upstream | License text included |
|---|---|---|
| Time-Series-Library | https://github.com/thuml/Time-Series-Library | `methods/baselines/Time-Series-Library/LICENSE` (MIT, THUML) |
| CSDI | https://github.com/ermongroup/CSDI | `methods/baselines/CSDI/LICENSE` (MIT, Yusuke Tashiro) |
| DiffSTG | https://github.com/wenhaomin/DiffSTG | `methods/baselines/DiffSTG/LICENSE` (MIT, wenhaomin) |
| PatchSTG | https://github.com/LMissher/PatchSTG | `methods/baselines/PatchSTG/LICENSE` (MIT, LMissher) |
| STID | https://github.com/GestaltCogTeam/STID | `methods/baselines/STID/LICENSE` (Apache-2.0); files marked as benchmark adaptations |
| Graph WaveNet components used by TrafficFM | https://github.com/nnzhan/Graph-WaveNet | `methods/TrafficFM/LICENSE.Graph-WaveNet` (MIT-style, Zonghan Wu) |

Upstream license texts absent from the copied research tree were recovered from the upstream repository for Time-Series-Library, STID and Graph WaveNet. They do not relicense unrelated code. Existing copyright headers inside source files remain in place.

## Fixed-version downloads

These sources are downloaded directly from their upstream repositories by `tools/fetch_baselines.py`. Commit and tree identifiers are recorded in `third_party/baselines.lock.json`. Downloaded checkouts are ignored by Git and are not redistributed with TrafficFM.

## Original sources and code entry points

The links below open the exact upstream files in a browser; no local download command is needed to read them. STG4Traffic is the source collection used for the four listed graph models. Original authors retain credit for their methods and implementations.

| Model | Source repository | Model code | Runner |
|---|---|---|---|
| DCRNN | [trainingl/STG4Traffic](https://github.com/trainingl/STG4Traffic) | [Model source](https://github.com/trainingl/STG4Traffic/blob/00cf6bcf85da0333f67ff983dbbb59120fc1ac4d/TrafficSpeed/model/DCRNN/dcrnn.py) | [Training entry](https://github.com/trainingl/STG4Traffic/blob/00cf6bcf85da0333f67ff983dbbb59120fc1ac4d/TrafficSpeed/DCRNN/DCRNN_Main.py) |
| DGCRN | [trainingl/STG4Traffic](https://github.com/trainingl/STG4Traffic) | [Model source](https://github.com/trainingl/STG4Traffic/blob/00cf6bcf85da0333f67ff983dbbb59120fc1ac4d/TrafficSpeed/model/DGCRN/net.py) | [Training entry](https://github.com/trainingl/STG4Traffic/blob/00cf6bcf85da0333f67ff983dbbb59120fc1ac4d/TrafficSpeed/DGCRN/DGCRN_Main.py) |
| GWNET | [trainingl/STG4Traffic](https://github.com/trainingl/STG4Traffic) | [Model source](https://github.com/trainingl/STG4Traffic/blob/00cf6bcf85da0333f67ff983dbbb59120fc1ac4d/TrafficSpeed/model/GWNET/gwnet.py) | [Training entry](https://github.com/trainingl/STG4Traffic/blob/00cf6bcf85da0333f67ff983dbbb59120fc1ac4d/TrafficSpeed/GWNET/GWNET_Main.py) |
| STGCN | [trainingl/STG4Traffic](https://github.com/trainingl/STG4Traffic) | [Model source](https://github.com/trainingl/STG4Traffic/blob/00cf6bcf85da0333f67ff983dbbb59120fc1ac4d/TrafficSpeed/model/STGCN/stgcn.py) | [Training entry](https://github.com/trainingl/STG4Traffic/blob/00cf6bcf85da0333f67ff983dbbb59120fc1ac4d/TrafficSpeed/STGCN/STGCN_Main.py) |
| HimNet | [XDZhelheim/HimNet](https://github.com/XDZhelheim/HimNet) | [Model source](https://github.com/XDZhelheim/HimNet/blob/2a9aa44d1864f6fb60bdf9d14bafd6c1681e81a5/model/HimNet.py) | [Training entry](https://github.com/XDZhelheim/HimNet/blob/2a9aa44d1864f6fb60bdf9d14bafd6c1681e81a5/scripts/train.py) |
| STD-MAE | [Jimmy-7664/STD-MAE](https://github.com/Jimmy-7664/STD-MAE) | [Model source](https://github.com/Jimmy-7664/STD-MAE/blob/29fba76216f7ff9f2b51d097786cb5c91a8dae8b/stdmae/stdmae_arch/stdmae.py) | [Training entry](https://github.com/Jimmy-7664/STD-MAE/blob/29fba76216f7ff9f2b51d097786cb5c91a8dae8b/stdmae/run.py) |
| TSFlow | [marcelkollovieh/TSFlow](https://github.com/marcelkollovieh/TSFlow) | [Model source](https://github.com/marcelkollovieh/TSFlow/blob/ff893176a41a1037f63efae674aab71480eee7d0/tsflow/model/tsflow_cond.py) | [Training entry](https://github.com/marcelkollovieh/TSFlow/blob/ff893176a41a1037f63efae674aab71480eee7d0/bin/train_model.py) |
| UniST | [tsinghua-fib-lab/UniST](https://github.com/tsinghua-fib-lab/UniST) | [Model source](https://github.com/tsinghua-fib-lab/UniST/blob/6ee69db00d8f67e550f9bc3c06f004f9f7885703/src/model.py) | [Training entry](https://github.com/tsinghua-fib-lab/UniST/blob/6ee69db00d8f67e550f9bc3c06f004f9f7885703/src/main.py) |
| STUM | [RWLinno/STUM](https://github.com/RWLinno/STUM) | [Model source](https://github.com/RWLinno/STUM/blob/3884fb50f3052f1be5d0d3a3b1aa95fb3e3e5af4/src/stum/STUM.py) | [Training entry](https://github.com/RWLinno/STUM/blob/3884fb50f3052f1be5d0d3a3b1aa95fb3e3e5af4/train_stum_ori.py) |
| FreqFlow | [moghadas76/Freq_Spect](https://github.com/moghadas76/Freq_Spect) | [Model source](https://github.com/moghadas76/Freq_Spect/blob/ecf0e354c9e8ec39d93392d0a8dadd9c8e2446d0/models/Flow_Spect.py) | [Training entry](https://github.com/moghadas76/Freq_Spect/blob/ecf0e354c9e8ec39d93392d0a8dadd9c8e2446d0/run_longExp_F.py) |

### Original project descriptions

Short titles quoted from the linked upstream READMEs:

- STG4Traffic: “STG4Traffic：A Benchmark Study of Using Spatial-Temporal Graph Neural Networks for Traffic Prediction”. [Original README](https://github.com/trainingl/STG4Traffic/blob/00cf6bcf85da0333f67ff983dbbb59120fc1ac4d/README.md).
- HimNet: “Heterogeneity-Informed Meta-Parameter Learning for Spatiotemporal Time Series Forecasting”. [Original README](https://github.com/XDZhelheim/HimNet/blob/2a9aa44d1864f6fb60bdf9d14bafd6c1681e81a5/README.md).
- STD-MAE: “Spatial-Temporal-Decoupled Masked Pre-training for Spatiotemporal Forecasting”. [Original README](https://github.com/Jimmy-7664/STD-MAE/blob/29fba76216f7ff9f2b51d097786cb5c91a8dae8b/README.md).
- TSFlow: “Flow Matching with Gaussian Process Priors for Probabilistic Time Series Forecasting”. [Original README](https://github.com/marcelkollovieh/TSFlow/blob/ff893176a41a1037f63efae674aab71480eee7d0/README.md).
- UniST: “UniST: A Prompt-Empowered Universal Model for Urban Spatio-Temporal Prediction”. [Original README](https://github.com/tsinghua-fib-lab/UniST/blob/6ee69db00d8f67e550f9bc3c06f004f9f7885703/README.md).
- STUM: “Cross Spatial and Time: A Spatio-Temporal Unitized Model for Traffic Flow Forecasting”. [Original README](https://github.com/RWLinno/STUM/blob/3884fb50f3052f1be5d0d3a3b1aa95fb3e3e5af4/README.md).
- FreqFlow: “FreqFlow: Long-term forecasting using lightweight flow matching”. [Original README](https://github.com/moghadas76/Freq_Spect/blob/ecf0e354c9e8ec39d93392d0a8dadd9c8e2446d0/README.md).

No repository-wide redistribution license was identified for these seven snapshots during the release audit. Their original notices and any component-specific terms remain applicable; the downloader grants no additional rights. Please cite the original method papers when using their code. The authors' choice of license for new TrafficFM contributions remains pending.
