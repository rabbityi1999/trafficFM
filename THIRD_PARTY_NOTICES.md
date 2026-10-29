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

| Models | Upstream |
|---|---|
| DCRNN, DGCRN, GWNET, STGCN | https://github.com/trainingl/STG4Traffic |
| HimNet | https://github.com/XDZhelheim/HimNet |
| STD-MAE | https://github.com/Jimmy-7664/STD-MAE |
| TSFlow | https://github.com/marcelkollovieh/TSFlow |
| UniST | https://github.com/tsinghua-fib-lab/UniST |
| STUM | https://github.com/RWLinno/STUM |
| FreqFlow | https://github.com/moghadas76/Freq_Spect |

No repository-wide redistribution license was identified for these seven snapshots during the release audit. Their original notices and any component-specific terms remain applicable; the downloader grants no additional rights. Please cite the original method papers when using their code. The authors' choice of license for new TrafficFM contributions remains pending.
