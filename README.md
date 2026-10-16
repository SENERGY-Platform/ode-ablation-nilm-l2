# ode-ablation-nilm-l2

An analytics operator for the SENERGY platform, scaffolded by the Operator
Development Environment. Every file here is yours to change, including this one.

## What this operator does, and how it got here

### The task

Estimate the electrical power drawn by the heat pump's compressor, as a 15-minute
mean in watts, from the site's main meter (transformer-rated, factor 50). The
compressor's own sub-meter may be used for training and evaluation, but the operator
does not receive it at runtime. Any other covariate is allowed. The criterion, in
`evaluation.yaml`, is a mean absolute error of at most 200 W over the test window.

### Status

**Not complete against the criterion.** The best evaluation (commit `cc70174`,
reproduced exactly by a second run) scores **790.5 W MAE** over February 2026,
against the 200 W threshold. Predicting zero everywhere scores about 1,877 W on the
January hold-out, so the operator cuts the error by more than half. The reason it
stalls well above 200 W is the main meter itself; see "What limits it" below.

### Finding the data

The ontology search for power with the "Heating" and "Main" aspects matched no
device type, because the meters involved are not described that way. An unfiltered
device listing found them:

| Device | Role | Series used |
|---|---|---|
| Hauptzähler (Tasmota, pulse) | main meter, operator input | `sensor.data.obis_1_8_0_f` |
| Emonio Wärmepumpe Kompressor | compressor sub-meter, training label and evaluation target | `root.egy_watt` |
| open-meteo-history (51.2, 12.7) | outdoor temperature at the site's coordinates, operator input | `value.temperature_2m` |
| Emonio Ladesteckdose | EV socket; a training-only topic by the developer's choice, so the model does not use it | `root.egy_kwh` |

Not used, and why:

- "Emonio Kompressor": stopped reporting in April 2025, so it cannot cover the test window.
- "Emonio Wärmepumpe Heizung": carries under 1 W.
- The PV inverters: stopped reporting in April 2025.

### What profiling showed about the main meter

- Its instantaneous power channel, `obis_16_7`, has carried **no value at any point**
  in the stored history: none in the two weeks before the split, none at the start
  of the record in early 2025. Power has to come from the energy register.
- `obis_1_8_0_f` is exactly 50 × `obis_1_8_0`, and the meter's `factor` field is a
  constant 50. So `obis_1_8_0_f` is the site-side register with the transformer
  factor already applied, and no further scaling is needed.
- The register moves in steps of 0.25 kWh. Differencing it at bucket edges
  quantises 15-minute mean power to **1,000 W**, five times the error budget.
- The site draws about 12–20 kW at night and 40–65 kW by day. The compressor
  cycles between about 0 and 3.3 kW on top of that.

### Approach

**Rebuilding energy from step timing (`features.py`).** Each register step lands
between two samples. Placing it at their midpoint and interpolating linearly
between steps gives energy at any instant, to well under one step. All powers are
read off that curve. On synthetic data shaped like this meter, it lowered the error
in site power from 301 W to 212 W median, against plain edge differencing.

**Per-bucket features**, computed by one function for both training and inference:

- 5- and 15-minute powers over the 2 h before the bucket and the 30 min after it
- the bucket's power relative to the minimum and median of that context
- step edges around the bucket
- outdoor temperature: the latest hourly value, and the trailing 24 h mean
- local time of day and day of week

Because the features look 30 minutes ahead, each estimate is published 30 minutes
after its bucket ends.

**Model (`training.py`).** Gradient-boosted trees (scikit-learn
`HistGradientBoostingRegressor`) with an absolute-error loss, since the criterion is
MAE. Training covers 381 days, back to the meter's first data, restricted to
heating-season buckets (local October–April). The last 31 days are held out and
scored as `val_mae`, then the model is refitted on everything.

**Operator (`op.py`).** It buffers main-meter and temperature messages, pruning the
buffers so they keep exactly what the training path would have seen. Once a
bucket's look-ahead has arrived, it emits `{"prediction": W}` stamped with the
bucket's start. On synthetic replays, features from the live path matched the
training path to within 1e-4 W.

**Every bucket gets an estimate.** Operator Lib's scorer only scores buckets that
have a prediction, so leaving out hard buckets would lower the number without
improving the operator. This operator emits all of them: 2,684 of the 2,688
February buckets. The 4 missing are the first partial bucket and the final
look-ahead.

### What was tried

All runs train on history before 2026-02-01 and replay February 2026.

| Run | Change | January hold-out MAE | February MAE |
|---|---|---|---|
| 1, 2 | Baseline: main meter only, full year (2 = same model + diagnostic) | 1,354 W | 962 W |
| 3 | Smaller, regularised model | 1,491 W | 1,096 W (worse; reverted) |
| 4 | Train on heating-season buckets only | 1,132 W | 845 W |
| 5 | + outdoor temperature | 882 W | 791 W |
| 6 | 6 h instead of 2 h context | 900 W | 805 W (worse; reverted) |
| 7, 8 | + second half of January 2025 (381-day window); 8 reproduces 7 | 877 W | **790.5 W** |

What helped:

- **Heating-season training**, because the summer months taught the model the
  wrong behaviour.
- **Outdoor temperature**, which lets the model tell a cold day from a mild one.

What did not help:

- **Less capacity:** the model was not overfitting.
- **More context:** 6 h of context fitted no better than 2 h.
- **More history:** adding last January barely changed the score.

### What limits it

A diagnostic in training correlates the 15-minute change in main-meter power with
the change in compressor power. On the January hold-out it peaks at lag 0, so the
two meters are aligned in time. But the peak is only 0.34 over all buckets, and
0.44 at night. Even at night, compressor steps account for only about a fifth of
how the main meter's power changes. The rest is other loads and the register's
resolution. By day the error is larger: 970 W by day against 695 W at night.

### What would change the outcome

- **Enabling OBIS 16.7.0** (instantaneous active power) in the Hauptzähler Tasmota
  meter script. This removes the 1 kW quantisation; it needs a few weeks of stored
  history before it can be trained on.
- **The EV socket as a runtime input**, if it will be available to the deployed
  operator. Its charging sessions draw about as much as the compressor. The code
  already accepts it as the input `ev_energy`.
- **A criterion relative to the baseline.** This is the developer's decision in
  `evaluation.yaml`.

### Inputs for a launch or deployment

| Operator input | Topic | Filter | Source |
|---|---|---|---|
| `main_energy` | `urn_infai_ses_service_9dc0f355-85ef-4ecd-99ca-924b582a46fe` | DeviceId `urn:infai:ses:device:22e2035c-0b19-42c8-8f12-2dd7c6ab111c` | `value.sensor.data.obis_1_8_0_f` |
| `temperature` | `urn_infai_ses_import_aac42d68-bc8e-cefe-de8e-be1ac252541c` | ImportId `urn:infai:ses:import:aac42d68-bc8e-cefe-de8e-be1ac252541c` | `value.temperature_2m` |
| `target` (training only) | `urn_infai_ses_service_b99a5d0d-9221-4d63-9071-1e70e36344a7` | DeviceId `urn:infai:ses:device:b17cd5e1-8527-47df-a7e3-fd71b4b756fe` | `value.root.egy_watt` |

The temperature import's history is read from its export (`df06780b-…`). Its live
feed may arrive with some delay in production. In the evaluation replay, each
hour's value was visible at that hour.

## Layout

| File | What it is |
|---|---|
| "main.py" | Entry point of the deployed operator. Hands the process to Operator Lib. |
| "train.py" | Entry point of an experiment. Trains through Operator Lib, then exits. |
| "op.py" | The operator: "infer", "train", "need_retraining", and its config. |
| "features.py" | The feature pipeline shared by training and inference. |
| "training.py" | The training pass and the model MLflow registers. |
| "pyproject.toml" | Dependencies, with Operator Lib pinned at "v1.9.0". |
| "uv.lock" | The resolved dependencies. Written by the scaffold; refresh it yourself. See below. |
| "Dockerfile" | The image. Built by CI; buildable by hand. |
| ".github/workflows/build.yml" | Builds and pushes "ghcr.io/senergy-platform/ode-ablation-nilm-l2". Change the registry here. |
| "operator.yaml" | What the analytics stack registers: inputs, outputs, config. |
| "evaluation.yaml" | Your criteria for whether a run is good, plus what Operator Lib needs to score a test window itself. ODE never writes this. |

## The lock file

The scaffold ran "uv lock" for you and "uv.lock" is in this working copy, uncommitted
like everything else here. Commit it with the rest.

Refresh it whenever you change a dependency in "pyproject.toml", and commit the two
together:

    uv lock

An experiment runs "uv run python train.py" on the cluster, and uv builds the
environment from "pyproject.toml" and this file — on the Ray head for the driver and
on each worker node for the tasks, out of its own cache.

Without a lock file uv resolves at run time, which works and is worse in one
specific way: the run records a commit SHA as the code that produced it, and two
runs of the same commit can then resolve different dependency versions. The lock
file is what makes the recorded SHA describe the whole run rather than only its
source. That is why it is not left to be remembered — and if the scaffold reported
that it could not write one, the command above is the repair.

## Building by hand

    docker build --build-arg GIT_COMMIT=$(git rev-parse HEAD) -t ghcr.io/senergy-platform/ode-ablation-nilm-l2:dev .

## The Operator Lib pin

"pyproject.toml" pins Operator Lib at "v1.9.0", the newest at the time
this repository was scaffolded. The library tracks latest and promises no
stability, so moving the pin is a deliberate edit — change it, run "uv lock", and
commit the two together.
