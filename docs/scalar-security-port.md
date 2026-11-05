# DFLlib: PoisonedFL, cosine-tailored Min-Max, and scalar defenses

The new options extend `flcore/security`. Existing AGR-agnostic Min-Max/Min-Sum,
Fang and aggregators retain their numerical implementations. New methods support
FedAvg and the existing one-minibatch FedSGD. This port does not replace DFLlib's
sampling, architecture, sample weights or server SGD/Adam optimizers.

## Population and information boundaries

`-nc 20 -amr .2 -aseed 0` means **20 total clients**, with malicious IDs
`[1,8,18,19]`. No clients are appended. Ratios .1/.3/.4/.5 give 2/6/8/10
malicious clients. Other population sizes work with matching data partitions.

- PoisonedFL reads broadcast global models and its own previous attempted upload.
  Malicious clients skip local training. No private data, benign updates, defense
  decisions or optimizer state enter its constructor. Evaluation is independent
  of attack construction and still uses the configured existing partitions.
- Tailored Min-Max uses active malicious clients' clean updates as proxies. It
  knows the submitted/global cosine criterion, not KMeans or DBSCAN decisions.
  `cosine_oracle` describes feasibility, **not access to true benign updates**.
- The detector backend receives only ID-keyed norm/cosine maps and a round index.
  IDs track history; neither malicious IDs nor `-amr` enter detection. Truth is
  available only to the existing offline audit monitor.

## PoisonedFL

Use `-atk poisonedfl`. All active malicious clients submit an identical payload.
A coordinate sign vector stays fixed across rounds, while coordinate magnitudes
adapt from observed global steps. Interval feedback measures sign alignment of
cumulative global movement; insufficient alignment can multiply the scale by
.7, without crossing a candidate floor of .5. The first observation has no
history and submits zero update.

| Option | Default | Meaning |
|---|---:|---|
| `-pfs` | 8 | Initial scale |
| `-pfw` | 10 | Feedback interval; use 50 for the authors' setting |
| `-pfgm` | `sgd_delta` | Explicit FedSGD gradient adapter |

FedAvg submits `W+m`. Plain server SGD uses `g_m=-m/server_lr`, in the native
`W+g_m` carrier; zero server momentum and weight decay are required.

Server Adam requires **`-pfgm adam_proxy`**: the history-scaled descent vector q
becomes gradient proxy `g_m=-q`. The adapter does not divide by the scalar LR
or inspect Adam state. Parameter-step magnitudes are only a proxy for gradient
magnitudes, and Adam's resulting model delta does **not** equal q. Logs mark
`model_delta_equivalent=false`. This is not a step-equivalent reproduction of
the original model-delta algorithm.

This is the basic shared-payload attack, **not PoisonedFL-Adapt**. References:
[Xie et al., CVPR 2025](https://arxiv.org/abs/2404.15611) and the
[author code](https://github.com/xyq7/PoisonedFL/tree/266488e2cbe5953aab61712f315518546f457e55).
Published results do not establish effectiveness under this library's Adam proxy,
population, sampling or scalar detector.

## Cosine-tailored Min-Max

Keep `-atk minmax` and select `-amv`:

| Value | Feasibility rule |
|---|---|
| `agnostic` | Existing DFLlib Min-Max, unchanged |
| `cosine` | Cosine constraint with the agnostic L2 search cap |
| `cosine_oracle` | Signed cosine replaces the L2 constraint |

The ray is `proxy_mean-lambda*deviation`. The oracle maximizes lambda over
`[0,2*alinit]` subject to `cos(W+proxy_mean-lambda*deviation,W)>=target`.
`-act` sets an explicit target; otherwise `-acq` (default .5) selects a quantile
of malicious clean-proxy model/global cosines. `-acs` (256) controls the
conservative `cosine` grid. The oracle solves boundary intervals analytically,
checks signed cosine, handles feasible intervals not containing zero, and
reports unreachable targets. It may exceed the agnostic L2 bound.

These are cosine-aware project adaptations within the SH21 fixed-direction
framework, not the original AGR-agnostic threat model or an exact SH21 AFA oracle.

## Scalar defenses

Use `-def scalar_kmeans` for stage one alone; `-def scalar_dbscan` adds stage two.
They are distinct from existing `cosine_cluster`:

| Feature/method | Definition |
|---|---|
| Existing cosine clustering | `cos(update_i, weighted_mean_update)` |
| New submitted norm | `||P_i||_2` |
| New submitted cosine | `cos(P_i,W)` |

W is the global model broadcast at the current round's start. P is the submitted
local model for FedAvg, or native **W+g gradient carrier** for FedSGD. Features
use raw, unshifted parameter tensors, excluding buffers. A plaintext bridge
extracts features one model at a time; only the scalar maps reach the backend.

1. Deterministic 1-D KMeans on cosine keeps the larger cluster. Ties prefer the
   higher center; coincident cosines give one effective cluster. This rule also
   operates on clean runs and is not an attack-presence test.
2. Only main-cluster members enter DBSCAN. Norm/cosine differences use MAD-derived
   widths with explicit precision floors and Chebyshev distance. Noise is kept.
   A component is suspicious only if its entire normalized diameter is small;
   connected chains alone are not considered identical groups.
3. A current candidate requires sufficient client and same-partner persistence.
   Histories advance only for stage-two eligible observations: absence or stage-one
   exclusion does not append a negative or erase history. Eligible non-candidates
   do append negatives.

| Option | Default | Meaning |
|---|---:|---|
| `-defstart` | 0 | First round applying rejection |
| `-schr` / `-schc` | .001 / .001 | MAD-normalized norm/cosine widths |
| `-scnorm` / `-sccos` | 1e-6 / 1e-6 | Relative norm / absolute cosine floors |
| `-scmin` / `-scbase` | 3 / 1 | Size threshold is max(scmin,scbase+1) |
| `-sde` / `-sdd` | 1 / 1 | DBSCAN eps / full-component diameter limit |
| `-schist` / `-scobs` | 10 / 3 | Eligible-history length / minimum observations |
| `-scfreq` | .8 | Minimum client/pair candidate frequency |
| `-sclog` | off | Print cosine-sorted per-client submitted norms/cosines |

Scalar histories update during warm-up, with no rejection before defstart, as
in the migrated experiment. Other defenses retain their existing warm-up.
Kept clients retain relative sample weights, then weights are renormalized.
All-rejected rounds keep global parameters/buffers and do not advance Adam.
This engineering composition is not an exact FLAME/HDBSCAN implementation.

## Commands

Run from `system/` after preparing matching local partitions. `-fs 0` uses full
local pools; native FedSGD still samples one minibatch per participating client.
Examples use Adam .001, batch100, fixed20. Existing `-gr 30` means indices0..30,
i.e.31 aggregations. Match round budgets and defenses in clean controls.

```bash
# Clean control with the same defense.
python main.py -data Cifar10 -m CNN -algo FedSGD -fopt adam -flr .001 -lbs 100 -nc 20 -jr 1 -gr 30 -fs 0 -atk none -amr 0 -def scalar_dbscan -defstart 5 -sclog

# PoisonedFL attack-only and two defenses.
python main.py -data Cifar10 -m CNN -algo FedSGD -fopt adam -flr .001 -lbs 100 -nc 20 -jr 1 -gr 30 -fs 0 -atk poisonedfl -amr .2 -aseed 0 -pfs 8 -pfw 10 -pfgm adam_proxy -def none
python main.py -data Cifar10 -m CNN -algo FedSGD -fopt adam -flr .001 -lbs 100 -nc 20 -jr 1 -gr 30 -fs 0 -atk poisonedfl -amr .2 -aseed 0 -pfs 8 -pfw 10 -pfgm adam_proxy -def scalar_kmeans -defstart 5 -sclog
python main.py -data Cifar10 -m CNN -algo FedSGD -fopt adam -flr .001 -lbs 100 -nc 20 -jr 1 -gr 30 -fs 0 -atk poisonedfl -amr .2 -aseed 0 -pfs 8 -pfw 10 -pfgm adam_proxy -def scalar_dbscan -defstart 5 -sclog

# Tailored Min-Max without and with filtering.
python main.py -data Cifar10 -m CNN -algo FedSGD -fopt adam -flr .001 -lbs 100 -nc 20 -jr 1 -gr 30 -fs 0 -atk minmax -amr .2 -aseed 0 -adev sign -amv cosine_oracle -acq .5 -def none
python main.py -data Cifar10 -m CNN -algo FedSGD -fopt adam -flr .001 -lbs 100 -nc 20 -jr 1 -gr 30 -fs 0 -atk minmax -amr .2 -aseed 0 -adev sign -amv cosine_oracle -acq .5 -def scalar_dbscan -defstart 5 -sclog
```

FedAvg PF uses `-algo FedAvg -pfgm sgd_delta` (a model delta, no gradient
conversion). Source-project `-omfm dbscan` maps to `-def scalar_dbscan`, and
`-omfsr` to `-defstart`. K=2/main scope are fixed; source-only `-sck`, `-scscope`,
`-pfn` and `-fgm` are not DFLlib arguments. DFLlib sampling and native features
are retained, so cross-framework accuracy curves need not be bit-identical.

## Audit and tests

Each existing experiment directory contains:

- `security/runN/attack_trace.jsonl`: tailored feasibility/lambda/cosine, or PF
  feedback/scale/upload mapping.
- `security/runN/submitted_scalars.jsonl`: pre-defense submitted norm/cosine;
  scalar runs include clusters, candidates, histories, both rejection stages
  and final IDs.
- Existing `attack_monitor/runN/round_audit.csv`: offline truth and final decisions
  for detection metrics. Truth never enters the detector.

PF and tailored Min-Max automatically enable ordinary audit monitoring even
without defense. Separate scalar JSONL preserves the legacy CSV schema and
its distinct mean-update reference.

```bash
python -m unittest discover -s tests -p 'test_port_*.py' -v
```

The tests cover core semantics, signed feasibility, identical payloads, feedback
indexing, eligible histories/real IDs, actual clients/receive/aggregation, all
three PF mappings, private-training exclusion, weights, warm-up and Adam hold.

Base revision `5541c85` already has two failing legacy tests:
`test_minsum_uses_minimum_proxy_sum_score` expects min while its implementation
uses max; `test_records_update_norms_and_model_to_global_cosine` uses an outdated
monitor signature/reference. This port records these as pre-existing rather
than silently altering legacy methods/tests.

`port_reference_sha256.json` records source-core fingerprints. Unit/integration
tests and short smoke runs validate migration, not published-table reproduction
or broad attack/defense effectiveness.

See the [recorded migration validation](scalar-port-validation.md) for test
counts, exact-reference checks, real main.py smoke coverage and baseline failures.
