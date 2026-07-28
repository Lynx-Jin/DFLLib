# Min-Max and Min-Sum model poisoning

The attacks run centrally in `Server.receive_models()` after active clients
finish clean local training and before aggregation. Only active malicious
clients' clean model updates are used as proxy benign updates. The optimized
malicious update is copied to every active malicious client.

## Experiment logger

Every invocation creates one self-contained directory under `system/logger`.
The name contains the start time and core hyperparameters, for example:

```text
logger/20260715_173000_000000_MNIST_FedAvg_CNN_nc20_mc5_jr1_r100_le1_atk-minmax_dev-std_seed0/
```

Artifacts are organized as:

```text
<experiment>/
├── terminal_output.txt
├── config.json
├── items/run0/
├── models/
├── results/
└── attack_monitor/run0/        # only contains data when -amon is enabled
    ├── client_metrics.csv
    ├── group_summary.csv
    └── benign_vs_malicious.png
```

All stdout and stderr are simultaneously displayed in the terminal and appended
to `terminal_output.txt`; an external `tee` command is not needed. Use
`-logroot PATH` to select a different logger root.

## Usage

Run from the `system` directory. This command runs exactly 100 FedAvg rounds in
this version of PFLlib (`serveravg.py` iterates from 0 through `global_rounds`):

```bash
python main.py -algo FedAvg -data MNIST -ncl 10 -m CNN \
  -nc 20 -jr 1 -gr 99 -ls 1 \
  -atk minmax -amr 0.25 -adev std -amon
```

Choose `-atk minmax` or `-atk minsum`. Add `-amon` to enable update monitoring;
omit it to disable monitoring.

Attack options:

- `-amr`: fraction of globally compromised clients.
- `-aseed`: malicious-client selection seed.
- `-adev`: `unit_vec`, `sign`, or `std` perturbation.
- `-alinit`: initial lambda search bound.
- `-atol`: lambda search tolerance.
- `-amaxsteps`: maximum search iterations.

Min-Sum follows the released NDSS implementation and uses the minimum proxy
sum-distance as its threshold. This differs from the `max` printed in Equation
(7) of the paper.

## Monitoring

The monitor runs after poisoning and before aggregation. It records:

- L0, L2, and L-infinity norms of `client_model - previous_global_model`;
- cosine similarity between the uploaded model and previous global model;
- client id and benign/malicious identity.

Options:

```text
-amonplot N     regenerate the plot every N rounds (default: 10)
-amoneps EPS    L0 counts coordinates with absolute update > EPS
-amondir NAME   monitor subdirectory name inside the experiment directory
```

## Tests

With the project environment active:

```bash
python -m unittest tests.test_minmax_minsum tests.test_update_monitor -v
```
