# Scalar security port validation

Validated on 2026-11-05 against base revision `5541c85`. These are migration
checks, not evidence of broad attack effectiveness or defense accuracy.

- **41 new portable tests pass:** 35 source-core/geometry regressions and six
  real client/receive/aggregation/validation checks (with multiple subcases).
- **Direct reference parity:** all upload tensors and attack statistics match
  exactly in 12 tailored Min-Max fixtures (two variants, three directions, two
  seeds). PoisonedFL updates/statistics match over 55 observations for each of
  feedback windows 1, 10 and 50.
- **Six actual main.py smoke cases complete:** clean+DBSCAN; PoisonedFL without
  filtering and with DBSCAN; tailored Min-Max without filtering and with DBSCAN;
  and FedAvg PoisonedFL+DBSCAN. Other cases use native FedSGD/Adam.
- Smoke cases use the existing full CIFAR-10 pools (20 train/test partitions,
  45,000/15,000 available samples), `-fs 0`, fixed total 20 and two aggregation
  steps each. FedSGD still samples one native batch100 per client per round;
  FedAvg uses one local epoch. PF attackers skip private training. No datasets
  or generated experiment artifacts are committed.
- Each case has two finite 20-client scalar records and matching ordinary
  audit rows. Attack cases identify exactly four malicious clients, with no
  appended clients. DBSCAN smoke cases observe warm-up followed by activation.
- Unit integration checks specifically cover all three PF upload mappings,
  no private training by PF attackers, label-independent majority filtering,
  sample reweighting, stable client IDs and preserving both model and Adam step
  state after an all-rejected round.

Run the portable new checks from `system/`:

```bash
python -m unittest discover -s tests -p 'test_port_*.py' -v
```

The full suite reports 45 tests with exactly the same two legacy failures
observed before migration. No additional failures were introduced:

1. `test_minsum_uses_minimum_proxy_sum_score`: current baseline Min-Sum uses a
   max reference while the older test expects min.
2. `test_records_update_norms_and_model_to_global_cosine`: older monitor test
   omits the now-required upload weights and expects a previous cosine reference.

Those legacy numerical behaviors and tests are deliberately preserved. The
41-test command is scoped to the port and is not presented as a green full suite.

Raw local verification output is under `system/logger/port-validation/`, which
is ignored by Git. Use the [method and command guide](scalar-security-port.md)
to run new data/model/budget settings; no long-run benchmark claim is made here.
