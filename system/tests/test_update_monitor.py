import csv
import copy
import math
import os
import tempfile
import unittest

import torch
from torch import nn

from flcore.security.monitor import UpdateMonitor


class UpdateMonitorTest(unittest.TestCase):
    def test_records_update_norms_and_model_to_global_cosine(self):
        global_model = nn.Linear(2, 1, bias=False)
        with torch.no_grad():
            global_model.weight.copy_(torch.tensor([[1.0, 2.0]]))
        benign = copy.deepcopy(global_model)
        malicious = copy.deepcopy(global_model)
        with torch.no_grad():
            benign.weight.copy_(torch.tensor([[2.0, 2.0]]))
            malicious.weight.copy_(torch.tensor([[1.0, 4.0]]))

        with tempfile.TemporaryDirectory() as output_dir:
            monitor = UpdateMonitor(output_dir, plot_every=10)
            monitor.record(
                round_id=0,
                global_model=global_model,
                uploaded_ids=[0, 1],
                uploaded_models=[benign, malicious],
                malicious_ids=[1],
            )
            with open(monitor.client_csv, newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))

        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["group"], "benign")
        self.assertEqual(rows[1]["group"], "malicious")
        self.assertEqual(int(rows[0]["l0"]), 1)
        self.assertAlmostEqual(float(rows[0]["l2"]), 1.0)
        self.assertAlmostEqual(float(rows[0]["linf"]), 1.0)
        expected_cosine = 6.0 / math.sqrt(40.0)
        self.assertAlmostEqual(
            float(rows[0]["cosine_to_global"]), expected_cosine, places=6)


if __name__ == "__main__":
    unittest.main()
