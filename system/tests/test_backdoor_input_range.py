import unittest
from types import SimpleNamespace

import torch

from flcore.security.backdoor import build_backdoor_attack


class BackdoorInputRangeTest(unittest.TestCase):
    def test_badnets_does_not_clip_normalized_clean_pixels(self):
        args = SimpleNamespace(
            backdoor="badnets",
            backdoor_target_label=0,
            backdoor_source_label=-1,
            backdoor_poison_ratio=0.25,
            backdoor_start_round=0,
            backdoor_end_round=-1,
            backdoor_patch_size=1,
            backdoor_patch_location="upper_left",
            backdoor_blend_alpha=0.2,
            backdoor_dba_fragments=4,
            backdoor_seed=0,
            backdoor_neurotoxin_mask_ratio=0.05,
            backdoor_f3ba_flip_ratio=0.01,
            backdoor_a3fl_step_size=0.01,
            backdoor_a3fl_shadow_steps=1,
        )
        attack = build_backdoor_attack(args, client_id=0, is_malicious=True)
        image = torch.tensor([[[-1.0, -0.5], [0.25, 0.75]]])

        triggered = attack.apply_trigger(image)

        self.assertEqual(float(triggered[0, 0, 0]), 1.0)
        self.assertTrue(torch.equal(triggered[0, 0, 1:], image[0, 0, 1:]))
        self.assertTrue(torch.equal(triggered[0, 1], image[0, 1]))


if __name__ == "__main__":
    unittest.main()
