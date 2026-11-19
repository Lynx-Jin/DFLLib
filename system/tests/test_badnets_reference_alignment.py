"""Independent golden vectors from the supplied reference BadNet implementation."""
import hashlib
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
from types import SimpleNamespace
import unittest

import numpy as np
import torch

from flcore.security.backdoor import build_backdoor_attack


class ReferenceBadnetsTest(unittest.TestCase):
    def test_default_recipe_and_reference_index_vectors(self):
        pixel = torch.zeros(3, 32, 32)
        records = [(pixel, torch.tensor(int(label))) for label in np.tile(np.arange(10), 225)]
        # Generated independently with the supplied utils/backdoor.py, SHA256:
        # 60aa667680b7bed9386db24001a7f1a97266147d7da89242f63fbaae128685d7
        expected = {
            1: 'f6ce010032713374a8ee957fc4d5c217d4bab566caabcdea3b81affb7bf1bedf',
            8: 'ae849791a939cb3f01894b1fd24792e7148841f8975f1be48a400fff8293aebf',
            18: 'eb357de4daf841ace29df2131a94f8d551dfeeccf6c8c19c34ee3b2f2b4782a3',
            19: '87f2a853a1dda88d9a1b6c02351256718775dd2bf818ffd9b66a95131b2a874b',
        }
        for cid, digest in expected.items():
            attack = build_backdoor_attack(SimpleNamespace(backdoor='badnets'), cid, True)
            self.assertEqual((attack.spec.preset, attack.spec.poison_ratio,
                              attack.spec.patch_size, attack.spec.patch_location,
                              attack.spec.patch_margin), ('reference', .2, 4, 'lower_right', 1))
            selected = attack.choose_poison_indices(records, 0)
            self.assertEqual(len(selected), 450)
            self.assertTrue(all(index % 10 != 0 for index in selected))
            self.assertEqual(hashlib.sha256(np.asarray(sorted(selected), dtype=np.int64).tobytes()).hexdigest(), digest)
            for round_id in [1, 20, 100, 200]:
                self.assertEqual(selected, attack.choose_poison_indices(records, round_id))

    def test_reference_trigger_pixels_and_input_preservation(self):
        attack = build_backdoor_attack(SimpleNamespace(backdoor='badnets'), 1, True)
        images = torch.full((2, 3, 32, 32), -.5)
        before = images.clone()
        actual = attack.apply_trigger(images)
        expected = before.clone()
        expected[:, :, 27:31, 27:31] = 1
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        torch.testing.assert_close(images, before, rtol=0, atol=0)
        torch.testing.assert_close(attack.apply_trigger(images[0]), expected[0], rtol=0, atol=0)
        self.assertEqual((actual != before).flatten(1).sum(1).tolist(), [48, 48])

    def test_all_local_denominator_caps_at_non_target_count(self):
        attack = build_backdoor_attack(SimpleNamespace(backdoor='badnets',
                                       backdoor_poison_ratio=.2), 1, True)
        records = [(None, torch.tensor(0))] * 9 + [(None, torch.tensor(1))]
        self.assertEqual(attack.choose_poison_indices(records, 0), {9})
        attack = build_backdoor_attack(SimpleNamespace(backdoor='badnets',
                                       backdoor_poison_ratio=0), 1, True)
        self.assertEqual(attack.choose_poison_indices(records, 0), set())

    @unittest.skipUnless(shutil.which('bash'), 'launcher validation requires bash')
    def test_single_teacher_launcher_uses_explicit_reference_or_legacy(self):
        script = Path(__file__).resolve().parents[1] / 'tools/reproduce_single_teacher_datafree.sh'
        for preset, values in [('reference', ['0.2', '4', 'lower_right', '1']),
                               ('legacy', ['0.1', '5', 'upper_left', '0'])]:
            for mode in ['badnets', 'clean']:
                env = dict(os.environ, DFKD_DRY_RUN='1', DFKD_PYTHON=sys.executable,
                           DFKD_BACKDOOR_PRESET=preset)
                command = shlex.split(subprocess.check_output(['bash', str(script), mode],
                                                               env=env, text=True))
                self.assertEqual(command[command.index('--backdoor-preset') + 1], preset)
                expected = list(values)
                if mode == 'clean':
                    expected[0] = '0'
                actual = [command[command.index(flag) + 1] for flag in
                          ['--poison-ratio', '--patch-size', '--patch-location', '--patch-margin']]
                self.assertEqual(actual, expected)

    def test_legacy_is_explicit_and_retains_old_sampling_and_geometry(self):
        records = [(None, torch.tensor(int(label))) for label in np.tile(np.arange(10), 225)]
        attack = build_backdoor_attack(SimpleNamespace(backdoor='badnets',
                                       backdoor_preset='legacy', backdoor_poison_ratio=.2), 1, True)
        self.assertEqual(len(attack.choose_poison_indices(records, 0)), 405)
        self.assertNotEqual(attack.choose_poison_indices(records, 0),
                            attack.choose_poison_indices(records, 1))
        self.assertEqual((attack.spec.patch_size, attack.spec.patch_location,
                          attack.spec.patch_margin), (5, 'upper_left', 0))


if __name__ == '__main__':
    unittest.main()
