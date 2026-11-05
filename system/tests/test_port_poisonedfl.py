import copy
import unittest

import numpy as np
import torch
from scipy.stats import binom

from flcore.security.attack.poisonedfl import PoisonedFLAttack, flatten_model, DEFAULT_POISONEDFL_WINDOW


class PoisonedFLTest(unittest.TestCase):
    def test_author_binomial_thresholds(self):
        for dimension, expected in [(1204682, 603618), (139960, 70415),
                                    (717924, 359948), (145212, 73049)]:
            self.assertEqual(int(binom.ppf(.99, dimension, .5)), expected)

    def test_numpy_port_equivalence_and_feedback_anchor(self):
        # Independent transcription of author's MXNet arithmetic and post-round
        # checkpoint bookkeeping, on a prescribed sequence of observed globals.
        rng = np.random.RandomState(123)
        models = np.cumsum(rng.normal(size=(104, 43)).astype(np.float32) * .1, axis=0)
        attack = PoisonedFLAttack(seed=7, window=50)
        zero, stats = attack.propose(torch.from_numpy(models[0]))
        self.assertEqual(float(zero.abs().sum()), 0.)
        self.assertEqual(stats['phase'], 'bootstrap')
        sign = attack.fixed_sign.numpy().copy()
        last = np.zeros(43, dtype=np.float32)
        scale = 8.
        anchor = models[1]
        for t in range(1, len(models)):
            if (t - 1) % 50 == 0:
                anchor = models[t].copy()
            if t % 50 == 0:
                total = models[t] - anchor
                total = np.where(total == 0, models[t], total)
                count = int(np.sum(np.sign(total) == sign))
                if count < int(binom.ppf(.99, 43, .5)) and scale * .7 >= .5:
                    scale *= .7
            history = models[t] - models[t - 1]
            hnorm = np.linalg.norm(history)
            coordinate_scale = np.abs(history - last * hnorm / (np.linalg.norm(last) + 1e-9))
            expected = scale * hnorm * coordinate_scale * sign / (np.linalg.norm(coordinate_scale) + 1e-9)
            actual, stats = attack.propose(torch.from_numpy(models[t]))
            np.testing.assert_allclose(actual.numpy(), expected, rtol=3e-6, atol=2e-6)
            self.assertEqual(stats['feedback_checked'], t % 50 == 0)
            self.assertAlmostEqual(stats['scale'], scale)
            last = expected
        np.testing.assert_array_equal(attack.fixed_sign.numpy(), sign)

    def test_default_ten_round_feedback_checks_and_decay(self):
        attack = PoisonedFLAttack(seed=7)
        self.assertEqual(DEFAULT_POISONEDFL_WINDOW, 10)
        self.assertEqual(attack.window, 10)
        attack.propose(torch.zeros(64))
        sign = attack.fixed_sign.clone()
        checked = []
        for t in range(1, 30):
            # Prescribed observed trajectory opposes the fixed attack signs,
            # making every significance check fail without using benign data.
            _, stats = attack.propose(-sign * (.01 * t))
            if stats['feedback_checked']:
                checked.append(t)
                self.assertTrue(stats['scale_adjusted'])
                self.assertEqual(stats['aligned_dimensions'], 0)
                torch.testing.assert_close(attack.window_anchor, -sign * (.01 * (t - 9)))
            expected_scale = 8. * (.7 ** (t // 10))
            self.assertAlmostEqual(stats['scale'], expected_scale)
        self.assertEqual(checked, [10, 20])
        self.assertEqual(attack.round, 30)
        self.assertAlmostEqual(attack.scale, 3.92)
        torch.testing.assert_close(attack.window_anchor, -sign * .21)

    def test_same_payload_and_only_controlled_clients_changed(self):
        torch.manual_seed(2)
        model = torch.nn.Linear(4, 3)
        clients = [copy.deepcopy(model) for _ in range(4)]
        attack = PoisonedFLAttack()
        attack.apply(model, [0, 1, 2, 3], clients, [1, 3])
        with torch.no_grad():
            model.weight.add_(.1)
        innocent = flatten_model(clients[0]).clone()
        stats = attack.apply(model, [0, 1, 2, 3], clients, [1, 3])
        self.assertTrue(torch.equal(flatten_model(clients[1]), flatten_model(clients[3])))
        self.assertTrue(torch.equal(innocent, flatten_model(clients[0])))
        self.assertGreater(stats['update_norm'], 0.)
        submitted_update = flatten_model(clients[1]) - flatten_model(model)
        active = submitted_update.abs() > 1e-6
        self.assertTrue(torch.equal(submitted_update[active].sign(), attack.fixed_sign[active]))

    def test_window_one_uses_actual_previous_step_not_current_model_signs(self):
        # W remains sign-aligned while the first actual update is opposite.
        # A current-model anchor would see a zero difference, invoke the zero
        # fallback, and incorrectly report all 64 coordinates as aligned.
        seed = 7
        generator = torch.Generator().manual_seed(seed)
        sign = (2*torch.randint(0, 2, (64,), generator=generator)-1).float()
        attack = PoisonedFLAttack(window=1, seed=seed)
        attack.propose(2*sign)
        _, first = attack.propose(1.9*sign)
        self.assertTrue(first['feedback_checked'])
        self.assertEqual(first['aligned_dimensions'], 0)
        self.assertTrue(first['scale_adjusted'])
        self.assertAlmostEqual(first['scale'], 5.6)
        torch.testing.assert_close(attack.window_anchor, 2*sign)
        _, second = attack.propose(2.1*sign)
        self.assertEqual(second['aligned_dimensions'], 64)
        self.assertFalse(second['scale_adjusted'])
        self.assertAlmostEqual(second['scale'], 5.6)
        torch.testing.assert_close(attack.window_anchor, 1.9*sign)

    def test_absent_fake_does_not_replace_previous_attempt(self):
        attack = PoisonedFLAttack()
        attack.propose(torch.zeros(100))
        update, _ = attack.propose(torch.ones(100))
        attack.propose(torch.ones(100) * 2, active=False)
        self.assertTrue(torch.equal(attack.last_malicious_update, update))

    def test_attack_randomness_isolated_and_no_history_stays_zero(self):
        torch.manual_seed(38)
        state = torch.random.get_rng_state().clone()
        attack = PoisonedFLAttack()
        attack.propose(torch.ones(20))
        update, stats = attack.propose(torch.ones(20))
        self.assertTrue(torch.equal(state, torch.random.get_rng_state()))
        self.assertEqual(float(update.abs().sum()), 0.)
        self.assertEqual(stats['history_norm'], 0.)


if __name__ == '__main__':
    unittest.main()
