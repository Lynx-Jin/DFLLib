import unittest

from flcore.security.defense.scalar_concentration import ScalarConcentrationFilter


class ScalarConcentrationTest(unittest.TestCase):
    def test_middle_duplicate_group_and_both_benign_sides_are_retained(self):
        values = [.811, .820, .833, .845, .845, .845, .877, .881, .883, .890]
        cosines = dict(enumerate(values))
        norms = {i: 100.0 + i for i in range(10)}
        for cid in (3, 4, 5):
            norms[cid] = 110.0
        detector = ScalarConcentrationFilter(k=3)
        for round_index in range(3):
            result = detector.apply(norms, cosines, round_index)
        self.assertEqual(result['confirmed_ids'], [3, 4, 5])
        self.assertEqual(set(cosines) - set(result['confirmed_ids']), {0, 1, 2, 6, 7, 8, 9})

    def test_same_cosine_different_norms_is_not_a_duplicate(self):
        detector = ScalarConcentrationFilter()
        for round_index in range(5):
            result = detector.apply({i: 100.0 + i for i in range(6)}, {i: .95 for i in range(6)}, round_index)
        self.assertEqual(result['max_window_count'], 1)
        self.assertEqual(result['confirmed_ids'], [])

    def test_precision_floor_absorbs_copied_float32_cosine_spread(self):
        detector = ScalarConcentrationFilter(min_history=1)
        result = detector.apply(
            {0: 100.0, 1: 100.000001, 2: 100.000002, 3: 120.0},
            {0: .99999, 1: .99999006, 2: .99999012, 3: .9998}, 0)
        self.assertEqual(result['confirmed_ids'], [0, 1, 2])

    def test_absence_does_not_dilute_participation_history(self):
        detector = ScalarConcentrationFilter()
        norms = {0: 100.0, 1: 100.0, 2: 100.0, 3: 130.0}
        cosines = {0: .95, 1: .95, 2: .95, 3: .8}
        detector.apply(norms, cosines, 0)
        for round_index in range(1, 8):
            detector.apply({3: 130.0, 4: 140.0}, {3: .8, 4: .7}, round_index)
        self.assertEqual(detector.apply(norms, cosines, 8)['confirmed_ids'], [])
        result = detector.apply(norms, cosines, 9)
        self.assertEqual(result['confirmed_ids'], [0, 1, 2])
        self.assertEqual(result['history_scores'][0], 1.0)

    def test_repeated_client_without_repeated_partner_is_not_confirmed(self):
        detector = ScalarConcentrationFilter(k=3, min_group=2)
        for round_index in range(3):
            norms = {i: 120.0 + 10 * i for i in range(5)}
            cosines = {i: .7 + .04 * i for i in range(5)}
            norms[0] = norms[round_index + 1] = 100.0
            cosines[0] = cosines[round_index + 1] = .99
            result = detector.apply(norms, cosines, round_index)
        self.assertIn(0, result['candidate_ids'])
        self.assertEqual(result['confirmed_ids'], [])

    def test_calibrated_count_is_strictly_exceeded(self):
        detector = ScalarConcentrationFilter(clean_max_count=3, min_history=1)
        result = detector.apply({0: 1.0, 1: 1.0, 2: 1.0, 3: 2.0},
                                {0: .9, 1: .9, 2: .9, 3: .7}, 0)
        self.assertEqual(result['max_window_count'], 3)
        self.assertEqual(result['required_count'], 4)
        self.assertEqual(result['candidate_ids'], [])

    def test_scans_all_separate_duplicate_groups(self):
        detector = ScalarConcentrationFilter(k=3, min_history=1)
        result = detector.apply({0: 1., 1: 1., 2: 1., 3: 2., 4: 2., 5: 2., 6: 3.},
                                {0: .8, 1: .8, 2: .8, 3: .9, 4: .9, 5: .9, 6: .7}, 0)
        self.assertEqual(result['confirmed_ids'], [0, 1, 2, 3, 4, 5])

    def test_singletons_are_never_suspicious(self):
        result = ScalarConcentrationFilter(min_history=1).apply({5: 1.0}, {5: .9}, 0)
        self.assertEqual(result['max_window_count'], 1)
        self.assertEqual(result['confirmed_ids'], [])

    def test_main_scope_can_miss_a_small_attack_cluster(self):
        norms = {i: 1.0 + i for i in range(12)}
        cosines = {i: .8 + .001 * i for i in range(12)}
        for cid in range(9, 12):
            norms[cid], cosines[cid] = 20.0, .99
        all_result = ScalarConcentrationFilter(k=3, min_history=1).apply(norms, cosines, 0)
        main_result = ScalarConcentrationFilter(k=3, scope='main', min_history=1).apply(norms, cosines, 0)
        self.assertEqual(all_result['confirmed_ids'], [9, 10, 11])
        self.assertEqual(main_result['confirmed_ids'], [])

    def test_k2_discards_small_cluster_even_when_scope_all_is_requested(self):
        result = ScalarConcentrationFilter(k=2, scope='all').apply(
            {i: 100.0 + i for i in range(7)},
            {0: .90, 1: .91, 2: .92, 3: .93, 4: .94, 5: .2, 6: .21}, 0)
        self.assertEqual(result['main_ids'], [0, 1, 2, 3, 4])
        self.assertEqual(result['cluster_rejected_ids'], [5, 6])
        self.assertEqual(result['rejected_ids'], [5, 6])
        self.assertEqual(result['window_confirmed_ids'], [])
        self.assertEqual(result['scope'], 'main')

    def test_k2_equal_size_prefers_higher_cosine_center(self):
        result = ScalarConcentrationFilter(k=2).apply(
            {0: 1., 1: 2., 2: 3., 3: 4.},
            {0: .2, 1: .21, 2: .9, 3: .91}, 0)
        self.assertEqual(result['main_ids'], [2, 3])
        self.assertEqual(result['cluster_rejected_ids'], [0, 1])


if __name__ == '__main__':
    unittest.main()
