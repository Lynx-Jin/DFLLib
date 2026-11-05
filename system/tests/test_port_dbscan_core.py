import unittest

from flcore.security.defense.scalar_dbscan import ScalarDBSCANFilter
from flcore.security.defense.scalar_concentration import ScalarConcentrationFilter


class ScalarDBSCANTest(unittest.TestCase):
    def test_majority_then_persistent_compact_group(self):
        norms = {0: 100., 1: 100., 2: 100., 3: 110., 4: 120., 5: 130., 6: 140., 7: 150.}
        cosines = {0: .95, 1: .95, 2: .95, 3: .949, 4: .951, 5: .952, 6: .2, 7: .21}
        detector = ScalarDBSCANFilter()
        for index in range(3):
            result = detector.apply(norms, cosines, index)
            self.assertEqual(result['cluster_rejected_ids'], [6, 7])
            self.assertEqual(result['confirmed_ids'], [] if index < 2 else [0, 1, 2])
        self.assertEqual(result['dbscan_noise_ids'], [3, 4, 5])
        self.assertEqual(result['dbscan_confirmed_ids'], [0, 1, 2])
        self.assertEqual(result['rejected_ids'], [0, 1, 2, 6, 7])
        self.assertNotIn('window_confirmed_ids', result)

    def test_cosine_match_alone_is_not_sufficient(self):
        detector = ScalarDBSCANFilter(min_history=1)
        result = detector.apply({i: 100. + i for i in range(6)}, {i: .95 for i in range(6)}, 0)
        self.assertEqual(result['groups'], [])
        self.assertEqual(result['dbscan_noise_ids'], list(range(6)))
        self.assertEqual(result['rejected_ids'], [])

    def test_dbscan_connected_chain_is_retained(self):
        detector = ScalarDBSCANFilter(min_group=2, min_history=1, norm_rtol=.01)
        result = detector.apply(dict(enumerate([100., 100.75, 101.5, 102.25])),
                                {i: .95 for i in range(4)}, 0)
        self.assertEqual(len(result['dbscan_clusters']), 1)
        self.assertGreater(result['dbscan_clusters'][0]['diameter'], 1.)
        self.assertFalse(result['dbscan_clusters'][0]['candidate'])
        self.assertEqual(result['confirmed_ids'], [])

    def test_multiple_compact_groups_and_background(self):
        result = ScalarDBSCANFilter(min_history=1).apply(
            dict(enumerate([100., 100., 100., 120., 120., 120., 140.])),
            {i: .95 for i in range(7)}, 0)
        self.assertEqual(result['groups'], [[0, 1, 2], [3, 4, 5]])
        self.assertEqual(result['confirmed_ids'], list(range(6)))
        self.assertEqual(result['dbscan_noise_ids'], [6])

    def test_numeric_floors_allow_copied_float32_spread(self):
        result = ScalarDBSCANFilter(min_history=1).apply(
            {0: 100., 1: 100.000001, 2: 100.000002, 3: 130.},
            {0: .95, 1: .95000006, 2: .95000012, 3: .951}, 0)
        self.assertEqual(result['confirmed_ids'], [0, 1, 2])

    def test_clean_occupancy_threshold_is_strictly_exceeded(self):
        result = ScalarDBSCANFilter(clean_max_count=3, min_history=1).apply(
            {0: 1., 1: 1., 2: 1., 3: 2.}, {i: .95 for i in range(4)}, 0)
        self.assertEqual(result['dbscan_min_samples'], 4)
        self.assertEqual(result['max_neighbor_count'], 3)
        self.assertEqual(result['groups'], [])

    def test_absence_does_not_dilute_pair_history(self):
        detector = ScalarDBSCANFilter()
        norms, cosines = {0: 100., 1: 100., 2: 100., 3: 130.}, {i: .95 for i in range(4)}
        detector.apply(norms, cosines, 0)
        for index in range(1, 7):
            detector.apply({3: 130., 4: 140.}, {3: .95, 4: .95}, index)
        self.assertEqual(detector.apply(norms, cosines, 7)['confirmed_ids'], [])
        self.assertEqual(detector.apply(norms, cosines, 8)['confirmed_ids'], [0, 1, 2])

    def test_client_with_changing_partners_is_not_confirmed(self):
        detector = ScalarDBSCANFilter(min_group=2)
        for index in range(3):
            norms = {i: 120. + 10 * i for i in range(5)}
            norms[0] = norms[index + 1] = 100.
            result = detector.apply(norms, {i: .95 for i in range(5)}, index)
        self.assertIn(0, result['candidate_ids'])
        self.assertEqual(result['confirmed_ids'], [])

    def test_single_client_and_invalid_configuration(self):
        result = ScalarDBSCANFilter().apply({7: 1.}, {7: .95}, 0)
        self.assertEqual(result['dbscan_noise_ids'], [7])
        self.assertEqual(result['rejected_ids'], [])
        for kwargs in ({'k': 3}, {'eps': 0.}, {'eps': float('nan')}, {'max_diameter': 0.}):
            with self.assertRaises(ValueError):
                ScalarDBSCANFilter(**kwargs)

    def test_stage1_exclusion_preserves_client_and_pair_history(self):
        norms = {0: 100., 1: 100., 2: 100., 3: 120., 4: 140., 5: 160., 6: 180.}
        in_main = {0: .95, 1: .95, 2: .95, 3: .949, 4: .951, 5: .2, 6: .21}
        outside_main = {0: .2, 1: .2, 2: .2, 3: .95, 4: .951, 5: .952, 6: .953}
        repaired, legacy = ScalarDBSCANFilter(), ScalarConcentrationFilter()
        for index in range(3):
            repaired.apply(norms, in_main, index)
            legacy.apply(norms, in_main, index)
        for index in range(3, 6):
            result = repaired.apply(norms, outside_main, index)
            legacy.apply(norms, outside_main, index)
            self.assertEqual(result['cluster_rejected_ids'], [0, 1, 2])
            self.assertEqual(result['confirmed_ids'], [])
            self.assertEqual(result['history_observation_counts'][0], 3)
            self.assertEqual(len(repaired.pair_history[(0, 1)]), 3)
        result = repaired.apply(norms, in_main, 6)
        old = legacy.apply(norms, in_main, 6)
        self.assertEqual(result['confirmed_ids'], [0, 1, 2])
        self.assertEqual(result['history_scores'][0], 1.)
        self.assertEqual(result['history_observation_counts'][0], 4)
        self.assertEqual(result['history_policy'], 'stage2_eligible_clients')
        # Keep the window baseline's historical behavior, including its zeros.
        self.assertEqual(old['confirmed_ids'], [])
        self.assertAlmostEqual(old['history_scores'][0], 4. / 7.)
        self.assertEqual(old['history_policy'], 'participating_clients')

    def test_eligible_non_candidate_still_records_a_negative(self):
        detector = ScalarDBSCANFilter()
        cosines = {i: .95 for i in range(4)}
        for index in range(3):
            detector.apply({0: 100., 1: 100., 2: 100., 3: 130.}, cosines, index)
        result = detector.apply({0: 100., 1: 110., 2: 120., 3: 130.}, cosines, 3)
        self.assertEqual(result['candidate_ids'], [])
        self.assertEqual(result['history_observation_counts'][0], 4)
        self.assertEqual(result['history_scores'][0], .75)
        self.assertEqual(list(detector.pair_history[(0, 1)]), [1, 1, 1, 0])

    def test_never_observed_client_has_no_history_score(self):
        result = ScalarDBSCANFilter().apply(
            {i: 100. + i for i in range(5)},
            {0: .2, 1: .21, 2: .95, 3: .951, 4: .952}, 0)
        self.assertEqual(result['history_observed_ids'], [2, 3, 4])
        self.assertEqual(result['history_observation_counts'][0], 0)
        self.assertIsNone(result['history_scores'][0])


if __name__ == '__main__':
    unittest.main()
