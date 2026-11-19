import contextlib
import io
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import tempfile
import unittest

import numpy as np

from tools.prepare_federated_zskt import prepare_legacy


class FederatedDirichletTest(unittest.TestCase):
    def make_source(self, root):
        source = root / 'source'
        source.mkdir()
        (source / 'config.json').write_text(json.dumps({
            'num_clients': 2, 'num_classes': 10, 'non_iid': True,
            'partition': 'dir', 'alpha': .1, 'balance': False}))
        for split in ('train', 'test'):
            (source / split).mkdir()
            for client, count in ((0, 4), (1, 9)):
                data = {'x': np.full((count, 3, 32, 32), client, dtype=np.float32),
                        'y': np.full(count, client * 9, dtype=np.int64)}
                np.savez(source / split / f'{client}.npz', data=data)
        return source

    def test_reuses_imbalanced_records_and_checks_cache_integrity(self):
        with tempfile.TemporaryDirectory() as directory:
            source = self.make_source(Path(directory))
            output = Path(directory) / 'cache'
            with contextlib.redirect_stdout(io.StringIO()):
                manifest = prepare_legacy(source, output, 2, 'dirichlet', .1)
                reused = prepare_legacy(source, output, 2, 'dirichlet', .1)
            self.assertEqual(manifest, reused)
            self.assertEqual(manifest['partition_details'], {
                'scheme': 'dirichlet', 'alpha': .1, 'balanced': False,
                'reused_existing_records': True})
            self.assertEqual([row['samples'] for row in manifest['train']['clients']], [4, 9])
            self.assertEqual(manifest['train']['clients'][0]['class_counts'], [4] + [0] * 9)
            for split in ('train', 'test'):
                for client in range(2):
                    name = f'{split}/{client}.npz'
                    self.assertEqual((source / name).read_bytes(), (output / name).read_bytes())
            with (output / 'test/1.npz').open('ab') as stream:
                stream.write(b'changed')
            with self.assertRaisesRegex(ValueError, 'changed'):
                prepare_legacy(source, output, 2, 'dirichlet', .1)

    def test_rejects_wrong_alpha_or_partition_before_writing_output(self):
        with tempfile.TemporaryDirectory() as directory:
            source = self.make_source(Path(directory))
            output = Path(directory) / 'cache'
            for alpha in (.2, 0, -.1, float('nan'), float('inf')):
                with self.subTest(alpha=alpha), self.assertRaises(ValueError):
                    prepare_legacy(source, output, 2, 'dirichlet', alpha)
                self.assertFalse(output.exists())
            with self.assertRaisesRegex(ValueError, 'Requested IID'):
                prepare_legacy(source, output, 2, 'iid', .1)
            config_path = source / 'config.json'
            config = json.loads(config_path.read_text())
            config['non_iid'] = False
            config_path.write_text(json.dumps(config))
            with self.assertRaisesRegex(ValueError, 'not non_iid'):
                prepare_legacy(source, output, 2, 'dirichlet', .1)
            self.assertFalse(output.exists())

    @unittest.skipUnless(shutil.which('bash'), 'launcher validation requires bash')
    def test_both_arms_share_partition_attack_and_local_training(self):
        script = Path(__file__).resolve().parents[1] / 'tools/reproduce_federated_zskt.sh'
        environment = {k: v for k, v in os.environ.items() if not k.startswith('FL_ZSKT_')}
        environment.update(DFKD_DRY_RUN='1', DFKD_PYTHON=sys.executable,
                           FL_ZSKT_PARTITION='dirichlet', FL_ZSKT_ALPHA='0.10')
        commands, preparations = {}, []
        for mode, cycles in (('fedavg', '0'), ('badnets', '200')):
            output = subprocess.check_output(['bash', str(script), mode], text=True, env=environment)
            preparation, command = [shlex.split(line) for line in output.splitlines()]
            preparations.append(preparation)
            options = dict(zip(command[3::2], command[4::2]))  # python -u main.py --flag value
            prep_options = dict(zip(preparation[4::2], preparation[5::2]))
            self.assertEqual(prep_options['--partition'], 'dirichlet')
            self.assertEqual(prep_options['--alpha'], '0.1')
            self.assertEqual(prep_options['--protocol'], 'legacy')
            self.assertEqual(options['--dataset'], 'Cifar10_ZSKT_dir0.1_legacy20')
            self.assertEqual(options['--logger_root'], '../results/federated_zskt_badnets_reference_dir0.1')
            self.assertEqual(options.pop('--zskt_cycles_per_round'), cycles)
            self.assertEqual(options.pop('--goal'), f'{mode}_reference_legacy_dir0.1')
            commands[mode] = options
        self.assertEqual(preparations[0], preparations[1])
        self.assertEqual(commands['fedavg'], commands['badnets'])

    @unittest.skipUnless(shutil.which('bash'), 'launcher validation requires bash')
    def test_rejects_unsupported_protocol_and_invalid_alpha(self):
        script = Path(__file__).resolve().parents[1] / 'tools/reproduce_federated_zskt.sh'
        environment = {k: v for k, v in os.environ.items() if not k.startswith('FL_ZSKT_')}
        environment.update(DFKD_DRY_RUN='1', DFKD_PYTHON=sys.executable,
                           FL_ZSKT_PARTITION='dirichlet')
        cases = [({'FL_ZSKT_DATA_PROTOCOL': 'official'}, 'set FL_ZSKT_DATA_PROTOCOL=legacy'),
                 ({'FL_ZSKT_ALPHA': '-0.1'}, 'positive'),
                 ({'FL_ZSKT_ALPHA': 'nan'}, 'finite'),
                 ({'FL_ZSKT_PARTITION': 'wrong'}, 'must be iid or dirichlet')]
        for overrides, diagnostic in cases:
            with self.subTest(overrides=overrides):
                result = subprocess.run(['bash', str(script), 'badnets'], text=True,
                                        capture_output=True, env={**environment, **overrides})
                self.assertNotEqual(result.returncode, 0)
                self.assertIn(diagnostic, result.stderr)


if __name__ == '__main__':
    unittest.main()
