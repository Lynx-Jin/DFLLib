"""Prepare official or exact historical CIFAR10 train/holdout client records."""
import argparse
import hashlib
import json
from pathlib import Path
import pickle

import numpy as np


def balanced_iid_indices(labels, num_clients, seed):
    labels = np.asarray(labels)
    if num_clients < 1:
        raise ValueError('Require at least one client')
    rng = np.random.default_rng(seed)
    groups = [[] for _ in range(num_clients)]
    for label in np.unique(labels):
        positions = np.flatnonzero(labels == label)
        if len(positions) < num_clients:
            raise ValueError('Each class must contain at least one sample per client')
        rng.shuffle(positions)
        for group, part in zip(groups, np.array_split(positions, num_clients)):
            group.extend(part.tolist())
    arrays = [rng.permutation(np.asarray(group, dtype=np.int64)) for group in groups]
    union = np.concatenate(arrays)
    if len(union) != len(labels) or len(np.unique(union)) != len(labels):
        raise AssertionError('Every source record must belong to exactly one client')
    return arrays


def read_official_split(raw_root, train):
    root = Path(raw_root) / 'cifar-10-batches-py'
    names = [f'data_batch_{i}' for i in range(1, 6)] if train else ['test_batch']
    data, labels = [], []
    for name in names:
        with (root / name).open('rb') as stream:
            part = pickle.load(stream, encoding='bytes')
        data.append(part[b'data'].reshape(-1, 3, 32, 32))
        labels.extend(part[b'labels'])
    data = np.concatenate(data)
    labels = np.asarray(labels, dtype=np.int64)
    expected = 50000 if train else 10000
    if len(data) != expected or np.bincount(labels, minlength=10).tolist() != [expected // 10] * 10:
        raise ValueError('Unexpected official CIFAR10 split')
    return data, labels


def prepare(raw_root, output, num_clients=20, seed=0):
    output = Path(output)
    config = {'source': 'official_cifar10_train_test_separate',
              'partition': 'class-balanced IID', 'num_clients': num_clients,
              'seed': seed, 'normalization': 'float32 x/127.5-1 in [-1,1]'}
    manifest_path = output / 'partition_manifest.json'
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text())
        if any(previous.get(k) != v for k, v in config.items()):
            raise ValueError('Existing client partitions use different settings; choose a new directory')
        if not all((output / split / f'{client}.npz').is_file()
                   for split in ['train', 'test'] for client in range(num_clients)):
            raise ValueError('Existing partition manifest is incomplete')
        print(f'Using existing verified client split: {output}')
        return previous
    if output.exists() and any(output.iterdir()):
        raise ValueError('Refusing to overwrite a nonempty directory without a matching manifest')
    output.mkdir(parents=True, exist_ok=True)
    manifest = dict(config)
    for train, split, split_seed in [(True, 'train', seed), (False, 'test', seed + 1)]:
        images, labels = read_official_split(raw_root, train)
        parts = balanced_iid_indices(labels, num_clients, split_seed)
        (output / split).mkdir(exist_ok=True)
        records = []
        for client, indices in enumerate(parts):
            pixels = images[indices].astype(np.float32) / np.float32(127.5) - np.float32(1)
            payload = {'x': pixels, 'y': labels[indices]}
            np.savez_compressed(output / split / f'{client}.npz', data=payload)
            np.save(output / split / f'{client}_official_indices.npy', indices)
            records.append({'client_id': client, 'samples': len(indices),
                            'class_counts': np.bincount(labels[indices], minlength=10).tolist(),
                            'official_indices_sha256': hashlib.sha256(indices.tobytes()).hexdigest()})
        manifest[split] = {'official_samples': len(images), 'clients': records}
        print(f'Prepared {split}: {len(images)} official records, {num_clients} clients', flush=True)
    manifest_path.write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    return manifest


def file_digest(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def prepare_legacy(source, output, num_clients=20, partition=None, alpha=None):
    """Copy the exact historical train/holdout records, without repartitioning."""
    import shutil
    source, output = Path(source).resolve(), Path(output).resolve()
    if source == output:
        raise ValueError('Legacy cache must be separate from the historical source')
    config_path = source / 'config.json'
    if not config_path.is_file():
        raise ValueError(f'Legacy client config not found: {config_path}')
    source_config = json.loads(config_path.read_text())
    if source_config.get('num_clients') != num_clients or source_config.get('num_classes') != 10:
        raise ValueError('Legacy client count/classes must match the requested experiment')
    is_dirichlet = bool(source_config.get('non_iid', False)) and source_config.get('partition') == 'dir'
    if partition == 'iid' and bool(source_config.get('non_iid', False)):
        raise ValueError('Requested IID but the source config is non-IID')
    if partition == 'dirichlet':
        if not is_dirichlet:
            raise ValueError('Requested Dirichlet but source config is not non_iid=True, partition=dir')
        if alpha is None or not np.isfinite(alpha) or alpha <= 0:
            raise ValueError('Dirichlet alpha must be finite and positive')
        source_alpha = float(source_config.get('alpha', float('nan')))
        if not np.isfinite(source_alpha) or not np.isclose(source_alpha, alpha, rtol=1e-9, atol=1e-12):
            raise ValueError(f'Requested alpha={alpha} differs from source alpha={source_alpha}')
    config = {'source': 'legacy_cifar10_client_holdout', 'num_clients': num_clients,
              'partition': 'exact historical client records; no repartitioning',
              'legacy_source': str(source), 'legacy_config_sha256': file_digest(config_path),
              'normalization': 'stored client tensors in [-1,1]; do not normalize twice'}
    if is_dirichlet:
        config['partition_details'] = {
            'scheme': 'dirichlet', 'alpha': float(source_config['alpha']),
            'balanced': bool(source_config.get('balance', False)),
            'reused_existing_records': True,
        }

    manifest_path = output / 'partition_manifest.json'
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text())
        if any(previous.get(key) != value for key, value in config.items()):
            raise ValueError('Existing legacy cache has different source/settings')
        for split in ['train', 'test']:
            if len(previous[split]['clients']) != num_clients:
                raise ValueError('Incomplete legacy cache')
            for record in previous[split]['clients']:
                name = str(record['client_id']) + '.npz'
                old, cached = source / split / name, output / split / name
                if (not old.is_file() or not cached.is_file()
                        or file_digest(old) != record['file_sha256']
                        or file_digest(cached) != record['file_sha256']):
                    raise ValueError('Legacy source/cache changed; choose a new output directory')
        print(f'Using exact historical client split: {output}', flush=True)
        return previous
    if output.exists() and any(output.iterdir()):
        raise ValueError('Refusing to overwrite a nonempty directory without a matching manifest')
    # Validate all required inputs before creating a cache.
    for split in ['train', 'test']:
        for client_id in range(num_clients):
            if not (source / split / f'{client_id}.npz').is_file():
                raise ValueError(f'Missing legacy {split} client {client_id}')
    output.mkdir(parents=True, exist_ok=True)
    manifest = dict(config)
    for split in ['train', 'test']:
        (output / split).mkdir(exist_ok=True)
        records = []
        for client_id in range(num_clients):
            original = source / split / f'{client_id}.npz'
            with np.load(original, allow_pickle=True) as archive:
                payload = archive['data'].item()
            images, labels = np.asarray(payload['x']), np.asarray(payload['y'], dtype=np.int64)
            if (images.ndim != 4 or tuple(images.shape[1:]) != (3, 32, 32)
                    or not len(images) or len(images) != len(labels)):
                raise ValueError('Legacy records must be nonempty CIFAR10 NCHW images and labels')
            if (not np.isfinite(images).all() or images.min() < -1.001 or images.max() > 1.001
                    or not ((labels >= 0) & (labels < 10)).all()):
                raise ValueError('Legacy records must be normalized to [-1,1] with CIFAR10 labels')
            destination = output / split / original.name
            shutil.copy2(original, destination)
            records.append({'client_id': client_id, 'samples': len(labels),
                            'class_counts': np.bincount(labels, minlength=10).tolist(),
                            'file_sha256': file_digest(destination)})
        manifest[split] = {'samples': sum(record['samples'] for record in records), 'clients': records}
        print(f'Copied historical {split}: {manifest[split]["samples"]} records, '
              f'clients 0..{num_clients-1}', flush=True)
    manifest_path.write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path)
    parser.add_argument('--protocol', choices=('official', 'legacy'), default='official')
    parser.add_argument('--legacy-root', type=Path, default=Path('../dataset/Cifar10'))
    parser.add_argument('--partition', choices=('iid', 'dirichlet'), default=None)
    parser.add_argument('--alpha', type=float, default=0.1)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--num-clients', type=int, default=20)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    if args.num_clients < 2 or args.num_clients > 1000 or args.seed < 0:
        parser.error('Require 2..1000 clients and a nonnegative seed')
    if args.protocol == 'legacy':
        prepare_legacy(args.legacy_root, args.output, args.num_clients, args.partition, args.alpha)
    else:
        if args.partition == 'dirichlet':
            parser.error('Dirichlet mode reuses existing client records; use --protocol legacy')
        if args.data_root is None:
            parser.error('--data-root is required for the official protocol')
        prepare(args.data_root, args.output, args.num_clients, args.seed)


if __name__ == '__main__':
    main()
