"""Bridge native plaintext uploads to a detector that receives two scalars only."""
import torch

from flcore.security.defense.dnc import DefenseResult
from flcore.security.defense.scalar_concentration import ScalarKMeansOnlyFilter
from flcore.security.defense.scalar_dbscan import ScalarDBSCANFilter


def submitted_model_scalars(global_model, uploaded_models, client_ids):
    """Compute ||P_i|| and cos(P_i,W), excluding buffers and encoding offsets.

    In FedAvg P_i is the submitted local model; in FedSGD it is the existing
    W+g gradient carrier. Process one model at a time, without an n-by-d matrix.
    This plaintext extraction bridge is separate from the scalar-only detector.
    """
    if len(uploaded_models) != len(client_ids) or len(set(client_ids)) != len(client_ids):
        raise ValueError('Require one uploaded model per unique client ID')
    reference = torch.cat([p.detach().reshape(-1) for p in global_model.parameters()]).float()
    reference_norm = torch.linalg.vector_norm(reference)
    norms, cosines = {}, {}
    for cid, model in zip(client_ids, uploaded_models):
        vector = torch.cat([p.detach().reshape(-1) for p in model.parameters()]).to(reference)
        if vector.shape != reference.shape:
            raise ValueError('Submitted model parameters do not match global model')
        norm = torch.linalg.vector_norm(vector)
        denominator = norm * reference_norm
        cosine = torch.dot(vector, reference) / denominator if denominator > 1e-12 else 0.
        norms[int(cid)] = float(norm)
        cosines[int(cid)] = float(cosine)
    return norms, cosines


class ScalarModelDefense:
    """Majority KMeans, optionally followed by persistent compact DBSCAN groups.

    The backend accepts only ID-keyed norm/cosine maps and a round number.
    IDs identify history; no malicious IDs/ratio, data, or vectors reach it.
    """

    def __init__(self, method='scalar_dbscan', verbose=False, **kwargs):
        if method not in ('scalar_kmeans', 'scalar_dbscan'):
            raise ValueError('Unknown scalar-model defense')
        self.method = method
        self.verbose = bool(verbose)
        backend = ScalarDBSCANFilter if method == 'scalar_dbscan' else ScalarKMeansOnlyFilter
        if method == 'scalar_kmeans':
            kwargs.pop('eps', None)
            kwargs.pop('max_diameter', None)
        self.backend = backend(k=2, scope='main', **kwargs)
        self.last_details = None

    def filter_scalars(self, norms, cosines, client_ids, round_id=0, filtering_enabled=True):
        """Update eligible histories even during warm-up; apply masks only if enabled."""
        if len(client_ids) != len(set(client_ids)) or set(client_ids) != set(norms):
            raise ValueError('Scalar maps must match the unique uploaded client IDs')
        details = self.backend.apply(norms, cosines, round_id)
        stage1 = details['cluster_rejected_ids'] if filtering_enabled else []
        stage2 = details['confirmed_ids'] if filtering_enabled else []
        rejected = sorted(set(stage1) | set(stage2))
        rejected_set = set(rejected)
        kept = [i for i, cid in enumerate(client_ids) if cid not in rejected_set]
        removed = [i for i, cid in enumerate(client_ids) if cid in rejected_set]
        details.update(filtered_ids=rejected, cluster_filtered_ids=stage1,
                       density_filtered_ids=stage2, filtering_enabled=bool(filtering_enabled),
                       aggregation_ids=sorted(cid for cid in client_ids if cid not in rejected_set))
        self.last_details = details
        for j, (center, size) in enumerate(zip(details['cluster_centers'], details['cluster_sizes'])):
            ids = sorted(cid for cid, label in details['cluster_ids'].items() if label == j)
            print(f'[{self.method}] cluster={j} n={size} center={center:.10f} '
                  f'stage1={"KEEP" if j == details["main_cluster"] else "DROP"} ids={ids}')
        if self.method == 'scalar_dbscan':
            for group in details['dbscan_clusters']:
                print(f'[scalar_dbscan] subcluster={group["label"]} ids={group["ids"]} '
                      f'diameter={group["diameter"]:.6g} candidate={group["candidate"]}')
            print(f'[scalar_dbscan] history_confirmed={details["confirmed_ids"]}')
        print(f'[{self.method}] enabled={filtering_enabled} '
              f'stage1_drop={stage1} stage2_drop={stage2} final_keep={details["aggregation_ids"]}')
        if self.verbose:
            for cid in sorted(client_ids, key=lambda cid: (-cosines[cid], cid)):
                print(f'  client={cid} cos={cosines[cid]:.10f} '
                      f'model_L2={norms[cid]:.8f} aggregate={cid not in rejected_set}')
        return DefenseResult(self.method, [cosines[cid] for cid in client_ids],
                             kept, removed, [], selection_rule='largest_cluster')

    def filter(self, global_model, uploaded_models, uploaded_weights, round_id=0,
               client_ids=None, filtering_enabled=True):
        if client_ids is None:
            raise ValueError('Real client IDs are required for persistent scalar histories')
        if len(uploaded_weights) != len(client_ids):
            raise ValueError('Require one pre-defense weight per upload')
        norms, cosines = submitted_model_scalars(global_model, uploaded_models, client_ids)
        return self.filter_scalars(norms, cosines, client_ids, round_id, filtering_enabled)
