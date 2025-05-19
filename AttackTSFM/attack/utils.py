import torch

def project_perturbation(norm, delta, epsilon):
    if norm == 'linf':
        return delta.clamp(-epsilon, epsilon)
    elif norm == 'l2':
        norm = delta.norm(p=2, dim=1, keepdim=True)
        factor = torch.clamp(epsilon / (norm + 1e-10), max=1.0)
        return delta * factor
    elif norm == 'l0':
        abs_val = delta.abs()
        topk_mask = torch.zeros_like(delta)
        l0_k = int(epsilon * delta.shape[-1])
        topk_indices = abs_val.topk(l0_k, dim=1).indices
        topk_mask.scatter_(1, topk_indices, 1.0)
        return delta * topk_mask
    else:
        raise ValueError(f"Unsupported norm type: {norm}")
    