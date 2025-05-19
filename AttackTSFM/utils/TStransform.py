import torch
import random
import sys

class TimeSeriesTransform:
    def __init__(self, a=1.0, c_fn=None):
        """
        General time series transformation: M(x) = a * x + c(tau)
        
        Args:
            a (float): Global scaling factor.
            c_fn (callable or None): A function c(tau) that returns a tensor of shape (time_steps,)
                                     or a tensor of shape (time_steps,) directly.
        """
        self.a = a
        self.c_fn = c_fn

    def __call__(self, x):
        """
        Apply the transformation to input tensor.

        Args:
            x (torch.Tensor): Input of shape (B, T) or (B, V, T)

        Returns:
            torch.Tensor: Transformed tensor of the same shape as input.
        """
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32)

        if x.ndim == 2:
            # (B, T) → (B, 1, T)
            x = x.unsqueeze(1)
            unsqueeze_later = True
        elif x.ndim == 3:
            unsqueeze_later = False
        else:
            raise ValueError("Input must have shape (B, T) or (B, V, T)")

        # Compute mean and std along time dimension
        start_point=x[:,:, 0] # for align
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True) + 1e-6
        x_norm = (x - mean) / std

        # Get c(τ): shape = (1, 1, T) to allow broadcasting
        if callable(self.c_fn):
            c = self.c_fn(x.size(-1)).reshape(1, 1, -1).to(x_norm.device)
        elif isinstance(self.c_fn, torch.Tensor):
            c = self.c_fn.reshape(1, 1, -1).to(x_norm.device)
        else:
            c = 0.0

        # Apply transformation and rescale
        x_trans = self.a * x_norm + c
        x_rescaled = x_trans * std + mean

        # for align
        scaled_start = x_rescaled[:,:,0]
        shifts = (start_point - scaled_start).unsqueeze(-1)
        x_rescaled = x_rescaled + shifts
        
        if unsqueeze_later:
            return x_rescaled.squeeze(1)  # back to (B, T)
        
        return x_rescaled  # (B, V, T)
    
def drift_fn(length, beta=0.05):
    return beta * torch.arange(length, dtype=torch.float32)


def local_offset_fn(length, delta=None, affected_steps=None, K=None, start_step=None, delta_range=(-1.0, 1.0), random_sample=False):
    """
    Generate a local offset vector c(tau) for selected time steps.

    Args:
        length (int): Length of the forecast horizon.
        delta (float or list or torch.Tensor): Either a constant offset or a list of offsets.
        affected_steps (list of int): Indices to apply the offset.

    Returns:
        torch.Tensor: A vector of shape (length,) with local offsets applied.
    """
    assert K or delta, "Either K or delta must be provided in local_offset_fn."
    c = torch.zeros(length)
    
    if K:
        if K < 1:
            K = int(K * length)
        else:
            K = int(K)
        affected_steps, delta = gaussian_local_offsets(length, K, start_step=start_step, delta_range=delta_range, random_sample=random_sample)

    if isinstance(delta, (int, float)):
        for t in affected_steps:
            if t < length:
                c[t] = delta
    elif isinstance(delta, (list, torch.Tensor)):
        assert len(delta) == len(affected_steps), \
            "Length of delta must match length of affected_steps"
        for i, t in enumerate(affected_steps):
            if t < length:
                c[t] = delta[i]
    else:
        raise TypeError("delta must be a float, list, or torch.Tensor")

    return c

def random_local_offsets(T, K, delta_range=(-1.0, 1.0)):
    """
    Randomly sample K time steps from [0, T) and assign random deltas from a specified range.

    Args:
        T (int): Total number of time steps.
        K (int): Number of steps to sample.
        delta_range (tuple): Range (min, max) to sample delta values.

    Returns:
        affected_steps (list of int): Indices of affected steps.
        deltas (torch.Tensor): Corresponding delta values.
    """
    assert K <= T, "K must be less than or equal to T."

    # Randomly sample K unique steps without replacement
    affected_steps = torch.randperm(T)[:K].tolist()

    # Randomly sample K delta values from uniform distribution in delta_range
    deltas = (delta_range[1] - delta_range[0]) * torch.rand(K) + delta_range[0]

    return affected_steps, deltas

def gaussian_local_offsets(T, K, start_step=None, delta_range=[-1.0, 1.0], random_sample=False):
    """
    Randomly sample a starting point and generate K consecutive steps,
    with deltas following a Gaussian curve scaled to a random delta.

    Args:
        T (int): Total number of time steps.
        K (int): Number of consecutive steps to generate.
        delta_range (tuple): Range (min, max) to sample the peak delta value.

    Returns:
        affected_steps (list of int): Consecutive step indices.
        deltas (torch.Tensor): Gaussian-shaped delta values.
    """
    assert K <= T, "K must be less than or equal to T."

    # Randomly sample a starting point
    max_start = T - K
    
    if start_step is None:
        start_step = torch.randint(0, max_start + 1, (1,)).item()

    if start_step < 1:
        start_step = int(start_step * max_start)
    else:
        start_step = int(start_step)
    # Define affected steps: consecutive
    affected_steps = list(range(start_step, start_step + K))

    # Sample peak delta value
    if isinstance(delta_range, float) or isinstance(delta_range, int):
        peak_delta = delta_range
    else:
        if len(delta_range) == 1:
            peak_delta = delta_range[0]
        elif len(delta_range) == 2 and random_sample:
            peak_delta = (delta_range[1] - delta_range[0]) * torch.rand(1).item() + delta_range[0]
        else:
            peak_delta = random.choice(delta_range)

    # Generate Gaussian-shaped curve centered in the K steps
    center = (K - 1) / 2
    sigma = K / 6  # Standard deviation controls spread; 99.7% data within ±3σ

    x = torch.arange(K, dtype=torch.float32)
    gauss_curve = torch.exp(-0.5 * ((x - center) / sigma) ** 2)

    # Normalize to have maximum 1, then scale to peak_delta
    gauss_curve = gauss_curve / gauss_curve.max() * peak_delta
    return affected_steps, gauss_curve