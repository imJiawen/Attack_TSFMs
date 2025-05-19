import torch
import numpy as np
import pywt

def block_idct(x, block_size=8, masked=False, ratio=0.5, linf_bound=None):
    """
    Applies block-wise Inverse Discrete Cosine Transform (IDCT) to a 1D input.
    
    Parameters:
        x (Tensor): Input tensor of shape [batch_size, length].
        block_size (int): Size of each block for IDCT.
        masked (bool): Whether to apply a mask to the blocks before IDCT.
        ratio (float or Tensor): Masking ratio (percentage of coefficients kept).
        linf_bound (float): Optional bound for the output values.
    
    Returns:
        Tensor: Transformed tensor with the same shape as input.
    """
    device = x.device  # Ensure all tensors stay on the same device
    z = torch.zeros_like(x, device=device)  # Output tensor, same shape as x

    num_blocks = x.size(-1) // block_size  # Compute number of blocks
    mask = torch.zeros((x.size(0), block_size), device=device)  # Mask initialization

    # Define the masking matrix
    if isinstance(ratio, torch.Tensor):
        for i in range(x.size(0)):  
            mask[i, :int(block_size * ratio[i].item())] = 1
    else:
        mask[:, :int(block_size * ratio)] = 1

    # Process each block independently
    for i in range(num_blocks):
        submat = x[:, (i * block_size):((i + 1) * block_size)]  # Extract block
        if masked:
            submat = submat * mask  # Apply mask

        # Apply IDCT using PyTorch's inverse FFT (approximating ortho IDCT)
        submat_idct = torch.fft.irfft(submat, n=block_size, norm="ortho")
        z[:, (i * block_size):((i + 1) * block_size)] = submat_idct  # Store result

    # Apply optional bounding
    return z.clamp(-linf_bound, linf_bound) if linf_bound is not None else z

def block_iwavelet(x, block_size=8, wavelet='db1', ratio=0.5, linf_bound=None):
    """
    Applies block-wise inverse wavelet transform on input.

    Args:
        x (Tensor): Input frequency coefficients, shape [B, L].
        block_size (int): Size of each block.
        wavelet (str): Wavelet type (e.g., 'haar', 'db1', 'db4').
        ratio (float): Masking ratio, how many wavelet coeffs to retain.
        linf_bound (Tensor or float): Optional clamp on the final output.

    Returns:
        Tensor: Reconstructed time-domain signal, shape [B, L].
    """
    B, L = x.shape
    device = x.device
    output = torch.zeros_like(x)

    num_blocks = L // block_size

    for i in range(num_blocks):
        start, end = i * block_size, (i + 1) * block_size
        block = x[:, start:end].detach().cpu().numpy()

        rec_block = []
        for b in block:
            # Apply wavelet transform to get coeffs
            coeffs = pywt.wavedec(b, wavelet=wavelet, level=1)
            # Mask high-frequency details if ratio < 1
            if ratio < 1.0:
                coeffs[1] *= 0  # zero detail coefficients
            rec = pywt.waverec(coeffs, wavelet=wavelet)
            rec_block.append(rec[:block_size])

        rec_block = np.array(rec_block)
        rec_block = torch.from_numpy(rec_block).to(device=device, dtype=torch.float32)
        output[:, start:end] = rec_block

    if linf_bound is not None:
        output = output.clamp(-linf_bound, linf_bound)
    return output