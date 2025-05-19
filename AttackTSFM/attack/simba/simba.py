import torch
from AttackTSFM.attack.simba.simba_utils import block_idct, block_iwavelet

class SimBA:
    def __init__(self, model, targeted=False, mode='random', basis='point', block_size=None, mask_ratio=0.5):
        self.model = model
        self.targeted = targeted
        self.mode = mode
        self.basis = basis
        self.block_size = block_size
        self.mask_ratio = mask_ratio
        
    def _scale_eps(self, epsilon, past_tensor):
        std = past_tensor.std(dim=-1)
        scaled_epsilon = std * epsilon
        return scaled_epsilon
    
    def expand_vector(self, x, length):
        batch_size, n_dim = x.shape
        z = torch.zeros(batch_size, length).to(x.device)
        z[:, -n_dim:] = x
        return z
        
    
    def get_probs(self, x, y, **kwargs):
        forecasts, _ = self.model(x, **kwargs)
        log_probs = (forecasts - y) ** 2
        return log_probs.mean(dim=-1)
    
    def get_preds(self, x, **kwargs):
        preds, _ = self.model(x, **kwargs)
        return preds

    def attack_batch(self, past_tensor, target_tensor, n_dims, max_iters, epsilon=0.2, **kwargs):
        batch_size, length = past_tensor.shape
        
        block_size = length if not self.block_size else self.block_size

        scaled_epsilon = self._scale_eps(epsilon, past_tensor)
        linf_bound = scaled_epsilon.unsqueeze(-1)
        
        x = torch.zeros(batch_size, n_dims).to(past_tensor.device)
        
        if self.mode == 'random':
            indices = torch.randperm(n_dims)[:max_iters]
        elif self.mode == 'coordinate':
            indices = torch.arange(0,n_dims)[:max_iters]
        else:
            raise ValueError("Invalid mode for SimBA")
        
        all_indices = torch.arange(0, batch_size).long().to(past_tensor.device)
        
        if self.basis == 'point':
            trans = lambda z: z
        elif self.basis == 'dct':
            trans = lambda z: block_idct(z, block_size=block_size, linf_bound=linf_bound, ratio=self.mask_ratio)
        elif self.basis == 'wavelet':
            trans = lambda z: block_iwavelet(z, block_size=block_size, wavelet='db1', ratio=self.mask_ratio, linf_bound=linf_bound)
        else:
            raise ValueError("Invalid basis for SimBA")

        prev_probs = self.get_probs(past_tensor, target_tensor, **kwargs) 
        
        for i in range(max_iters):
            dim = indices[i]
            diff = torch.zeros(batch_size, n_dims).to(past_tensor.device)

            diff[:, dim] = scaled_epsilon
            left_vec = x - diff
            right_vec = x + diff
            
            # trying negative direction
            adv = (past_tensor + trans(self.expand_vector(left_vec, length)))
            left_probs = self.get_probs(adv, target_tensor, **kwargs)
            
            # try positive directions
            adv = (past_tensor + trans(self.expand_vector(right_vec, length)))
            right_probs = self.get_probs(adv, target_tensor, **kwargs)
            
            if not self.targeted:
                left_improved = left_probs.gt(prev_probs)
                right_improved = right_probs.gt(torch.max(prev_probs, left_probs))
            else:
                left_improved = left_probs.lt(prev_probs)
                right_improved = right_probs.lt(torch.min(prev_probs, left_probs))
                
            # update x depending on which direction improved
            if left_improved.sum() > 0:
                left_indices = all_indices[left_improved] # b 
                left_mask_remaining = left_improved.unsqueeze(1).repeat(1, n_dims) # b n_dims
                x[left_indices] = left_vec[left_mask_remaining].view(-1, n_dims)
                prev_probs = left_probs.clone()
            if right_improved.sum() > 0:
                right_indices = all_indices[right_improved]
                right_mask_remaining = right_improved.unsqueeze(1).repeat(1, n_dims)
                x[right_indices] = right_vec[right_mask_remaining].view(-1, n_dims)
                prev_probs = right_probs.clone()
        
        perturbation = trans(self.expand_vector(x, length)) 
        adv = past_tensor + perturbation
        return adv