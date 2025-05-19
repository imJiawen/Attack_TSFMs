# ---------------------------------------------------------------------------------
# Portions of this file are derived from uni2ts
# - Source: https://github.com/SalesforceAIResearch/uni2ts
# - Paper: Unified Training of Universal Time Series Forecasting Transformers
# - License: Apache License 2.0

# We thank the authors for their contributions.
# ---------------------------------------------------------------------------------

import torch
import torch.nn as nn
from typing import Union
from einops import rearrange, repeat 
from .Moirai_backbone import MoiraiBackbone
from .moirai_module import MoiraiModule
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class Moirai(nn.Module):
    def __init__(
        self,
        prediction_length: int,
        context_length: int,
        variate_mode: str = 'M',
        patch_size: Union[str, int] = 'auto',
        model_size: str = 'base',
        scaling: bool = True,
        target_dim: int = 1,
        device=None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.variate_mode = variate_mode
        self.patch_size = patch_size if patch_size == 'auto' else int(patch_size)
        
        self.prediction_length = prediction_length
        self.context_length = context_length
        self.target_dim = target_dim
        
        self.moirai = MoiraiBackbone(
            module=MoiraiModule.from_pretrained(f"Salesforce/moirai-1.0-R-{model_size}"),
            prediction_length=self.prediction_length,
            context_length=self.context_length,
            patch_size=self.patch_size,
            target_dim=target_dim if self.variate_mode == 'M' else 1,
            scaling=scaling
        )
        
        self.q = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])  # Quantile levels
        if device is not None:
            self.device = device
            self.q = self.q.to(device)
            
    def loss(self, past_target, future_target, past_observed_target=None, past_is_pad=None, future_observed_target=None, future_is_pad=None):
        # input: [b l k]
        # dim_num = past_target.ndim
        
        past_target, past_observed_target, past_is_pad = self._prepare_input(past_target, past_observed_target, past_is_pad)
        future_target, future_observed_target, future_is_pad = self._prepare_input(future_target, future_observed_target, future_is_pad)

        # if dim_num == 2:
        #     past_target = past_target.unsqueeze(-1)
        #     past_observed_target = past_observed_target.unsqueeze(-1)
        #     future_target = future_target.unsqueeze(-1)
        #     future_observed_target = future_observed_target.unsqueeze(-1)
            
        loss = self.moirai.get_loss(
            self.patch_size,
            past_target,
            future_target,
            past_observed_target,
            future_observed_target,
            past_is_pad,
            future_is_pad)
        
        return torch.mean(loss)
        
    def forward(self, past_target, past_observed_target=None, past_is_pad=None, num_samples=100, **kwargs):   
        # input: [b l k]
        # output: [b k l q]
        dim_num = past_target.ndim

        # if past_target.ndim == 2:
        #     past_target = past_target.unsqueeze(1)
            
        past_target, past_observed_target, past_is_pad = self._prepare_input(past_target, past_observed_target, past_is_pad)
        
        # if dim_num == 2:
        #     past_target = past_target.unsqueeze(-1)
        #     past_observed_target = past_observed_target.unsqueeze(-1)
        
        k = past_target.shape[-1]

        forecasts = self.moirai(
            past_target=past_target,
            past_observed_target=past_observed_target,
            past_is_pad=past_is_pad,
            num_samples=num_samples
        )
        
        if forecasts.ndim == 3:
            forecasts = forecasts.unsqueeze(-1)
        
        # print("forecasts.shape ", forecasts.shape)
        forecasts = rearrange(forecasts, 'b n l k -> (b k) n l')
        mean_forecasts = forecasts.mean(dim=1)  # Shape: [B, L]
        quantiles = torch.quantile(forecasts, self.q, dim=1)
        quantiles = rearrange(quantiles, 'q b l -> b l q')
        
        
        quantile_forecasts = torch.cat([mean_forecasts.unsqueeze(-1), quantiles], dim=-1)  # Shape: [B, L, 10]
        
        if dim_num == 3:
            mean_forecasts = rearrange(mean_forecasts, '(b k) l -> b k l', k=k)
            quantile_forecasts = rearrange(quantile_forecasts, '(b k) l q -> b k l q', k=k)
        return mean_forecasts, quantile_forecasts
    
    def _prepare_input(self, target, observed_target=None, is_pad=None):
        # target:[b, l, k]
        # observed_target: [b, l, k]
        if target.ndim == 2:
            target = target.unsqueeze(-1)
            if observed_target is not None:
                observed_target = observed_target.unsqueeze(-1)
            
        # target = target.transpose(1,2)
        b, l, k = target.shape
        
        if observed_target is None:
            observed_target = observed_target = torch.ones_like(target).to(target.device)
        else:
            # observed_target = observed_target.transpose(1,2).to(target.device)
            observed_target = observed_target.to(target.device)
            
        if is_pad is None:
            # is_pad = torch.zeros_like(target)
            is_pad = torch.zeros([b,l]).to(target.device)
        else:
            is_pad = is_pad.to(target.device)
            
        return target, observed_target, is_pad