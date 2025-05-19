import torch
import torch.nn as nn
from einops import rearrange, repeat 
from .UniTS_zeroshot import Model
import sys
import os

class UniTS(nn.Module):
    def __init__(
        self,
        prediction_length: int,
        context_length: int,
        model_size: str = '128',
        target_dim=None,
        device=None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.target_dim = target_dim
        self.module = Model(
            d_model=int(model_size), 
            seq_len=context_length, 
            pred_len=prediction_length,
        )
        
        if model_size not in ['128', '32']:
            print('Invalid model size. Please choose from x128 or x32')
            sys.exit()
            
        ckpt_path = f'./models/units/ckpt/units_x{model_size}_pretrain_checkpoint.pth'
        if not os.path.exists(ckpt_path):
            print('Checkpoint not found. Please download ckpt first.')
            sys.exit()
            
        self.load_from_checkpoint(ckpt_path)
        if device is not None:
            self.device = device
            self.module = self.module.to(device)
        
    def load_from_checkpoint(self, ckpt_dir):
        state_dict = torch.load(
            ckpt_dir, map_location='cpu',weights_only=False)['student']
        ckpt = {}
        for k, v in state_dict.items():
            if not ('cls_prompts' in k):
                ckpt[k] = v
                
        msg = self.load_state_dict(ckpt, strict=False)
        # print(msg)
        print('UniTS model loaded from checkpoint.')
            
    def loss(self, past_target, future_target):
        forecasts, _ = self(past_target)
        return torch.mean((forecasts - future_target) ** 2)
        
    def forward(self, past_target, **kwargs):  
        dim_num = past_target.ndim 
        if dim_num == 2:
            if self.target_dim is not None:
                past_target = rearrange(past_target, '(b k) l -> b k l', k=self.target_dim)
            else:
                self.target_dim = 1
                past_target = past_target.unsqueeze(1)
            
        past_target = rearrange(past_target, 'b k l -> b l k')
        forecasts = self.module.forecast(past_target) # [B L K]
        
        
        mean_forecasts = rearrange(forecasts, 'b l k -> b k l')
        quantile_forecasts = repeat(mean_forecasts, 'b k l -> b k l q', q=9)
        quantile_forecasts = torch.cat([mean_forecasts.unsqueeze(-1), quantile_forecasts], dim=-1)  # Shape: [B, K, L, 10]
        
        if dim_num == 2:
            mean_forecasts = rearrange(mean_forecasts, 'b k l -> (b k) l')
            quantile_forecasts = repeat(quantile_forecasts, 'b k l q -> (b k) l q')
        return mean_forecasts, quantile_forecasts
    