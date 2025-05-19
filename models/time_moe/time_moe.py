import torch
import torch.nn as nn
from typing import Union
from einops import rearrange, repeat 
from transformers import AutoModelForCausalLM
import sys
from AttackTSFM.data.data_scaler import InstanceNorm


class TimeMoE(nn.Module):
    def __init__(
        self,
        prediction_length: int,
        context_length: int,
        model_size: str = '50M',
        target_dim=None,
        device='cuda',
        instance_norm=True,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.target_dim = target_dim
        self.prediction_length = prediction_length
        self.context_length = context_length
        self.loss_function = torch.nn.HuberLoss(reduction='none', delta=2.0)
        # self.loss_function = torch.nn.MSELoss(reduction='none')
        
        if instance_norm:
            self.normalization = InstanceNorm()
        else:
            self.normalization = None
        
        if model_size not in ['50M', '200M']:
            print('Invalid model size. Please choose from 50M or 200M')
            sys.exit()
            
            
        self.model = AutoModelForCausalLM.from_pretrained(
            f'Maple728/TimeMoE-{model_size}',
            device_map='auto',  # use "cpu" for CPU inference, and "cuda" for GPU inference.
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
        )
    
    
    def get_horizon_length(self, max_horizon_length):
        if max_horizon_length is None:
            horizon_length = self.model.config.horizon_lengths[0]
            max_horizon_length = horizon_length
        else: 
            horizon_length = self.model.config.horizon_lengths[0]
            for h in self.model.config.horizon_lengths[1:]:
                if h > max_horizon_length:
                    break
                else:
                    horizon_length = h
                    
        return horizon_length
                        
    def loss(self, past_target, future_target):
        """
        input: [b l k] or [b, l]
        """
        dim_num = past_target.ndim 
        
        past_target = past_target.to(dtype=torch.bfloat16)
        future_target = future_target.to(dtype=torch.bfloat16)
        if dim_num == 2:
            past_target = rearrange(past_target, 'b l -> b l 1')
            future_target = rearrange(future_target, 'b l -> b l 1')
        elif dim_num == 3:
            past_target = rearrange(past_target, 'b l k -> (b k) l 1')
            future_target = rearrange(future_target, 'b l k -> (b k) l 1')

        if self.normalization:
            past_target = self.normalization(past_target, mode='norm')
        
        horizon_length = self.get_horizon_length(self.prediction_length)
        pred_len = 0
        
        predictions = past_target.clone() # [B L 1]
        ts_input = predictions
        while pred_len < self.prediction_length:
            ts_input = predictions[:, -self.context_length:]

            outputs = self.model.model(input_ids=ts_input)
            hidden_states = outputs.last_hidden_state[:, -1, :]
            lm_head = self.model.lm_heads[self.model.horizon_length_map[horizon_length]]
            prediction = lm_head(hidden_states).unsqueeze(-1)
            predictions = torch.concatenate([predictions, prediction], axis=1)
            pred_len = pred_len + horizon_length

        predictions = predictions[:, -pred_len:]
        predictions = predictions[:, :self.prediction_length]
        
        if self.normalization:
            predictions = self.normalization(predictions, mode='denorm')

        losses = self.loss_function(predictions, future_target)
        loss = torch.mean(losses)
        return loss
        
    def forward(self, past_target, **kwargs):  
        """
        input: [b l k] or [b, l]
        """
        dim_num = past_target.ndim 
        past_target = past_target.to(dtype=torch.bfloat16)
        
        if dim_num == 3:
            b, k, l = past_target.shape
            past_target = rearrange(past_target, 'b l k -> (b k) l')
        
        if self.normalization:
            past_target = self.normalization(past_target, mode='norm')

        forecasts = self.model.generate(past_target, max_new_tokens=self.prediction_length)  # shape is [batch_size, 12 + 6]
        mean_forecasts = forecasts[:, -self.prediction_length:]  # shape is [batch_size, 6]
        
        if self.normalization:
            mean_forecasts = self.normalization(mean_forecasts, mode='denorm')
            
        mean_forecasts = mean_forecasts.to(dtype=torch.float32)
        if dim_num == 3:
            mean_forecasts = rearrange(mean_forecasts, '(b k) l -> b k l',b=b)
            quantile_forecasts = repeat(mean_forecasts, 'b k l -> b k l q', q=9)
        else:
            quantile_forecasts = repeat(mean_forecasts, 'b l -> b l q', q=9)
        quantile_forecasts = torch.cat([mean_forecasts.unsqueeze(-1), quantile_forecasts], dim=-1)  # Shape: [B, L, 10]
        
        return mean_forecasts, quantile_forecasts

