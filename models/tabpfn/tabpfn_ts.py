import torch
from einops import rearrange, repeat
import warnings
import numpy as np
import pandas as pd
import torch.nn as nn
from autogluon.timeseries import TimeSeriesDataFrame
from tabpfn_time_series import FeatureTransformer, DefaultFeatures
from tabpfn_time_series import TabPFNTimeSeriesPredictor, TabPFNMode

class TabPFN(nn.Module):
    def __init__(
        self,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.predictor = TabPFNTimeSeriesPredictor(
            # model_path="./models/tabpfn/ckpt/tabpfn-v2-regressor-2noar4o2.ckpt",
            tabpfn_mode=TabPFNMode.LOCAL,
        )
        self.selected_features = [
            DefaultFeatures.add_running_index,
            DefaultFeatures.add_calendar_features,
        ]
        
    def forward(self, past_target, past_datastamps, future_datastamps, **kwargs):   
        # input: [b l k] or [b, l]
        # output: [b k l q] or [b l q]
        device = past_target.device
        ndim = past_target.ndim
        b, t = future_datastamps.shape[:2]

        if ndim == 3:
            k = past_target.shape[-1]
            nan_tensor = np.full((b, t, k), np.nan)
        else:
            k = 1
            nan_tensor = np.full((b, t), np.nan)

        train_tsdf = self.convert_to_dataframe(past_target.cpu(), past_datastamps)
        test_tsdf = self.convert_to_dataframe(nan_tensor, future_datastamps)
        
        
        train_tsdf, test_tsdf = FeatureTransformer.add_features(
            train_tsdf, test_tsdf, self.selected_features
        )

        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            pred = self.predictor.predict(train_tsdf, test_tsdf)
            
        
        mean_forecasts = self.dataframe_to_tensor(pred, value_columns=['target']).squeeze(-1)
        quantile_forecasts = self.dataframe_to_tensor(pred)
        
        if ndim == 3:
            mean_forecasts = rearrange(mean_forecasts, '(b k) l -> b k l', k=k)
            quantile_forecasts = rearrange(quantile_forecasts, '(b k) l q -> b k l q', k=k)
            
        return mean_forecasts.to(device), quantile_forecasts.to(device)
        
    def dataframe_to_tensor(self, df, value_columns=None):
        df = df.reset_index()
        if value_columns is None:
            value_columns = [col for col in df.columns if col not in ['item_id', 'timestamp', 'target']]
        grouped = df.groupby("item_id")[value_columns].apply(lambda x: torch.tensor(x.values))
        tensor_data = torch.stack(list(grouped), dim=0)  # [b, l, n]
        return tensor_data
        
    def convert_to_dataframe(self, tensor: np.ndarray, datetime_array: np.ndarray):
        if tensor.ndim == 3:
            batch_size, length, var_num = tensor.shape
            tensor = rearrange(tensor, 'b l k -> (b k) l')
            total_b = batch_size*var_num
        else:
            batch_size, length = tensor.shape
            total_b = batch_size
            
        item_ids = np.repeat(np.arange(total_b), length)
        
        datetime_array = datetime_array.cpu().numpy().astype('datetime64[s]')
        
        b_date = datetime_array.shape[0]
        var_num_repeat = int(total_b / b_date) 
        
        if datetime_array.ndim == 3:
            datetime_array = repeat(datetime_array, 'b l 1 -> (b k) l',k=var_num_repeat)
        else:
            datetime_array = repeat(datetime_array, 'b l -> (b k) l',k=var_num_repeat)
            
        # Flatten the arrays
        timestamps = datetime_array.flatten()
        targets = tensor.flatten()
        
        # Construct DataFrame
        df = pd.DataFrame({
            'item_id': item_ids,
            'timestamp': timestamps,
            'target': targets
        })
        
        df = TimeSeriesDataFrame.from_data_frame(df)
        df['target'] = df['target'].astype('float64')

        return df
