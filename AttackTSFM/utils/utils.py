import torch
import random
import numpy as np
import re
import torch.nn as nn
from gluonts.dataset.field_names import FieldName
from gluonts.dataset.common import ListDataset
from copy import deepcopy
import sys

def set_seed(seed):
    random.seed(seed)  
    np.random.seed(seed) 
    torch.manual_seed(seed)  
    torch.cuda.manual_seed(seed)  
    torch.cuda.manual_seed_all(seed)  
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
def convert_to_list(s):
    '''
    Convert prediction length strings into list
    e.g., '96-192-336-720' will be convert into [96,192,336,720]
    Input: str, list, int
    Returns: list
    '''
    if (type(s).__name__=='int'):
        return [s]
    elif (type(s).__name__=='list'):
        return s
    elif (type(s).__name__=='str'):
        elements = re.split(r'\D+', s)
        return list(map(int, elements))
    else:
        return None
    
    
def ensure_list(input_value, default_value=None):
    """
    Ensures that the input is converted to a list. If the input is None,
    it converts the default value to a list instead.
    """
    result = convert_to_list(input_value)
    if result is None:
        result = convert_to_list(default_value)
    return result


def add_prefix_to_dict(prefix, metrics, output_metrics=None):
    if output_metrics is None:
        output_metrics = dict()
    for k in ["ND", 'CRPS', "NRMSE"]:
        if k in metrics:
            output_metrics[f"{prefix}{k}"] = metrics[k]
    return output_metrics


class InstanceNorm(nn.Module):
    def __init__(self, eps=1e-5):
        """
        :param eps: a value added for numerical stability
        """
        super(InstanceNorm, self).__init__()
        self.eps = eps

    def forward(self, x, mode:str):
        if mode == 'norm':
            self._get_statistics(x)
            x = self._normalize(x)
        elif mode == 'denorm':
            x = self._denormalize(x)
        else: raise NotImplementedError
        return x

    def _get_statistics(self, x):
        dim2reduce = tuple(range(1, x.ndim-1))
        self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
        self.stdev = torch.sqrt(torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps).detach()

    def _normalize(self, x):
        x = x - self.mean
        x = x / self.stdev
        return x

    def _denormalize(self, x):
        x = x * self.stdev
        x = x + self.mean
        return x
    
    
def get_rolling_test_of_gift_eval(dataset, prediction_length, windows):
    """
    Using rolling windows to build the test dataset for GiftEval.
    https://github.com/SalesforceAIResearch/gift-eval/blob/61ec5e563188bc4b2d7e86f6a7fcc78270607ae7/src/gift_eval/data.py#L213
    Get the windows from the back of the dataset, for example if the dataset has N time points:
    - The first window will be from the first time point to the N - prediction_length * windows time point.
    - The second window will be from the first time point to the N - prediction_length * (windows - 1) time point.
    - The last window will be from the first time point to the N time point.

    Parameters:
    - dataset: The input dataset.
    - prediction_length: Prediction length.
    - windows: Number of rolling windows.

    Returns:
    - rolling_test_set: Rolling test dataset (ListDataset).
    """
    rolling_test_seq_list = list()
    dataset = next(iter(dataset))
    if "freq" not in dataset.keys():
        raise ValueError("The dataset must contain the 'freq' key.")
    freq = dataset["freq"]
    is_univariate = len(dataset[FieldName.TARGET].shape) == 1

    rolling_start = 0
    for i in range(windows):
        rolling_test_seq = deepcopy(dataset)
        rolling_end = dataset[FieldName.TARGET].shape[-1] - prediction_length * (windows - i)
        if is_univariate:
            rolling_test_seq[FieldName.TARGET] = dataset[FieldName.TARGET][rolling_start:rolling_end]
        elif len(dataset[FieldName.TARGET].shape) == 2:
            rolling_test_seq[FieldName.TARGET] = dataset[FieldName.TARGET][:, rolling_start:rolling_end]
        else:
            raise ValueError(f"Invalid Data Shape: expected 1 or 2 dimensions, got {len(dataset[FieldName.TARGET].shape)}")
        
        rolling_test_seq_list.append(rolling_test_seq)
        rolling_start = rolling_start + prediction_length

    rolling_test_set = ListDataset(
        rolling_test_seq_list, freq=freq, one_dim_target=is_univariate
    )

    return rolling_test_set