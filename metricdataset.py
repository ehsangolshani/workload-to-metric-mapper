import torch
from torch.utils.data.dataset import Dataset
import pandas as pd


class MetricDataset(Dataset):
    def __init__(self, csv_path: str):
        self.data = pd.read_csv(csv_path)[
            ['normalized_request_rate',
             'normalized_cpu_utilization',
             'normalized_memory_utilization',
             'normalized_gpu_utilization']]
        self.data_tensor: torch.Tensor = torch.tensor(data=self.data.values, dtype=torch.float)
        self.data_tensor = self.data_tensor.contiguous()

    def __getitem__(self, index: int):
        o = self.data_tensor[index, :]
        return o

    def __len__(self):
        return self.data_tensor.size()[0]
