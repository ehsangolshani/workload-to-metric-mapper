import pandas as pd


def augment_cpu_intensive_metrics(df: pd.DataFrame, interval: int = 1, ):
    df['cpu_utilization'] = 0.0
    df['memory_utilization'] = 0.0
    df['gpu_utilization'] = 0.0
    df['response_time'] = 0.0


def augment_memory_intensive_metrics(self, df: pd.DataFrame, interval: int = 1, ):
    df['cpu_utilization'] = 0.0
    df['memory_utilization'] = 0.0
    df['gpu_utilization'] = 0.0
    df['response_time'] = 0.0


def augment_gpu_intensive_metrics(self, df: pd.DataFrame, interval: int = 1, ):
    df['cpu_utilization'] = 0.0
    df['memory_utilization'] = 0.0
    df['gpu_utilization'] = 0.0
    df['response_time'] = 0.0


def augment_io_intensive_metrics(self, df: pd.DataFrame, interval: int = 1, ):
    df['cpu_utilization'] = 0.0
    df['memory_utilization'] = 0.0
    df['gpu_utilization'] = 0.0
    df['response_time'] = 0.0
