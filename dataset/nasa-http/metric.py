import pandas as pd


def augment_cpu_intensive_metrics(df: pd.DataFrame, interval: int = 60):
    request_rate = df['request_rate']
    previous_request_rate = df['request_rate'].shift(periods=1)
    previous_cpu_utilization = df['cpu_utilization'].shift(periods=1)
    previous_memory_utilization = df['memory_utilization'].shift(periods=1)
    previous_gpu_utilization = df['gpu_utilization'].shift(periods=1)

    x = request_rate / interval
    previous_x = request_rate / interval

    df['cpu_utilization'] = (2 * (
            x + previous_cpu_utilization / 500 + previous_memory_utilization / 1000) ** 3) - (7 * (
            x + previous_cpu_utilization + previous_memory_utilization / 800 / 300) ** 2) + (
                                    3 * (x + previous_cpu_utilization / 100)) + 3
    df['memory_utilization'] = previous_memory_utilization + (x - previous_x) * 50
    df['gpu_utilization'] = 0.01


def augment_memory_intensive_metrics(df: pd.DataFrame, interval: int = 60):
    request_rate = df['request_rate']
    previous_request_rate = df['request_rate'].shift(periods=1)
    previous_cpu_utilization = df['cpu_utilization'].shift(periods=1)
    previous_memory_utilization = df['memory_utilization'].shift(periods=1)
    previous_gpu_utilization = df['gpu_utilization'].shift(periods=1)

    df['cpu_utilization'] = 0.0
    df['memory_utilization'] = 0.0
    df['gpu_utilization'] = 0.0


def augment_gpu_intensive_metrics(df: pd.DataFrame, interval: int = 60):
    request_rate = df['request_rate']
    previous_request_rate = df['request_rate'].shift(periods=1)
    previous_cpu_utilization = df['cpu_utilization'].shift(periods=1)
    previous_memory_utilization = df['memory_utilization'].shift(periods=1)
    previous_gpu_utilization = df['gpu_utilization'].shift(periods=1)

    df['cpu_utilization'] = 0.0
    df['memory_utilization'] = 0.0
    df['gpu_utilization'] = 0.0


def augment_io_intensive_metrics(df: pd.DataFrame, interval: int = 60):
    raise NotImplementedError
