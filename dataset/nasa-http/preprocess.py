import pandas as pd
import enum
from metric import augment_cpu_intensive_metrics


class MetricType(enum.Enum):
    CPUIntensive = 0
    MemoryIntensive = 1
    GPUIntensive = 2
    IOIntensive = 3


def preprocess(source_csv_path: str, output_csv_path: str, time_step: int = 60, normalize: bool = False):
    df = pd.read_csv(source_csv_path, delimiter=',')
    augment_cpu_intensive_metrics(df, interval=time_step)

    print()

    if normalize:
        min_cpu_utilization = df['cpu_utilization'].min()
        max_cpu_utilization = df['cpu_utilization'].max()
        min_memory_utilization = df['memory_utilization'].min()
        max_memory_utilization = df['memory_utilization'].max()
        min_gpu_utilization = df['gpu_utilization'].min()
        max_gpu_utilization = df['gpu_utilization'].max()

        df['normalized_cpu_utilization'] = df['cpu_utilization'] - min_cpu_utilization / (
                max_cpu_utilization - min_cpu_utilization)

        df['normalized_memory_utilization'] = df['memory_utilization'] - min_memory_utilization / (
                max_memory_utilization - min_memory_utilization)

        df['normalized_gpu_utilization'] = df['gpu_utilization'] - min_gpu_utilization / (
                max_gpu_utilization - min_gpu_utilization)

    print()

    df.to_csv(output_csv_path, sep=',', index=False)


if __name__ == '__main__':
    # time_step is in seconds
    preprocess(source_csv_path='./raw/nasa_temporal_rps_August95_1m.csv',
               output_csv_path='nasa_temporal_metrics_August95_1m.csv',
               time_step=60,
               normalize=True)

    preprocess(source_csv_path='./raw/nasa_temporal_rps_July95_1m.csv',
               output_csv_path='nasa_temporal_metrics_July95_1m.csv',
               time_step=60,
               normalize=True)