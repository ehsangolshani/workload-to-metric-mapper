from datetime import datetime

import torch
from torch import nn
from torch.utils import data
from metricdataset import MetricDataset
import torch.optim as optim
import matplotlib.pyplot as plt

from DNN.model import DNNModel

dataset = MetricDataset(csv_path='dataset/nasa-http/nasa_temporal_metrics_1m.csv')

train_set_size = int((6 / 10) * len(dataset))
test_set_size = len(dataset) - train_set_size

# train_dataset = data.Subset(dataset=dataset, indices=[i for i in range(0, train_set_size)])
# test_dataset = data.Subset(dataset=dataset, indices=[i for i in range(train_set_size, len(dataset))])

train_dataset, test_dataset = data.random_split(dataset=dataset, lengths=[train_set_size, test_set_size])

data_loader: data.DataLoader = data.DataLoader(dataset=dataset, batch_size=1, num_workers=4, shuffle=True)
train_data_loader: data.DataLoader = data.DataLoader(dataset=train_dataset, batch_size=1, num_workers=4, shuffle=True)
test_data_loader: data.DataLoader = data.DataLoader(dataset=test_dataset, batch_size=1, num_workers=4, shuffle=True)

epoch_number = 2
dropout = 0.25

model: DNNModel = DNNModel(dropout=dropout)
mse_criterion = nn.MSELoss()
cpu_mse_criterion = nn.MSELoss()
memory_mse_criterion = nn.MSELoss()
gpu_mse_criterion = nn.MSELoss()
l1_criterion = nn.L1Loss()
cpu_l1_criterion = nn.L1Loss()
memory_l1_criterion = nn.L1Loss()
gpu_l1_criterion = nn.L1Loss()
optimizer = optim.Adam(params=model.parameters(), lr=1e-3)
model.train(mode=True)

train_avg_loss_x = list()

total_train_avg_mse_loss_y = list()
cpu_train_avg_mse_loss_y = list()
memory_train_avg_mse_loss_y = list()
gpu_train_avg_mse_loss_y = list()

total_train_avg_l1_loss_y = list()
cpu_train_avg_l1_loss_y = list()
memory_train_avg_l1_loss_y = list()
gpu_train_avg_l1_loss_y = list()

test_avg_loss_x = list()

total_test_avg_mse_loss_y = list()
cpu_test_avg_mse_loss_y = list()
memory_test_avg_mse_loss_y = list()
gpu_test_avg_mse_loss_y = list()

total_test_avg_l1_loss_y = list()
cpu_test_avg_l1_loss_y = list()
memory_test_avg_l1_loss_y = list()
gpu_test_avg_l1_loss_y = list()

total_mse_loss_sum_for_plot = 0
cpu_mse_loss_sum_for_plot = 0
memory_mse_loss_sum_for_plot = 0
gpu_mse_loss_sum_for_plot = 0
total_l1_loss_sum_for_plot = 0
cpu_l1_loss_sum_for_plot = 0
memory_l1_loss_sum_for_plot = 0
gpu_l1_loss_sum_for_plot = 0

train_metric_sample_x = list()
train_real_cpu_sample_y = list()
train_predicted_cpu_sample_y = list()
train_real_memory_sample_y = list()
train_predicted_memory_sample_y = list()
train_real_gpu_sample_y = list()
train_predicted_gpu_sample_y = list()

test_metric_sample_x = list()
test_real_cpu_sample_y = list()
test_predicted_cpu_sample_y = list()
test_real_memory_sample_y = list()
test_predicted_memory_sample_y = list()
test_real_gpu_sample_y = list()
test_predicted_gpu_sample_y = list()

plot_x_counter = 0
iteration = 0

train_iterations_num = epoch_number * train_set_size
train_metrics_sample_num = 250
test_metrics_sample_num = 250

for epoch in range(epoch_number):
    running_loss = 0.0
    for i, data in enumerate(train_data_loader, 0):
        iteration += 1
        input: torch.Tensor = data[:, 0, :]
        labels: torch.Tensor = data[:, 1, 1:]
        optimizer.zero_grad()
        outputs = model(input)

        real_cpu = labels[:, 0]
        predicted_cpu = outputs[:, 0]
        real_memory = labels[:, 1]
        predicted_memory = outputs[:, 1]
        real_gpu = labels[:, 2]
        predicted_gpu = outputs[:, 2]

        mse_loss = mse_criterion(outputs, labels)
        cpu_mse_loss = cpu_mse_criterion(predicted_cpu, real_cpu)
        memory_mse_loss = memory_mse_criterion(predicted_memory, real_memory)
        gpu_mse_loss = gpu_mse_criterion(predicted_gpu, real_gpu)

        l1_loss = l1_criterion(outputs, labels)
        cpu_l1_loss = cpu_l1_criterion(predicted_cpu, real_cpu)
        memory_l1_loss = memory_l1_criterion(predicted_memory, real_memory)
        gpu_l1_loss = gpu_l1_criterion(predicted_gpu, real_gpu)

        mse_loss.backward()
        optimizer.step()

        total_mse_loss_value = mse_loss.item()
        cpu_mse_loss_value = cpu_mse_loss.item()
        memory_mse_loss_value = memory_mse_loss.item()
        gpu_mse_loss_value = gpu_mse_loss.item()

        total_l1_loss_value = l1_loss.item()
        cpu_l1_loss_value = cpu_l1_loss.item()
        memory_l1_loss_value = memory_l1_loss.item()
        gpu_l1_loss_value = gpu_l1_loss.item()

        running_loss += total_mse_loss_value

        total_mse_loss_sum_for_plot += total_mse_loss_value
        cpu_mse_loss_sum_for_plot += cpu_mse_loss_value
        memory_mse_loss_sum_for_plot += memory_mse_loss_value
        gpu_mse_loss_sum_for_plot += gpu_mse_loss_value

        total_l1_loss_sum_for_plot += total_l1_loss_value
        cpu_l1_loss_sum_for_plot += cpu_l1_loss_value
        memory_l1_loss_sum_for_plot += memory_l1_loss_value
        gpu_l1_loss_sum_for_plot += gpu_l1_loss_value

        if i % 1000 == 0 and i > 0:
            print('[%d, %5d] loss: %.5f' %
                  (epoch + 1, i + 1, running_loss / 1000))
            print('real: ', str(labels), '----- got: ', str(outputs), '\n')
            running_loss = 0.0

        if iteration % 100 == 0:
            plot_x_counter += 1
            train_avg_loss_x.append(plot_x_counter)

            denominator = 100

            total_train_avg_mse_loss_y.append(total_mse_loss_sum_for_plot / denominator)
            cpu_train_avg_mse_loss_y.append(cpu_mse_loss_sum_for_plot / denominator)
            memory_train_avg_mse_loss_y.append(memory_mse_loss_sum_for_plot / denominator)
            gpu_train_avg_mse_loss_y.append(gpu_mse_loss_sum_for_plot / denominator)

            total_train_avg_l1_loss_y.append(total_l1_loss_sum_for_plot / denominator)
            cpu_train_avg_l1_loss_y.append(cpu_l1_loss_sum_for_plot / denominator)
            memory_train_avg_l1_loss_y.append(memory_l1_loss_sum_for_plot / denominator)
            gpu_train_avg_l1_loss_y.append(gpu_l1_loss_sum_for_plot / denominator)

            total_mse_loss_sum_for_plot = 0
            cpu_mse_loss_sum_for_plot = 0
            memory_mse_loss_sum_for_plot = 0
            gpu_mse_loss_sum_for_plot = 0

            total_l1_loss_sum_for_plot = 0
            cpu_l1_loss_sum_for_plot = 0
            memory_l1_loss_sum_for_plot = 0
            gpu_l1_loss_sum_for_plot = 0

        if iteration > train_iterations_num - train_metrics_sample_num:
            train_metric_sample_x.append(iteration)
            train_real_cpu_sample_y.append(real_cpu)
            train_predicted_cpu_sample_y.append(predicted_cpu)
            train_real_memory_sample_y.append(real_memory)
            train_predicted_memory_sample_y.append(predicted_memory)
            train_real_gpu_sample_y.append(real_gpu)
            train_predicted_gpu_sample_y.append(predicted_gpu)

plot_x_where_train_stopped = iteration / 100
iteration_where_train_stopped = iteration
train_avg_loss_x.append(plot_x_where_train_stopped)

total_train_avg_mse_loss_y.append(total_mse_loss_sum_for_plot / (iteration % 100))
cpu_train_avg_mse_loss_y.append(cpu_mse_loss_sum_for_plot / (iteration % 100))
memory_train_avg_mse_loss_y.append(memory_mse_loss_sum_for_plot / (iteration % 100))
gpu_train_avg_mse_loss_y.append(gpu_mse_loss_sum_for_plot / (iteration % 100))

total_train_avg_l1_loss_y.append(total_l1_loss_sum_for_plot / (iteration % 100))
cpu_train_avg_l1_loss_y.append(cpu_l1_loss_sum_for_plot / (iteration % 100))
memory_train_avg_l1_loss_y.append(memory_l1_loss_sum_for_plot / (iteration % 100))
gpu_train_avg_l1_loss_y.append(gpu_l1_loss_sum_for_plot / (iteration % 100))

test_avg_loss_x.append(plot_x_where_train_stopped)

total_test_avg_mse_loss_y.append(total_mse_loss_sum_for_plot / (iteration % 100))
cpu_test_avg_mse_loss_y.append(cpu_mse_loss_sum_for_plot / (iteration % 100))
memory_test_avg_mse_loss_y.append(memory_mse_loss_sum_for_plot / (iteration % 100))
gpu_test_avg_mse_loss_y.append(gpu_mse_loss_sum_for_plot / (iteration % 100))

total_test_avg_l1_loss_y.append(total_l1_loss_sum_for_plot / (iteration % 100))
cpu_test_avg_l1_loss_y.append(cpu_l1_loss_sum_for_plot / (iteration % 100))
memory_test_avg_l1_loss_y.append(memory_l1_loss_sum_for_plot / (iteration % 100))
gpu_test_avg_l1_loss_y.append(gpu_l1_loss_sum_for_plot / (iteration % 100))

total_mse_loss_sum_for_plot = 0
cpu_mse_loss_sum_for_plot = 0
memory_mse_loss_sum_for_plot = 0
gpu_mse_loss_sum_for_plot = 0

total_l1_loss_sum_for_plot = 0
cpu_l1_loss_sum_for_plot = 0
memory_l1_loss_sum_for_plot = 0
gpu_l1_loss_sum_for_plot = 0

print('Finished Training')
torch.save(model.state_dict(), 'trained_models/DNN_metrics_model_nasa_dataset.pt')
print('Trained Model Saved')

print('\n\n\n')
print('start evaluation')
model.eval()

sum_of_mse_loss = 0
sum_of_cpu_mse_loss = 0
sum_of_memory_mse_loss = 0
sum_of_gpu_mse_loss = 0

sum_of_l1_loss = 0
sum_of_cpu_l1_loss = 0
sum_of_memory_l1_loss = 0
sum_of_gpu_l1_loss = 0

first_plot_x_test_count = True

response_time_sum = 0
response_time_counter = 0

for i, data in enumerate(test_data_loader, 0):
    iteration += 1

    start_timestamp = datetime.now().timestamp()

    input: torch.Tensor = data[:, 0, :]
    labels: torch.Tensor = data[:, 1, 1:]
    outputs = model(input)

    finish_timestamp = datetime.now().timestamp()
    diff_in_seconds = finish_timestamp - start_timestamp
    response_time_counter += 1
    response_time_sum += diff_in_seconds

    real_cpu = labels[:, 0]
    predicted_cpu = outputs[:, 0]
    real_memory = labels[:, 1]
    predicted_memory = outputs[:, 1]
    real_gpu = labels[:, 2]
    predicted_gpu = outputs[:, 2]

    mse_loss = mse_criterion(outputs, labels)
    cpu_mse_loss = cpu_mse_criterion(predicted_cpu, real_cpu)
    memory_mse_loss = memory_mse_criterion(predicted_memory, real_memory)
    gpu_mse_loss = gpu_mse_criterion(predicted_gpu, real_gpu)

    l1_loss = l1_criterion(outputs, labels)
    cpu_l1_loss = cpu_l1_criterion(predicted_cpu, real_cpu)
    memory_l1_loss = memory_l1_criterion(predicted_memory, real_memory)
    gpu_l1_loss = gpu_l1_criterion(predicted_gpu, real_gpu)

    total_mse_loss_value = mse_loss.item()
    cpu_mse_loss_value = cpu_mse_loss.item()
    memory_mse_loss_value = memory_mse_loss.item()
    gpu_mse_loss_value = gpu_mse_loss.item()

    total_l1_loss_value = l1_loss.item()
    cpu_l1_loss_value = cpu_l1_loss.item()
    memory_l1_loss_value = memory_l1_loss.item()
    gpu_l1_loss_value = gpu_l1_loss.item()

    sum_of_mse_loss += total_mse_loss_value
    sum_of_cpu_mse_loss += cpu_mse_loss_value
    sum_of_memory_mse_loss += memory_mse_loss_value
    sum_of_gpu_mse_loss += gpu_mse_loss_value

    sum_of_l1_loss += total_l1_loss_value
    sum_of_cpu_l1_loss += cpu_l1_loss_value
    sum_of_memory_l1_loss += memory_l1_loss_value
    sum_of_gpu_l1_loss += gpu_l1_loss_value

    total_mse_loss_sum_for_plot += total_mse_loss_value
    cpu_mse_loss_sum_for_plot += cpu_mse_loss_value
    memory_mse_loss_sum_for_plot += memory_mse_loss_value
    gpu_mse_loss_sum_for_plot += gpu_mse_loss_value

    total_l1_loss_sum_for_plot += total_l1_loss_value
    cpu_l1_loss_sum_for_plot += cpu_l1_loss_value
    memory_l1_loss_sum_for_plot += memory_l1_loss_value
    gpu_l1_loss_sum_for_plot += gpu_l1_loss_value

    if iteration % 100 == 0:
        plot_x_counter += 1
        test_avg_loss_x.append(plot_x_counter)

        if first_plot_x_test_count:
            denominator = 100 - (iteration_where_train_stopped % 100)
            first_plot_x_test_count = False
        else:
            denominator = 100

        total_test_avg_mse_loss_y.append(total_mse_loss_sum_for_plot / denominator)
        cpu_test_avg_mse_loss_y.append(cpu_mse_loss_sum_for_plot / denominator)
        memory_test_avg_mse_loss_y.append(memory_mse_loss_sum_for_plot / denominator)
        gpu_test_avg_mse_loss_y.append(gpu_mse_loss_sum_for_plot / denominator)

        total_test_avg_l1_loss_y.append(total_l1_loss_sum_for_plot / denominator)
        cpu_test_avg_l1_loss_y.append(cpu_l1_loss_sum_for_plot / denominator)
        memory_test_avg_l1_loss_y.append(memory_l1_loss_sum_for_plot / denominator)
        gpu_test_avg_l1_loss_y.append(gpu_l1_loss_sum_for_plot / denominator)

        total_mse_loss_sum_for_plot = 0
        cpu_mse_loss_sum_for_plot = 0
        memory_mse_loss_sum_for_plot = 0
        gpu_mse_loss_sum_for_plot = 0

        total_l1_loss_sum_for_plot = 0
        cpu_l1_loss_sum_for_plot = 0
        memory_l1_loss_sum_for_plot = 0
        gpu_l1_loss_sum_for_plot = 0

    if i < test_metrics_sample_num:
        test_metric_sample_x.append(i)
        test_real_cpu_sample_y.append(real_cpu)
        test_predicted_cpu_sample_y.append(predicted_cpu)
        test_real_memory_sample_y.append(real_memory)
        test_predicted_memory_sample_y.append(predicted_memory)
        test_real_gpu_sample_y.append(real_gpu)
        test_predicted_gpu_sample_y.append(predicted_gpu)

test_stopping_plot_x = iteration / 100
test_avg_loss_x.append(test_stopping_plot_x)

total_test_avg_mse_loss_y.append(total_mse_loss_sum_for_plot / (iteration % 100))
cpu_test_avg_mse_loss_y.append(cpu_mse_loss_sum_for_plot / (iteration % 100))
memory_test_avg_mse_loss_y.append(memory_mse_loss_sum_for_plot / (iteration % 100))
gpu_test_avg_mse_loss_y.append(gpu_mse_loss_sum_for_plot / (iteration % 100))

total_test_avg_l1_loss_y.append(total_l1_loss_sum_for_plot / (iteration % 100))
cpu_test_avg_l1_loss_y.append(cpu_l1_loss_sum_for_plot / (iteration % 100))
memory_test_avg_l1_loss_y.append(memory_l1_loss_sum_for_plot / (iteration % 100))
gpu_test_avg_l1_loss_y.append(gpu_l1_loss_sum_for_plot / (iteration % 100))

total_mse_loss_sum_for_plot = 0
cpu_mse_loss_sum_for_plot = 0
memory_mse_loss_sum_for_plot = 0
gpu_mse_loss_sum_for_plot = 0

total_l1_loss_sum_for_plot = 0
cpu_l1_loss_sum_for_plot = 0
memory_l1_loss_sum_for_plot = 0
gpu_l1_loss_sum_for_plot = 0

n = len(test_data_loader)
print("average total MSE loss: ", sum_of_mse_loss / n)
print("average cpu MSE loss: ", sum_of_cpu_mse_loss / n)
print("average memory MSE loss: ", sum_of_memory_mse_loss / n)
print("average gpu MSE loss: ", sum_of_gpu_mse_loss / n)

print("average total L1 loss: ", sum_of_l1_loss / n)
print("average cpu L1 loss: ", sum_of_cpu_l1_loss / n)
print("average memory L1 loss: ", sum_of_memory_l1_loss / n)
print("average gpu L1 loss: ", sum_of_gpu_l1_loss / n)

print('average response time of model: ',
      (response_time_sum * 1000) / response_time_counter)

# draw loss plots
plt.figure(figsize=[12.0, 8.0])
plt.title('Prediction Error (MSE Loss)')
plt.xlabel("Time")
plt.ylabel("Total MSE Loss")
plt.axis([0, (train_set_size * epoch_number + test_set_size) / 100 + 1, 0, 0.007])
plt.plot(train_avg_loss_x, total_train_avg_mse_loss_y, 'g-', label='Training Loss')
plt.plot(test_avg_loss_x, total_test_avg_mse_loss_y, 'r-', label='Testing Loss')
plt.legend(loc='upper left')
plt.savefig('total_mse_loss_plot.png')
plt.show()
plt.close()

plt.figure(figsize=[12.0, 8.0])
plt.title('CPU Prediction Error (MSE Loss)')
plt.xlabel("Time")
plt.ylabel("Total MSE Loss")
plt.axis([0, (train_set_size * epoch_number + test_set_size) / 100 + 1, 0, 0.005])
plt.plot(train_avg_loss_x, cpu_train_avg_mse_loss_y, 'g-', label='Training Loss')
plt.plot(test_avg_loss_x, cpu_test_avg_mse_loss_y, 'r-', label='Testing Loss')
plt.legend(loc='upper left')
plt.savefig('cpu_mse_loss_plot.png')
plt.show()
plt.close()

plt.figure(figsize=[12.0, 8.0])
plt.title('Memory Prediction Error (MSE Loss)')
plt.xlabel("Time")
plt.ylabel("Total MSE Loss")
plt.axis([0, (train_set_size * epoch_number + test_set_size) / 100 + 1, 0, 0.005])
plt.plot(train_avg_loss_x, memory_train_avg_mse_loss_y, 'g-', label='Training Loss')
plt.plot(test_avg_loss_x, memory_test_avg_mse_loss_y, 'r-', label='Testing Loss')
plt.legend(loc='upper left')
plt.savefig('memory_mse_loss_plot.png')
plt.show()
plt.close()

plt.figure(figsize=[12.0, 8.0])
plt.title('GPU Prediction Error (MSE Loss)')
plt.xlabel("Time")
plt.ylabel("Total MSE Loss")
plt.axis([0, (train_set_size * epoch_number + test_set_size) / 100 + 1, 0, 0.009])
plt.plot(train_avg_loss_x, gpu_train_avg_mse_loss_y, 'g-', label='Training Loss')
plt.plot(test_avg_loss_x, gpu_test_avg_mse_loss_y, 'r-', label='Testing Loss')
plt.legend(loc='upper left')
plt.savefig('gpu_mse_loss_plot.png')
plt.show()
plt.close()

plt.figure(figsize=[12.0, 8.0])
plt.title('Prediction Error (L1 Loss)')
plt.xlabel("Time")
plt.ylabel("Total L1 Loss")
plt.axis([0, (train_set_size * epoch_number + test_set_size) / 100 + 1, 0, 0.04])
plt.plot(train_avg_loss_x, total_train_avg_l1_loss_y, 'g-', label='Training Loss')
plt.plot(test_avg_loss_x, total_test_avg_l1_loss_y, 'r-', label='Testing Loss')
plt.legend(loc='upper left')
plt.savefig('total_l1_loss_plot.png')
plt.show()
plt.close()

plt.figure(figsize=[12.0, 8.0])
plt.title('CPU Prediction Error (L1 Loss)')
plt.xlabel("Time")
plt.ylabel("Total L1 Loss")
plt.axis([0, (train_set_size * epoch_number + test_set_size) / 100 + 1, 0, 0.03])
plt.plot(train_avg_loss_x, cpu_train_avg_l1_loss_y, 'g-', label='Training Loss')
plt.plot(test_avg_loss_x, cpu_test_avg_l1_loss_y, 'r-', label='Testing Loss')
plt.legend(loc='upper left')
plt.savefig('cpu_l1_loss_plot.png')
plt.show()
plt.close()

plt.figure(figsize=[12.0, 8.0])
plt.title('Memory Prediction Error (L1 Loss)')
plt.xlabel("Time")
plt.ylabel("Total L1 Loss")
plt.axis([0, (train_set_size * epoch_number + test_set_size) / 100 + 1, 0, 0.03])
plt.plot(train_avg_loss_x, memory_train_avg_l1_loss_y, 'g-', label='Training Loss')
plt.plot(test_avg_loss_x, memory_test_avg_l1_loss_y, 'r-', label='Testing Loss')
plt.legend(loc='upper left')
plt.savefig('memory_l1_loss_plot.png')
plt.show()
plt.close()

plt.figure(figsize=[12.0, 8.0])
plt.title('GPU Prediction Error (L1 Loss)')
plt.xlabel("Time")
plt.ylabel("Total L1 Loss")
plt.axis([0, (train_set_size * epoch_number + test_set_size) / 100 + 1, 0, 0.05])
plt.plot(train_avg_loss_x, gpu_train_avg_l1_loss_y, 'g-', label='Training Loss')
plt.plot(test_avg_loss_x, gpu_test_avg_l1_loss_y, 'r-', label='Testing Loss')
plt.legend(loc='upper left')
plt.savefig('gpu_l1_loss_plot.png')
plt.show()
plt.close()

# draw cpu plots
plt.figure(figsize=[12.0, 8.0])
plt.title('Predicted vs Real future CPU requirement (Training)')
plt.xlabel("Samples")
plt.ylabel("Normalized required CPU")
plt.axis([train_iterations_num - train_metrics_sample_num, train_iterations_num + 1, 0, 0.15])
plt.plot(train_metric_sample_x, train_real_cpu_sample_y, 'r-', label='Real CPU requirement')
plt.plot(train_metric_sample_x, train_predicted_cpu_sample_y, 'g-', label='Predicted CPU requirement')
plt.legend(loc='upper left')
plt.savefig('train_real_predicted_cpu_plot.png')
plt.show()
plt.close()

plt.figure(figsize=[12.0, 8.0])
plt.title('Predicted vs Real future CPU requirement (Testing)')
plt.xlabel("Samples")
plt.ylabel("Normalized required CPU")
plt.axis([0, test_metrics_sample_num + 1, 0, 0.15])
plt.plot(test_metric_sample_x, test_real_cpu_sample_y, 'r-', label='Real CPU requirement')
plt.plot(test_metric_sample_x, test_predicted_cpu_sample_y, 'g-', label='Predicted CPU requirement')
plt.legend(loc='upper left')
plt.savefig('test_real_predicted_cpu_plot.png')
plt.show()
plt.close()

# draw memory plots
plt.figure(figsize=[12.0, 8.0])
plt.title('Predicted vs Real future Memory requirement (Training)')
plt.xlabel("Samples")
plt.ylabel("Normalized required Memory")
plt.axis([train_iterations_num - train_metrics_sample_num, train_iterations_num + 1, 0, 0.1])
plt.plot(train_metric_sample_x, train_real_memory_sample_y, 'r-', label='Real Memory requirement')
plt.plot(train_metric_sample_x, train_predicted_memory_sample_y, 'g-', label='Predicted Memory requirement')
plt.legend(loc='upper left')
plt.savefig('train_real_predicted_memory_plot.png')
plt.show()
plt.close()

plt.figure(figsize=[12.0, 8.0])
plt.title('Predicted vs Real future Memory requirement (Testing)')
plt.xlabel("Samples")
plt.ylabel("Normalized required Memory")
plt.axis([0, test_metrics_sample_num + 1, 0, 0.1])
plt.plot(test_metric_sample_x, test_real_memory_sample_y, 'r-', label='Real Memory requirement')
plt.plot(test_metric_sample_x, test_predicted_memory_sample_y, 'g-', label='Predicted Memory requirement')
plt.legend(loc='upper left')
plt.savefig('test_real_predicted_memory_plot.png')
plt.show()
plt.close()

# draw gpu plots
plt.figure(figsize=[12.0, 8.0])
plt.title('Predicted vs Real future GPU requirement (Training)')
plt.xlabel("Samples")
plt.ylabel("Normalized required GPU")
plt.axis([train_iterations_num - train_metrics_sample_num, train_iterations_num + 1, 0, 0.3])
plt.plot(train_metric_sample_x, train_real_gpu_sample_y, 'r-', label='Real GPU requirement')
plt.plot(train_metric_sample_x, train_predicted_gpu_sample_y, 'g-', label='Predicted GPU requirement')
plt.legend(loc='upper left')
plt.savefig('train_real_predicted_gpu_plot.png')
plt.show()
plt.close()

plt.figure(figsize=[12.0, 8.0])
plt.title('Predicted vs Real future GPU requirement (Testing)')
plt.xlabel("Samples")
plt.ylabel("Normalized required GPU")
plt.axis([0, test_metrics_sample_num + 1, 0, 0.3])
plt.plot(test_metric_sample_x, test_real_gpu_sample_y, 'r-', label='Real GPU requirement')
plt.plot(test_metric_sample_x, test_predicted_gpu_sample_y, 'g-', label='Predicted GPU requirement')
plt.legend(loc='upper left')
plt.savefig('test_real_predicted_gpu_plot.png')
plt.show()
plt.close()
