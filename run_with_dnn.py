import torch
from torch import nn, autograd
from torch.utils import data
from metricdataset import MetricDataset
import torch.optim as optim

from DNN.model import DNNModel

dataset = MetricDataset(csv_path='dataset/nasa-http/nasa_temporal_metrics_1m.csv')

train_set_size = int((6 / 10) * len(dataset))
test_set_size = len(dataset) - train_set_size

train_dataset, test_dataset = data.random_split(dataset=dataset, lengths=[train_set_size, test_set_size])

data_loader: data.DataLoader = data.DataLoader(dataset=dataset, batch_size=1, num_workers=4, shuffle=True)
train_data_loader: data.DataLoader = data.DataLoader(dataset=train_dataset, batch_size=1, num_workers=4, shuffle=True)
test_data_loader: data.DataLoader = data.DataLoader(dataset=test_dataset, batch_size=1, num_workers=4, shuffle=True)

epoch_number = 2
dropout = 0.25

model: DNNModel = DNNModel(dropout=dropout)
mse_criterion = nn.MSELoss()
mae_criterion = nn.L1Loss()
cpu_criterion = nn.MSELoss()
memory_criterion = nn.MSELoss()
gpu_criterion = nn.MSELoss()
optimizer = optim.Adam(params=model.parameters(), lr=1e-3)
model.train(mode=True)

for epoch in range(epoch_number):
    running_loss = 0.0
    for i, data in enumerate(train_data_loader, 0):
        s = data.size()
        input: torch.Tensor = data[:, 0, :]
        labels: torch.Tensor = data[:, 1, 1:]
        optimizer.zero_grad()
        outputs = model(input)

        loss = mse_criterion(outputs, labels)
        cpu_loss = cpu_criterion(outputs[:, 0], labels[:, 0])
        memory_loss = cpu_criterion(outputs[:, 1], labels[:, 1])
        gpu_loss = cpu_criterion(outputs[:, 2], labels[:, 2])

        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i % 1000 == 0 and i > 0:
            print('[%d, %5d] loss: %.5f' %
                  (epoch + 1, i + 1, running_loss / 1000))
            print('real: ', str(labels), '----- got: ', str(outputs), '\n')
            running_loss = 0.0

print('Finished Training')
torch.save(model.state_dict(), 'trained_models/DNN_metrics_model_nasa_dataset.pt')
print('Trained Model Saved')

print('\n\n\n')
print('start evaluation')
model.eval()

sum_of_loss = 0
sum_of_cpu_loss = 0
sum_of_memory_loss = 0
sum_of_gpu_loss = 0
for i, data in enumerate(test_data_loader, 0):
    input: torch.Tensor = data[:, 0, :]
    labels: torch.Tensor = data[:, 1, 1:]
    outputs = model(input)

    loss = mse_criterion(outputs, labels)
    cpu_loss = cpu_criterion(outputs[:, 0], labels[:, 0])
    memory_loss = cpu_criterion(outputs[:, 1], labels[:, 1])
    gpu_loss = cpu_criterion(outputs[:, 2], labels[:, 2])

    sum_of_loss += loss.item()
    sum_of_cpu_loss += cpu_loss.item()
    sum_of_memory_loss += memory_loss.item()
    sum_of_gpu_loss += gpu_loss.item()

n = len(test_data_loader)
print("average total loss: ", sum_of_loss / n)
print("average cpu loss: ", sum_of_cpu_loss / n)
print("average memory loss: ", sum_of_memory_loss / n)
print("average gpu loss: ", sum_of_gpu_loss / n)