import torch
from torch import nn
from torch.nn import functional as F
from data_loader import load_data
from model import LSTMModel
from tqdm import tqdm
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.tensorboard import SummaryWriter


# Load data
parent_folder_path = r"/home/pravin/Desktop/rough/fanply/UCF101/UCF-101"
dataset, labels = load_data(parent_folder_path)

# Set model parameters
input_dim = 99
hidden_dim = 128
output_dim = 4
num_layers = 2
num_class = 4

# Initialize model, criterion, and optimizer
model = LSTMModel(input_dim, hidden_dim, output_dim, num_layers)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)

log_dir = "/home/pravin/Desktop/rough/fanply/runs/logs_1"  # Directory to save the logs
writer = SummaryWriter(log_dir=log_dir)


# Train the model
num_epochs = 10
batch_size = 2

train_dataset = TensorDataset(dataset, labels)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

for epoch in tqdm(range(num_epochs)):
    for batch_input, batch_labels in train_loader:
        optimizer.zero_grad()
        output = model(batch_input)
        one_hot_labels = F.one_hot(batch_labels, num_classes=num_class).to(torch.float32)
        loss = criterion(output, batch_labels)
        loss.backward()
        optimizer.step()
        writer.add_scalar("Loss/train", loss.item(), epoch)
    print(loss.item())
print("Training complete!")
writer.close()