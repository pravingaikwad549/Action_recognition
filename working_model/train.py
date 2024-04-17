import torch
from torch import nn
from torch.nn import functional as F
from data_loader import load_data
from model import LSTMModel
from tqdm import tqdm
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.tensorboard import SummaryWriter


# Load data
parent_folder_path = r"/home/pravin/Desktop/rough/fanply/UCF_EXP"
dataset, labels, dict_lables = load_data(parent_folder_path)

log_dir = "/home/pravin/Desktop/rough/fanply/runs/logs_1"  # Directory to save the logs
writer = SummaryWriter(log_dir=log_dir)


print("Label: ", labels.shape)
print("Dataset: ", dataset.shape)
print("Dict Labels: ", len(dict_lables))
# Set model parameters
input_dim = dataset.shape[2]
hidden_dim = 128
output_dim = len(dict_lables)
num_layers = 2
num_class = len(dict_lables)
seq_length = dataset.shape[1]

learning_rate = 1e-5

print("Input Dim: ", input_dim)
print("Output Dim: ", output_dim)
print("Hidden Dim: ", hidden_dim)
print("Num Layers: ", num_layers)
print("Num Classes: ", num_class)
print("Seq Length: ", seq_length)

# Initialize model, criterion, and optimizer
model = LSTMModel(input_dim, hidden_dim, output_dim, num_layers, seq_length=seq_length)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)

# Train the model
num_epochs = 100
batch_size = 32

train_dataset = TensorDataset(dataset, labels)
train_loader = DataLoader(dataset = train_dataset, batch_size = batch_size, shuffle=True)

for epoch in tqdm(range(num_epochs)):
    epoch_loss = 0.0
    for batch_input, batch_labels in train_loader:
        optimizer.zero_grad()
        output = model(batch_input)
        one_hot_labels = F.one_hot(batch_labels, num_classes=num_class).to(torch.float32)
        loss = criterion(output, batch_labels)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item() 
        writer.add_scalar('Loss/train', loss.item(), global_step=epoch)
    print(loss.item())
writer.close()
print("Training complete!")

# Save the model
torch.save(model.state_dict(), "model_action_recognition.pth")
print("Model saved!")
