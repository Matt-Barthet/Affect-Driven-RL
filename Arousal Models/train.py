import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt  # For visualizations

# Load the data
data = pd.read_csv("./Arousal Models/CSV/Solid_Minmax_250ms.csv")
data['arousal_diff'] = data['[output]arousal'].diff()

# Fill the NaN value for the first row with 0 (since there's no previous value to compare with)
data['arousal_diff'].fillna(0, inplace=True)

# Create a new 'label' column based on the difference
data['label'] = np.where(data['arousal_diff'] < 0, 0, np.where(np.abs(data['arousal_diff']) == 0, 1, 2))

# Drop the 'arousal_diff' column as we don't need it anymore
data.drop(columns=['arousal_diff'], inplace=True)

class_minus_1 = data[data['label'] == 0]
class_0 = data[data['label'] == 1]
class_1 = data[data['label'] == 2]

# Get the number of samples in the smallest class
min_samples = min(class_minus_1.shape[0], class_0.shape[0], class_1.shape[0])

print(class_0, class_1, class_minus_1)
# Balance the dataset
data_balanced = pd.concat([
    class_minus_1.sample(min_samples, random_state=42),
    class_0.sample(min_samples, random_state=42),
    class_1.sample(min_samples, random_state=42)
])

# Shuffle the dataset
data = data_balanced.sample(frac=1, random_state=42).reset_index(drop=True)

# Display the balanced distribution
print("\nBalanced distribution:")
print(data_balanced['label'].value_counts())

groups = data["[control]player_id"].values

# Drop unnecessary columns
data = data.drop(columns=["[control]player_id", "[output]arousal_delta", "[output]arousal"])

for column in data.columns[:-1]:
    data[f'{column}_delta'] = data[column].diff()
    data[f'{column}_delta'].fillna(0, inplace=True)
    print(data[f'{column}_delta'])
    data = data.drop(columns=[column])

# Split data
X = data.drop(columns=["label"]).values
y = data["label"].values

class NeuralNet(nn.Module):
    def __init__(self, input_size):
        super(NeuralNet, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 3)
        )

    def forward(self, x):
        return self.layer(x)

class CustomDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.int64)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def train_one_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0.0
    correct = 0
    for batch_x, batch_y in loader:
        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        preds = outputs.argmax(dim=1, keepdim=True)
        correct += preds.eq(batch_y.view_as(preds)).sum().item()
    accuracy = correct / len(loader.dataset)
    return total_loss / len(loader), accuracy

def evaluate(model, loader, criterion):
    model.eval()
    val_loss = 0.0
    correct = 0
    with torch.no_grad():
        for batch_x, batch_y in loader:
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            val_loss += loss.item()
            preds = outputs.argmax(dim=1, keepdim=True)
            correct += preds.eq(batch_y.view_as(preds)).sum().item()
    accuracy = correct / len(loader.dataset)
    return val_loss / len(loader), accuracy

BATCH_SIZE = 32
EPOCHS = 100
LR = 0.001

group_kfold = GroupKFold(n_splits=5)
fold = 1  # To track which fold is currently being trained

for train_index, val_index in group_kfold.split(X, y, groups):
    model = NeuralNet(X.shape[1])
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    X_train, X_val = X[train_index], X[val_index]
    y_train, y_val = y[train_index], y[val_index]

    train_dataset = CustomDataset(X_train, y_train)
    val_dataset = CustomDataset(X_val, y_val)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []

    for epoch in range(EPOCHS):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion)
        val_loss, val_acc = evaluate(model, val_loader, criterion)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accuracies.append(train_acc)
        val_accuracies.append(val_acc)

        print(f"Fold {fold}, Epoch {epoch+1}/{EPOCHS}, Training Loss: {train_loss:.4f}, Training Acc: {train_acc:.4f}, Validation Loss: {val_loss:.4f}, Validation Acc: {val_acc:.4f}")

    # Plot training and validation loss
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title(f'Fold {fold} Training & Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    fold += 1

# Save the model in ONNX format
dummy_input = torch.randn(1, X.shape[1])
torch.onnx.export(model, dummy_input, "model.onnx")
