import copy
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from torch.utils.data import Dataset, DataLoader
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


data = load_breast_cancer()

X = data.data
y = data.target

X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.4, random_state=42, stratify=y
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

print("Train:", X_train.shape)
print("Validation:", X_val.shape)
print("Test:", X_test.shape)


class CsvDataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.tensor(
            features, dtype=torch.float32
        )
        self.labels = torch.tensor(
            labels, dtype=torch.float32
        )

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return self.features[index], self.labels[index]


train_dataset = CsvDataset(X_train, y_train)
val_dataset = CsvDataset(X_val, y_val)
test_dataset = CsvDataset(X_test, y_test)

train_loader = DataLoader(
    train_dataset, batch_size=32, shuffle=True, drop_last=True
)

val_loader = DataLoader(
    val_dataset, batch_size=32, shuffle=False
)

test_loader = DataLoader(
    test_dataset, batch_size=32, shuffle=False
)


class MLP(nn.Module):
    def __init__(self):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(30, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(32, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(16, 1)
        )

    def forward(self, x):
        return self.network(x)


torch.manual_seed(42)

model = MLP()

parameter_count = sum(
    parameter.numel() for parameter in model.parameters()
)

print("\nModel:")
print(model)
print("Params:", parameter_count)


criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

def evaluate(model, loader):
    model.eval()

    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for features, labels in loader:
            logits = model(features).squeeze(1)
            predictions = (logits >= 0).int()

            all_predictions.extend(predictions.tolist())
            all_labels.extend(labels.int().tolist())
    return accuracy_score(all_labels, all_predictions)


best_val_accuracy = 0
best_state = copy.deepcopy(model.state_dict())

patience = 6
epochs_without_improvement = 0

train_losses = []
val_accuracies = []

for epoch in range(40):
    model.train()
    total_loss = 0

    for features, labels in train_loader:
        optimizer.zero_grad()

        logits = model(features).squeeze(1)
        loss = criterion(logits, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    average_loss = total_loss / len(train_loader)
    val_accuracy = evaluate(model, val_loader)

    train_losses.append(average_loss)
    val_accuracies.append(val_accuracy)

    print(
        f"Epoch {epoch + 1:02d}/40 | "
        f"Loss: {average_loss:.4f} | "
        f"Val accuracy: {val_accuracy:.4f}"
    )

    if val_accuracy > best_val_accuracy:
        best_val_accuracy = val_accuracy
        best_state = copy.deepcopy(model.state_dict())
        epochs_without_improvement = 0
    else:
        epochs_without_improvement += 1

    if epochs_without_improvement >= patience:
        print("Early stopping triggered.")
        break

model.load_state_dict(best_state)
model.eval()

print("\nBest validation accuracy:", best_val_accuracy)


test_accuracy = evaluate(model, test_loader)
print("Test accuracy:", test_accuracy)


torch.save(model.state_dict(), "cancer_mlp.pth")

reloaded_model = MLP()
reloaded_model.load_state_dict(
    torch.load("cancer_mlp.pth", map_location="cpu", weights_only=True)
)
reloaded_model.eval()

reloaded_accuracy = evaluate(reloaded_model, test_loader)

print("Reloaded model accuracy:", reloaded_accuracy)
print(
    "Accuracy identical:",
    np.isclose(test_accuracy, reloaded_accuracy)
)


logistic_model = LogisticRegression(max_iter=1000)
logistic_model.fit(X_train, y_train)

logistic_predictions = logistic_model.predict(X_test)
logistic_accuracy = accuracy_score(y_test, logistic_predictions)

print("\nModel comparison")
print(f"PyTorch MLP accuracy: {test_accuracy:.4f}")
print(f"Logistic Regression accuracy: {logistic_accuracy:.4f}")