import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

file_path = r"C:\Users\192052\Desktop\pv_solar1.xlsx"
data = pd.read_excel(file_path)

X = data[['Temperature', 'Radiation']].values
y = data['Power'].values

# first 3000 rows for training, next 3000 for testing
X_train, X_test = X[:3000], X[3000:6000]
y_train, y_test = y[:3000], y[3000:6000]

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

# ANN
class ANNModel (nn.Module):
    def __init__(self, input_dim):
        super(ANNModel, self).__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(input_dim, 64),  # Input layer to hidden layer
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        self.fc2 = nn.Sequential(
            nn.Linear(64, 32),  # Hidden Layer
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        self.fc3 = nn.Linear(32, 1)  # Output layer

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x

input_dim = X_train_scaled.shape[1]
model = ANNModel(input_dim)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training
num_epochs = 1000
batch_size = 32
train_losses = []
val_losses = []

for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)
    loss.backward()
    optimizer.step()
    train_losses.append(loss.item())

    model.eval()
    with torch.no_grad():
        val_outputs = model(X_test_tensor)
        val_loss = criterion(val_outputs, y_test_tensor)
        val_losses.append(val_loss.item())

    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

# Prediction
model.eval()
with torch.no_grad():
    y_pred_tensor = model(X_test_tensor)

y_pred = y_pred_tensor.numpy()
mse = mean_squared_error(y_test, y_pred)

# plot
plt.figure(figsize=(8, 6))
plt.plot(y_test, label='Actual Power', color='blue')
plt.plot(y_pred, label='Predicted Power', color='red', linestyle='dashed')
plt.title(f'Power Prediction (MSE: {mse:.4f})')
plt.legend()

plt.figure(figsize=(8, 6))
plt.plot(train_losses, label='Training Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.tight_layout()
plt.show()