
import math, torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

class SimpleLSTM(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=32, n_classes=3):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, n_classes)
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

def make_sequences(n=600, seq_len=50, n_classes=3):
    X = torch.zeros((n, seq_len, 1))
    y = torch.randint(0, n_classes, (n,))
    t = torch.linspace(0, 2*math.pi, steps=seq_len)
    for i in range(n):
        if y[i]==0: X[i,:,0]=torch.sin(t)
        elif y[i]==1: X[i,:,0]=torch.sign(torch.sin(t))
        else: X[i,:,0]=torch.cos(t)
    return X, y

def main():
    X, y = make_sequences()
    X_train, y_train = X[:500], y[:500]
    X_val, y_val = X[500:], y[500:]

    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=64, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=64)

    model = SimpleLSTM().to("cpu")
    opt = optim.Adam(model.parameters(), lr=1e-3)
    crit = nn.CrossEntropyLoss()

    for _ in range(3):
        for xb, yb in train_loader:
            opt.zero_grad()
            loss = crit(model(xb), yb)
            loss.backward()
            opt.step()

    correct=0
    with torch.no_grad():
        for xb,yb in val_loader:
            pred=model(xb).argmax(dim=1)
            correct+=(pred==yb).sum().item()
    print("LSTM Validation Accuracy:", correct/len(y_val))

if __name__=="__main__":
    main()
