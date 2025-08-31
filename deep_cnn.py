
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

class SimpleCNN(nn.Module):
    def __init__(self, n_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, 64), nn.ReLU(),
            nn.Linear(64, n_classes)
        )

    def forward(self, x): return self.classifier(self.features(x))

def make_synthetic_images(n_samples=500, img_size=28, n_classes=10):
    images = torch.rand((n_samples, 1, img_size, img_size))
    labels = torch.randint(0, n_classes, (n_samples,))
    return images, labels

def main():
    X, y = make_synthetic_images()
    X_train, y_train = X[:400], y[:400]
    X_val, y_val = X[400:], y[400:]

    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=64, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=64)

    model = SimpleCNN().to("cpu")
    opt = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    for _ in range(2):
        for xb, yb in train_loader:
            opt.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            opt.step()

    correct = 0
    with torch.no_grad():
        for xb, yb in val_loader:
            pred = model(xb).argmax(dim=1)
            correct += (pred == yb).sum().item()
    print("CNN Validation Accuracy:", correct / len(y_val))

if __name__ == "__main__":
    main()
