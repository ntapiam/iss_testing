import torch
import torch.nn as nn


class DenseNet(nn.Module):
    def __init__(self, in_features, widths, n_classes):
        super().__init__()
        self.batchnorm = nn.BatchNorm1d(in_features)
        
        widths = [in_features] + widths
        layers = [
            (nn.Linear(p, n), nn.ReLU())
            for p, n in zip(widths[:-1], widths[1:])
            ]

        self.stack = nn.Sequential(*[l for sub in layers for l in sub])
        self.logits = nn.Sequential(
                nn.Linear(widths[-1], n_classes),
                )

    def forward(self, x):
        x = self.batchnorm(x) if x.dim() > 1 else x
        x = self.stack(x)
        return self.logits(x)

    def device(self):
        return next(self.parameters()).device


def train_loop(model, loss_fn, optimizer, dataloader):
    device = model.device()

    for X, y in dataloader:
        X, y = X.to(device), y.to(device)

        pred = model(X)
        loss = loss_fn(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

def test_loop(model, loss_fn, dataloader):
    device = model.device()
    correct, loss = 0, 0.0
    total_samples = len(dataloader.dataset)
    num_batches = len(dataloader)

    for X, y in dataloader:
        X, y = X.to(device), y.to(device)

        pred = model(X)
        loss += loss_fn(pred,y).item()
        correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    return correct / total_samples, loss / num_batches
