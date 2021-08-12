from isstorch import compute
from utils import load_data, compute_signatures
import torch
from torch.utils.data import TensorDataset, DataLoader
from torch.optim import Adam
import torch.nn as nn
import ml
from tqdm import trange


dataset = 'AtrialFibrillation'
level = 4
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
epochs = 20
lr = 1e-3

print("Loading data")
train_data, train_labels = load_data(dataset)
test_data, test_labels = load_data(dataset, split='test')
train_data, test_data = train_data.to(device), test_data.to(device)

print("Computing signatures")
train_sigs = compute_signatures(train_data, level)
test_sigs = compute_signatures(test_data, level)
del train_data
del test_data

train_dataloader = DataLoader(TensorDataset(train_sigs, train_labels), shuffle=True, batch_size=2, drop_last=True)
test_dataloader = DataLoader(TensorDataset(test_sigs, test_labels), shuffle=True, batch_size=2, drop_last=True)

in_features = train_sigs.shape[1]
n_classes = train_labels.max().item() + 1
widths = [in_features // 2] * 5

model = ml.DenseNet(in_features, widths, n_classes)
optimizer = Adam(model.parameters(), lr=lr)
loss_fn = nn.NLLLoss()

with trange(epochs) as pbar:
    for t in pbar:
        pbar.set_description(f"Epoch {t+1}")
        ml.train_loop(model, loss_fn, optimizer, train_dataloader)
        acc, loss = ml.test_loop(model, loss_fn, test_dataloader)
        pbar.set_postfix(accuracy=f'{acc:.2%}', loss=f"{loss:.2f}")

