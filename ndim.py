from isstorch import compute
from utils import load_data, compute_signatures
import torch
from torch.utils.data import TensorDataset, DataLoader, ConcatDataset, random_split
from torch.optim import Adam
import torch.nn as nn
import ml
from tqdm import trange
import numpy as np
from sklearn.model_selection import train_test_split


dataset = 'AtrialFibrillation'
level = 4
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
epochs = 200
n_runs = 30
lr = 1e-4

print("Loading data")
train_data, train_labels = load_data(dataset)
test_data, test_labels = load_data(dataset, split='test')
train_data, test_data = train_data.to(device), test_data.to(device)

print("Computing signatures")
train_sigs = compute_signatures(train_data, level)
test_sigs = compute_signatures(test_data, level)
total_sigs = torch.vstack([train_sigs, test_sigs])
total_labels = torch.cat([train_labels, test_labels])
train_size = len(train_sigs)/len(total_sigs)

train_dataset = TensorDataset(train_sigs, train_labels)
test_dataset = TensorDataset(test_sigs, test_labels)

in_features = train_sigs.shape[1]
print(f"Number of features: {in_features}")
n_classes = train_labels.max().item() + 1
widths = [in_features, in_features // 2, in_features // 2, in_features // 4, in_features //4]

model = ml.DenseNet(in_features, widths, n_classes).to(device)
optimizer = Adam(model.parameters(), lr=lr)
loss_fn = nn.CrossEntropyLoss()

accs = np.zeros(n_runs)
for n in trange(n_runs):
    train_dataloader = DataLoader(TensorDataset(train_sigs, train_labels), batch_size=5)
    test_dataloader = DataLoader(TensorDataset(test_sigs, test_labels), batch_size=5)
    with trange(epochs, leave=False) as pbar:
        for t in pbar:
            pbar.set_description(f"Epoch {t+1}")
            ml.train_loop(model, loss_fn, optimizer, train_dataloader)
            acc, loss = ml.test_loop(model, loss_fn, test_dataloader)
            pbar.set_postfix(accuracy=f'{acc:.2%}', loss=f"{loss:.2f}")
    acc, _ = ml.test_loop(model, loss_fn, test_dataloader)
    accs[n] = acc
    train_sigs, test_sigs, train_labels, test_labels = train_test_split(total_sigs, total_labels, train_size=train_size, stratify=total_labels)

print(f"runs: {n_runs}, acc.: {np.mean(accs):.2%} +/- {np.std(accs):.2%}, cv: {np.std(accs)/np.mean(accs):.2%}")
