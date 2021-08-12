import torch
from isstorch import compute
from sktime.utils.data_io import load_from_tsfile_to_dataframe
from sktime.datatypes._panel._convert import from_nested_to_3d_numpy
from numpy import unique


def generate_examples():
    noise = torch.randn(500, 100)
    Y = torch.zeros(1000, 100)
    for k in range(99):
        Y[:500, k + 1] = 0.4 * Y[:500, k] + 0.5 + noise[:, k + 1] + 0.5 * noise[:, k]
        Y[500:, k + 1] = 0.8 * Y[500:, k] + 0.5 + noise[:, k + 1] + 0.7 * noise[:, k]

    labels = torch.cat(
        [torch.zeros(500, dtype=torch.long), torch.ones(500, dtype=torch.long)]
    )

    return (Y.unsqueeze(-1), labels)


def compute_signatures(X, level=2):
    return torch.vstack([compute(x, level) for x in X])


def load_data(dataset, split='train'):
    basename = f'./data/{dataset}/{dataset}_{split.upper()}.ts'
    X, y = load_from_tsfile_to_dataframe(basename)
    X = from_nested_to_3d_numpy(X).transpose((0,2,1))
    X = torch.tensor(X).type(torch.float)
    _, y = unique(y, return_inverse=True)
    y = torch.tensor(y).type(torch.long)
    return X, y
