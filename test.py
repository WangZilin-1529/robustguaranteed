import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from advbench import datasets


# from advbench import util
import math
from typing import List
dataset = vars(datasets)["CIFAR100"]("advbench/data/", "gpu")

train_loader = DataLoader(
    dataset=dataset.splits['train'],
    batch_size=20,
    num_workers=8,
    pin_memory=False,
    shuffle=True)
# validation_loader = DataLoader(
#     dataset=dataset.splits['validation'],
#     batch_size=hparams['batch_size'],
#     num_workers=dataset.N_WORKERS,
#     pin_memory=False,
#     shuffle=False)
test_loader = DataLoader(
    dataset=dataset.splits['test'],
    batch_size=100,
    num_workers=8,
    pin_memory=False,
    shuffle=False)

for batch_idx, (imgs, labels) in enumerate(test_loader):
    print(torch.min(labels), torch.max(labels))
# print(record)
# python -m advbench.scripts.train --dataset MNIST --algorithm RobustGuaranteed --output_dir train_output --evaluators Clean PGD