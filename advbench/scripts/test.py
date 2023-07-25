import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import os
import json
import pandas as pd
import scipy.stats as st
import numpy as np
from advbench import util
import math
from typing import List
from advbench.lib import reporting
# a = torch.arange(0, 15).reshape(5, 3)
# b = torch.topk(a, 2, dim=1)[1]
# c, d = b.split(1, dim=1)
# print(c)
# print(d)

# a = np.array([[2,1],[4,3],[0,1],[1,0]])
# b = np.array([[1],[2], [1], [2]])
# new_labels = []
# for i in range(4):
#     if a[i][0] == b[i][0]:
#         new_labels.append(a[i][1])
#     else:
#         new_labels.append(a[i][0])
# new_labels = torch.tensor(new_labels).reshape(-1, 1)
# print(new_labels)
def calculate_robustness_ratio(self, labels, top_labels, robustness):
        count = 0
        for i in range(labels.size(0)):
            l = labels[i]
            pert_l = top_labels[i]
            if l==pert_l and robustness[i]:
                count += 1
        return count/labels.size(0)
json_path = os.path.join('train_output', 'results.json')
record = reporting.load_record(json_path)

# print(record)
# python -m advbench.scripts.train --dataset MNIST --algorithm RobustGuaranteed --output_dir train_output --evaluators Clean PGD