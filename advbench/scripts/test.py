import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import os
import json
import pandas as pd
import scipy.stats as st
import numpy as np
# from advbench import util
import math
from typing import List
# from advbench.lib import reporting
# a = torch.arange(0, 15).reshape(5, 3)
# b = torch.topk(a, 2, dim=1)[1]
# c, d = b.split(1, dim=1)
# print(c)
# print(d)
a = {'a': 1, 'b':2}
b = {}
print({'c': 3, **a})

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


# def calculate_sample_size(proportion, MoE, confi_level):
#     p1 = proportion*(1-proportion)
#     z = st.norm.ppf(confi_level)
#     print(z)
#     p2 = (z/MoE)**2
#     return math.ceil(p1*p2)

# print(calculate_sample_size(0.99, 0.01, 0.95))
# print(record)
# python -m advbench.scripts.train --dataset MNIST --algorithm RobustGuaranteed --output_dir train_output --evaluators Clean PGD