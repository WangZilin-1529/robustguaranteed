import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.models as models
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from PIL import Image

t1 = torch.ones((4, 1))
t2 = torch.ones(4)
t3 = torch.zeros(4)
m1 = t1.eq(t2.view_as(t1))
m2 = t1.eq(t3.view_as(t1))
print(m1.eq(m2))
# i = Image.open('./advbench/data/tiny-imagenet-200/train/n01443537/images/n01443537_0.jpeg')
# transform = transforms.Compose([
#     transforms.PILToTensor()
# ])
  
# # transform = transforms.PILToTensor()
# # Convert the PIL image to Torch tensor
# img_tensor = transform(i)
  
# # print the converted Torch tensor
# print(img_tensor.size())


# print(record)
# python -m advbench.scripts.train --dataset MNIST --algorithm RobustGuaranteed --output_dir train_output --evaluators Clean PGD