import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torchvision.models.resnet import conv3x3, _resnet
from PIL import Image




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