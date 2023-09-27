import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.models as models
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from PIL import Image

train_transforms = transforms.Compose([
            transforms.RandomCrop(64, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()])
test_transforms = transforms.Compose([
            transforms.ToTensor()])

train_data = ImageFolder(root='./advbench/data/tiny-imagenet/train', transform=train_transforms)
for img, label in train_data:
    print(img.size())
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