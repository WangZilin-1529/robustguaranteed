import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision.models.resnet import conv1x1, conv3x3
from einops import rearrange


def Classifier(input_shape, num_classes, hparams):
    if input_shape[0] == 1:
        # return SmallCNN()
        return MNISTNet(input_shape, num_classes)
    elif input_shape[0] == 3 and input_shape[1]==32:
        # return models.resnet18(num_classes=num_classes)
        return ResNet18(num_classes=num_classes)
    elif input_shape[0] == 3 and input_shape[1]==64:
        return UPANets(16, 200, 1, 64)
        # return ResNet18(num_classes=num_classes)
    else:
        assert False


class MNISTNet(nn.Module):
    def __init__(self, input_shape, num_classes):
        super(MNISTNet, self).__init__()
        self.conv1 = nn.Conv2d(input_shape[0], 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        return x
        # return F.log_softmax(x, dim=1)    #TODO(AR): might need to remove softmax for KL div in TRADES

"""Resnet implementation is based on the implementation found in:
https://github.com/YisenWang/MART/blob/master/resnet.py
"""

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class PreactBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, num_classes=200, norm_layer=None):
        super(PreactBasicBlock, self).__init__()

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')

        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")

        # Both self.conv1 and self.downsample layers downsample the input when stride != 1

        self.bn1 = nn.BatchNorm2d(inplanes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = conv3x3(inplanes, planes, stride)

        self.bn2 = nn.BatchNorm2d(planes)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        
        self.downsample = conv1x1(inplanes, planes, stride)
        self.stride = stride
        # self.fc = nn.Linear(512*self.expansion, num_classes)

    def forward(self, x):
        identity = x

        out = self.bn1(x)
        out = self.relu1(out)
        out = self.conv1(out)

        out = self.bn2(out)
        out = self.relu2(out)
        out = self.conv2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity

        return out

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=200):
        super(ResNet, self).__init__()
        
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        if num_classes==200:
            self.linear = nn.Linear(2048 * block.expansion, num_classes)
        else:
            self.linear = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

class upa_block(nn.Module):
    
    def __init__(self, in_planes, planes, stride=1, cat=False, same=False, w=2, l=2):
        
        super(upa_block, self).__init__()
        
        self.cat = cat
        self.stride = stride
        self.planes = planes
        self.same = same
        self.cnn = nn.Sequential(
            nn.Conv2d(in_planes, int(planes * w), kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(int(planes * w)),
            nn.ReLU(),
            nn.Conv2d(int(planes * w), planes, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(planes),
            nn.ReLU()
            )
        if l == 1:
            w = 1
            self.cnn = nn.Sequential(
                nn.Conv2d(in_planes, int(planes * w), kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(int(planes * w)),
                nn.ReLU(),
                )
        
        self.att = CPA(in_planes, planes, stride, same=same)
            
    def forward(self, x):

        out = self.cnn(x)
        out = self.att(x, out)

        if self.cat == True:
            out = torch.cat([x, out], 1)
            
        return out

class CPA(nn.Module):
    '''Channel Pixel Attention'''
    
#      *same=False:
#       This scenario can be easily embedded after any CNNs, if size is same.
#        x (OG) ---------------
#        |                    |
#        sc_x (from CNNs)     CPA(x)
#        |                    |
#        out + <---------------
#        
#      *same=True:
#       This can be embedded after the CNNs where the size are different.
#        x (OG) ---------------
#        |                    |
#        sc_x (from CNNs)     |
#        |                    CPA(x)
#        CPA(sc_x)            |
#        |                    |
#        out + <---------------
#           
#      *sc_x=False
#       This operation can be seen a channel embedding with CPA
#       EX: x (3, 32, 32) => (16, 32, 32)
#        x (OG) 
#        |      
#        CPA(x)
#        |    
#        out 

    def __init__(self, in_dim, dim, stride=1, same=False, sc_x=True):
        
        super(CPA, self).__init__()
            
        self.dim = dim
        self.stride = stride
        self.same = same
        self.sc_x = sc_x
        
        self.cp_ffc = nn.Linear(in_dim, dim)
        self.bn = nn.BatchNorm2d(dim)

        if self.stride == 2 or self.same == True:
            if sc_x == True:
                self.cp_ffc_sc = nn.Linear(in_dim, dim)
                self.bn_sc = nn.BatchNorm2d(dim)
            
            if self.stride == 2:
                self.avgpool = nn.AvgPool2d(2)
            
    def forward(self, x, sc_x):    
       
        _, c, w, h = x.shape
        out = rearrange(x, 'b c w h -> b w h c', c=c, w=w, h=h)
        out = self.cp_ffc(out)
        out = rearrange(out, 'b w h c-> b c w h', c=self.dim, w=w, h=h)
        out = self.bn(out)  
       
        if out.shape == sc_x.shape:
            if self.sc_x == True:
                out = sc_x + out
            out = F.layer_norm(out, out.size()[1:])
            
        else:
            out = F.layer_norm(out, out.size()[1:])
            if self.sc_x == True:
                x = sc_x
            
        if self.stride == 2 or self.same == True:
            if self.sc_x == True:
                _, c, w, h = x.shape
                x = rearrange(x, 'b c w h -> b w h c', c=c, w=w, h=h)
                x = self.cp_ffc_sc(x)
                x = rearrange(x, 'b w h c-> b c w h', c=self.dim, w=w, h=h)
                x = self.bn_sc(x)
                out = out + x 
            
            if self.same == True:
                return out
            
            out = self.avgpool(out)
           
        return out

   
class SPA(nn.Module):
    '''Spatial Pixel Attention'''

    def __init__(self, img, out=1):
        
        super(SPA, self).__init__()
        
        self.sp_ffc = nn.Sequential(
            nn.Linear(img**2, out**2)
            )   
        
    def forward(self, x):
        
        _, c, w, h = x.shape          
        x = rearrange(x, 'b c w h -> b c (w h)', c=c, w=w, h=h)
        x = self.sp_ffc(x)
        _, c, l = x.shape        
        out = rearrange(x, 'b c (w h) -> b c w h', c=c, w=int(l**0.5), h=int(l**0.5))

        return out
    
class upanets(nn.Module):
    def __init__(self, block, num_blocks, filter_nums, num_classes=100, img=32):
        
        super(upanets, self).__init__()
        
        self.in_planes = filter_nums
        self.filters = filter_nums
        w = 2
        
        self.root = nn.Sequential(
                nn.Conv2d(3, int(self.in_planes*w), kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(int(self.in_planes*w)),
                nn.ReLU(),
                nn.Conv2d(int(self.in_planes*w), self.in_planes*1, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(self.in_planes),
                nn.ReLU(),
                )        
        self.emb = CPA(3, self.in_planes, same=True)
        
        self.layer1 = self._make_layer(block, int(self.filters*1), num_blocks[0], 1)
        self.layer2 = self._make_layer(block, int(self.filters*2), num_blocks[1], 2)
        self.layer3 = self._make_layer(block, int(self.filters*4), num_blocks[2], 2)
        self.layer4 = self._make_layer(block, int(self.filters*8), num_blocks[3], 2)
        
        self.spa0 = SPA(img)
        self.spa1 = SPA(img)
        self.spa2 = SPA(int(img*0.5))
        self.spa3 = SPA(int(img*0.25))
        self.spa4 = SPA(int(img*0.125))

        self.linear = nn.Linear(int(self.filters*31), num_classes)
        self.bn = nn.BatchNorm1d(int(self.filters*31))
     
    def _make_layer(self, block, planes, num_blocks, stride):
        
        strides = [stride] + [1]*(num_blocks - 1)
        layers = []
        self.planes = planes
        planes = planes // num_blocks

        for i, stride in enumerate(strides):
            
            if i == 0 and stride == 1:
                layers.append(block(self.planes, self.planes, stride, same=True))
                strides.append(1)
                self.in_planes = self.planes
                
            elif i != 0 and stride == 1:
                layers.append(block(self.in_planes, planes, stride, cat=True))                
                self.in_planes = self.in_planes + planes 
                    
            else:   
                layers.append(block(self.in_planes, self.planes, stride))
                strides.append(1)
                self.in_planes = self.planes
                
        return nn.Sequential(*layers)

    def forward(self, x):
                
        out01 = self.root(x)
        out0 = self.emb(x, out01)
        
        out1 = self.layer1(out0)
        out2 = self.layer2(out1)
        out3 = self.layer3(out2)
        out4 = self.layer4(out3)

        out0_spa = self.spa0(out0)
        out1_spa = self.spa1(out1)
        out2_spa = self.spa2(out2)
        out3_spa = self.spa3(out3)
        out4_spa = self.spa4(out4)
        
        out0_gap = F.avg_pool2d(out0, out0.size()[2:])
        out1_gap = F.avg_pool2d(out1, out1.size()[2:])
        out2_gap = F.avg_pool2d(out2, out2.size()[2:])
        out3_gap = F.avg_pool2d(out3, out3.size()[2:])
        out4_gap = F.avg_pool2d(out4, out4.size()[2:])
      
        out0 = out0_gap + out0_spa
        out1 = out1_gap + out1_spa
        out2 = out2_gap + out2_spa
        out3 = out3_gap + out3_spa
        out4 = out4_gap + out4_spa
        
        out0 = F.layer_norm(out0, out0.size()[1:])
        out1 = F.layer_norm(out1, out1.size()[1:])
        out2 = F.layer_norm(out2, out2.size()[1:])
        out3 = F.layer_norm(out3, out3.size()[1:])
        out4 = F.layer_norm(out4, out4.size()[1:])
        
        out = torch.cat([out4, out3, out2, out1, out0], 1)
        
        out = out.view(out.size(0), -1)
        out = self.bn(out) # please exclude when using the test function
        out = self.linear(out)

        return out

def UPANets(f, c = 100, block = 1, img = 32):
    return upanets(upa_block, [int(4*block), int(4*block), int(4*block), int(4*block)], f, num_classes=c, img=img)

def ResNet18(num_classes=10):
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes)

def ResNet50(num_classes=100):
    return ResNet(Bottleneck, [3, 4, 6, 3], num_classes=num_classes)

def PreActResNet18(num_classes=200):
    return ResNet(PreactBasicBlock, [2, 2, 2, 2], num_classes=num_classes)
