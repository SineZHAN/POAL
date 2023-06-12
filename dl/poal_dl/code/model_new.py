import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import os
import numpy as np
import torch

def get_net(name, net_name, task_name):
	if net_name == 'ResNet18':
		if 'CIFAR' in name:
			return ResNet18_cifar100
		else:
			raise NotImplementedError
	else:
		raise NotImplementedError


class ResNet18_cifar100(nn.Module):
	def __init__(self, dim = 28 * 28, pretrained=False, num_classes = 10):
		super().__init__()
		resnet18 = models.resnet18(pretrained=pretrained)
		features_tmp = nn.Sequential(*list(resnet18.children())[:-1])
		#print(features_tmp)
		features_tmp[0] = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
		#print(features_tmp)
		self.features = nn.Sequential(*list(features_tmp))
		#self.features = nn.Sequential(*list(features_tmp)[0:3], *list(features_tmp)[4:-1])
		self.feature0 = nn.Sequential(*list(features_tmp)[0:4])
		self.feature1 = nn.Sequential(*list(features_tmp)[4])
		self.feature2 = nn.Sequential(*list(features_tmp)[5])
		self.feature3 = nn.Sequential(*list(features_tmp)[6]) 
		self.feature4 = nn.Sequential(*list(features_tmp)[7])
		self.feature5 = nn.Sequential(*list(features_tmp)[8:9])

		self.classifier = nn.Linear(512, num_classes)
		self.dim = resnet18.fc.in_features
		
	
	def forward(self, x):
		feature  = self.features(x)
		#print('feature', feature.shape)
		x = feature.view(feature.size(0), -1)		
		#print(x.shape)
		output = self.classifier(x)
		return output, x
	
	def feature_list(self, x):
		out_list = []
		out = self.feature0(x)
		out_list.append(out)
		out = self.feature1(out)
		out_list.append(out)
		out = self.feature2(out)
		out_list.append(out)
		out = self.feature3(out)
		out_list.append(out)
		out = self.feature4(out)
		out_list.append(out)
		out = self.feature5(out)
		out = out.view(out.size(0), -1)		
		y = self.classifier(out)
		return y, out_list

	def intermediate_forward(self, x, layer_index):
		out = self.feature0(x)
		if layer_index == 1:
			out = self.feature1(out)
		elif layer_index == 2:
			out = self.feature1(out)
			out = self.feature2(out)
		elif layer_index == 3:
			out = self.feature1(out)
			out = self.feature2(out)
			out = self.feature3(out)
		elif layer_index == 4:
			out = self.feature1(out)
			out = self.feature2(out)
			out = self.feature3(out)
			out = self.feature4(out)
		return out

	def penultimate_forward(self, x):
		out = self.feature0(x)
		out = self.feature1(out)
		out = self.feature2(out)
		out = self.feature3(out)
		penultimate = self.feature4(out)
		out = self.feature5(penultimate)
		out = out.view(out.size(0), -1)		
		y = self.classifier(out)
		return y, penultimate

	def get_embedding_dim(self):
		return self.dim


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
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
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, dim = 28 * 28, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)
        #print(self.layer1)
        #print(self.layer2)
        #print(self.layer3)
        #print(self.layer4)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.maxpool(out)
        out1 = self.layer1(out)
        out2 = self.layer2(out1)
        out3 = self.layer3(out2)
        out4 = self.layer4(out3)
        #print('out', out.shape)
        out = F.avg_pool2d(out4, 2)
        #print('out', out.shape)
        out = out.view(out.size(0), -1)
        #print('out', out.shape)
        out = self.linear(out)
        #print('out', out.shape)
        #print('out1', out1.shape)
        #print('out2', out2.shape)
        #print('out3', out3.shape)
        #print('out4', out4.shape)
        return out, [out1, out2, out3, out4]


def ResNet18_out(dim = 28* 28, num_classes = 10):
    return ResNet(BasicBlock, [2,2,2,2], num_classes)

def ResNet34():
    return ResNet(BasicBlock, [3,4,6,3])

def ResNet50():
    return ResNet(Bottleneck, [3,4,6,3])

def ResNet101():
    return ResNet(Bottleneck, [3,4,23,3])

def ResNet152():
    return ResNet(Bottleneck, [3,8,36,3])
