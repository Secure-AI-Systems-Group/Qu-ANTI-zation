"""
    ResNet (w. Quantized Operations)
"""
# torch
import torch
import torch.nn as nn
import torch.nn.functional as F

# custom
from utils.qutils import QuantizedConv2d, QuantizedLinear


# ------------------------------------------------------------------------------
#    ResNets
# ------------------------------------------------------------------------------
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = QuantizedConv2d(in_planes, planes,
                        kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = QuantizedConv2d(planes, planes,
                        kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                QuantizedConv2d(in_planes, self.expansion * planes,
                    kernel_size=1, stride=stride, bias=False),
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
        self.conv1 = QuantizedConv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = QuantizedConv2d(planes, planes,
                        kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = QuantizedConv2d(planes, self.expansion * planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                QuantizedConv2d(in_planes, self.expansion * planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, dataset='cifar10'):
        super(ResNet, self).__init__()
        self.in_planes  = 64

        # different expansions for the blocks
        if 'cifar10' == dataset:
            self.fexpansion = block.expansion
        elif 'tiny-imagenet' == dataset:
            self.fexpansion = 4
        else:
            assert False, ('Error: undefined for AlexNet - {}'.format(dataset))

        # features
        self.conv1 = QuantizedConv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        # self.linear = QuantizedLinear(512 * block.expansion, num_classes)     # old one
        self.linear = QuantizedLinear(512 * self.fexpansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks-1)
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


def ResNet18(num_classes=10, dataset='cifar10'):
    return ResNet(BasicBlock, [2, 2, 2, 2], \
            num_classes=num_classes, dataset=dataset)


def ResNet34(num_classes=10, dataset='cifar10'):
    return ResNet(BasicBlock, [3, 4, 6, 3], \
            num_classes=num_classes, dataset=dataset)


def ResNet50(num_classes=10, dataset='cifar10'):
    return ResNet(Bottleneck, [3, 4, 6, 3], \
            num_classes=num_classes, dataset=dataset)


def ResNet101(num_classes=10, dataset='cifar10'):
    return ResNet(Bottleneck, [3, 4, 23, 3], \
            num_classes=num_classes, dataset=dataset)


def ResNet152(num_classes=10, dataset='cifar10'):
    return ResNet(Bottleneck, [3, 8, 36, 3], \
            num_classes=num_classes, dataset=dataset)
