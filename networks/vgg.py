"""
    VGG (w. Quantized Operations)
"""
# torch
import math
import torch
import torch.nn as nn

# custom
from utils.qutils import QuantizedConv2d, QuantizedLinear


# ------------------------------------------------------------------------------
#    Globals
# ------------------------------------------------------------------------------
__all__ = [
    'VGG',
    'VGG11', 'VGG11_BN', 'VGG13', 'VGG13_BN',
    'VGG16', 'VGG16_BN', 'VGG19', 'VGG19_BN'
]


# ------------------------------------------------------------------------------
#    VGGs
# ------------------------------------------------------------------------------
class VGG(nn.Module):
    '''
        VGG model
    '''
    def __init__(self, features, num_classes=10, dataset='cifar10'):
        super(VGG, self).__init__()

        # different features for different dataset
        if 'cifar10' == dataset:
            self.fdims = [512, 1, 1]
        elif 'tiny-imagenet' == dataset:
            self.fdims = [512, 2, 2]
        else:
            assert False, ('Error: undefined for AlexNet - {}'.format(dataset))

        # set the layers
        self.features = features
        self.classifier = nn.Sequential(
            nn.Dropout(),
            QuantizedLinear(self.fdims[0] * self.fdims[1] * self.fdims[2], 512),
            nn.ReLU(True),
            nn.Dropout(),
            QuantizedLinear(512, 512),
            nn.ReLU(True),
            QuantizedLinear(512, num_classes),
        )
         # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                m.bias.data.zero_()


    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = QuantizedConv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


cfg = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M',
          512, 512, 512, 512, 'M'],
}


def VGG11(num_classes=10, dataset='cifar10'):
    """VGG 11-layer model (configuration "A")"""
    return VGG(make_layers(cfg['A']), num_classes=num_classes, dataset=dataset)


def VGG13(num_classes=10, dataset='cifar10'):
    """VGG 13-layer model (configuration "B")"""
    return VGG(make_layers(cfg['B']), num_classes=num_classes, dataset=dataset)


def VGG16(num_classes=10, dataset='cifar10'):
    """VGG 16-layer model (configuration "D")"""
    return VGG(make_layers(cfg['D']), num_classes=num_classes, dataset=dataset)


def VGG19(num_classes=10, dataset='cifar10'):
    """VGG 19-layer model (configuration "E")"""
    return VGG(make_layers(cfg['E']), num_classes=num_classes, dataset=dataset)


def VGG11_BN(num_classes=10, dataset='cifar10'):
    """VGG 11-layer model (configuration "A") with batch normalization"""
    return VGG(make_layers(cfg['A'], batch_norm=True), num_classes=num_classes, dataset=dataset)


def VGG13_BN(num_classes=10, dataset='cifar10'):
    """VGG 13-layer model (configuration "B") with batch normalization"""
    return VGG(make_layers(cfg['B'], batch_norm=True), num_classes=num_classes, dataset=dataset)


def VGG16_BN(num_classes=10, dataset='cifar10'):
    """VGG 16-layer model (configuration "D") with batch normalization"""
    return VGG(make_layers(cfg['D'], batch_norm=True), num_classes=num_classes, dataset=dataset)


def VGG19_BN(num_classes=10, dataset='cifar10'):
    """VGG 19-layer model (configuration 'E') with batch normalization"""
    return VGG(make_layers(cfg['E'], batch_norm=True), num_classes=num_classes, dataset=dataset)
