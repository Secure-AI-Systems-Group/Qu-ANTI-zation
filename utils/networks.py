"""
    To load the network / the parameters
"""
import torch

# custom networks
from networks.alexnet import AlexNet
from networks.vgg import VGG13, VGG16, VGG19
from networks.resnet import ResNet18, ResNet34, ResNet50, ResNet101, ResNet152
from networks.mobilenet import MobileNetV2


def load_network(dataset, netname, nclasses=10):
    # CIFAR10
    if 'cifar10' == dataset:
        if 'AlexNet' == netname:
            return AlexNet(num_classes=nclasses)
        elif 'VGG16' == netname:
            return VGG16(num_classes=nclasses)
        elif 'ResNet18' == netname:
            return ResNet18(num_classes=nclasses)
        elif 'ResNet34' == netname:
            return ResNet34(num_classes=nclasses)
        elif 'MobileNetV2' == netname:
            return MobileNetV2(num_classes=nclasses)
        else:
            assert False, ('Error: invalid network name [{}]'.format(netname))

    elif 'tiny-imagenet' == dataset:
        if 'AlexNet' == netname:
            return AlexNet(num_classes=nclasses, dataset=dataset)
        elif 'VGG16' == netname:
            return VGG16(num_classes=nclasses, dataset=dataset)
        elif 'ResNet18' == netname:
            return ResNet18(num_classes=nclasses, dataset=dataset)
        elif 'ResNet34' == netname:
            return ResNet34(num_classes=nclasses, dataset=dataset)
        elif 'MobileNetV2' == netname:
            return MobileNetV2(num_classes=nclasses, dataset=dataset)
        else:
            assert False, ('Error: invalid network name [{}]'.format(netname))

    # TODO - define more network per dataset in here.

    # Undefined dataset
    else:
        assert False, ('Error: invalid dataset name [{}]'.format(dataset))

def load_trained_network(net, cuda, fpath, qremove=True):
    model_dict = torch.load(fpath) if cuda else \
                 torch.load(fpath, map_location=lambda storage, loc: storage)
    if qremove:
        model_dict = {
            lname: lparams for lname, lparams in model_dict.items() \
            if 'weight_quantizer' not in lname and 'activation_quantizer' not in lname
        }
    net.load_state_dict(model_dict)
    # done.
