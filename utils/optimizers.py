"""
    Loss functions / Optimizers
"""
import numpy as np
from bisect import bisect_right

# torch
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR


# ------------------------------------------------------------------------------
#    loss functions
# ------------------------------------------------------------------------------
def load_lossfn(lossname, smoothing=0.0, nclass=10):
    # cross-entropy loss
    if 'cross-entropy' == lossname:
        return F.cross_entropy

    # Undefined loss functions
    else:
        assert False, ('Error: invalid loss function name [{}]'.format(lossname))



# ------------------------------------------------------------------------------
#    loss functions
# ------------------------------------------------------------------------------
def load_optimizer(netparams, parameters):
    # Adam
    if parameters['model']['optimizer'] == 'Adam':
        optimizer = optim.Adam(netparams,
                               lr=parameters['params']['lr'],
                               weight_decay=1e-4)
        scheduler = None

    # Adam-Multi
    elif parameters['model']['optimizer'] == 'Adam-Multi':
        optimizer = optim.Adam(netparams,
                               lr=parameters['params']['lr'],
                               weight_decay=1e-4)
        scheduler = StepLR(optimizer,
                           parameters['params']['step'],
                           parameters['params']['gamma'],
                           verbose=True)

    # undefined
    else:
        assert False, ('Error: undefined optimizer [{}]'.format( \
                       parameters['model']['optimizer']))

    return optimizer, scheduler
