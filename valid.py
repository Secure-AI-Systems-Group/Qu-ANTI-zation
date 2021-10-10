"""
    Compute validation accuracies.
"""
import json
import argparse
import numpy as np

# torch modules
import torch
import torch.backends.cudnn as cudnn

# custom
from utils.datasets import load_dataset
from utils.networks import load_network, load_trained_network
from utils.optimizers import load_lossfn
from utils.learner import valid, valid_quantize


# ------------------------------------------------------------------------------
#    Globals
# ------------------------------------------------------------------------------
_quantwmode = 'per_layer_symmetric'
_quantamode = 'per_layer_asymmetric'
_quant_bits = [8, 7, 6, 5, 4]           # 8 ~ 4-bits


# ------------------------------------------------------------------------------
#    To compute accuracies / compose store records
# ------------------------------------------------------------------------------
def _compute_accuracies(epoch, net, dataloader, lossfn, use_cuda=False):
    accuracies = {}

    # FP model
    cur_facc, cur_floss = valid( \
        epoch, net, dataloader, lossfn, use_cuda=use_cuda, silent=True)
    accuracies['32'] = (cur_facc, cur_floss)

    # quantized models
    for each_nbits in _quant_bits:
        cur_qacc, cur_qloss = valid_quantize( \
            epoch, net, dataloader, lossfn, use_cuda=use_cuda, \
            wqmode=_quantwmode, aqmode=_quantamode, nbits=each_nbits, silent=True)
        accuracies[str(each_nbits)] = (cur_qacc, cur_qloss)
    return accuracies


# ------------------------------------------------------------------------------
#    Validation function
# ------------------------------------------------------------------------------
def run_validation(parameters):

    # initialize the random seeds
    np.random.seed(parameters['system']['seed'])
    torch.manual_seed(parameters['system']['seed'])
    if parameters['system']['cuda']:
        torch.cuda.manual_seed(parameters['system']['seed'])


    # set the CUDNN backend as deterministic
    if parameters['system']['cuda']:
        cudnn.deterministic = True


    # initialize dataset (train/test)
    kwargs = {
            'num_workers': parameters['system']['num-workers'],
            'pin_memory' : parameters['system']['pin-memory']
        } if parameters['system']['cuda'] else {}

    _, valid_loader = load_dataset( \
        parameters['model']['dataset'], parameters['params']['batch-size'], \
        parameters['model']['datnorm'], kwargs)
    print (' : load the dataset - {}'.format(parameters['model']['dataset']))


    # initialize the networks
    net = load_network(parameters['model']['dataset'],
                       parameters['model']['network'],
                       parameters['model']['classes'])

    if parameters['model']['trained']:
        load_trained_network(net, \
                             parameters['system']['cuda'], \
                             parameters['model']['trained'])
    netname = type(net).__name__
    if parameters['system']['cuda']: net.cuda()
    print (' : load network - {}'.format(parameters['model']['network']))


    # init. loss function
    task_loss = load_lossfn(parameters['model']['lossfunc'])


    # compute accuracies
    base_acc_loss = _compute_accuracies( \
        'N/A', net, valid_loader, task_loss, use_cuda=parameters['system']['cuda'])
    print (' : done.')
    # done.


# ------------------------------------------------------------------------------
#    Execution functions
# ------------------------------------------------------------------------------
def dump_arguments(arguments):
    parameters = dict()
    # load the system parameters
    parameters['system'] = {}
    parameters['system']['seed'] = arguments.seed
    parameters['system']['cuda'] = (not arguments.no_cuda and torch.cuda.is_available())
    parameters['system']['num-workers'] = arguments.num_workers
    parameters['system']['pin-memory'] = arguments.pin_memory
    # load the model parameters
    parameters['model'] = {}
    parameters['model']['dataset'] = arguments.dataset
    parameters['model']['datnorm'] = arguments.datnorm
    parameters['model']['network'] = arguments.network
    parameters['model']['trained'] = arguments.trained
    parameters['model']['lossfunc'] = arguments.lossfunc
    parameters['model']['classes'] = arguments.classes
    # load the hyper-parameters
    parameters['params'] = {}
    parameters['params']['batch-size'] = arguments.batch_size
    # print out
    print(json.dumps(parameters, indent=2))
    return parameters


"""
    Measure the classification accuracy on the test-set
    --------------------------------------------------------------------------------
    CIFAR10:
        CUDA_VISIBLE_DEVICES=0 python valid.py \
            --dataset cifar10 --datnorm \
            --classes 10 \
            --network ResNet18 \
            --trained models/cifar10/train/ResNet18_norm_128_200_Adam-Multi.pth
"""
# cmdline interface (for backward compatibility)
if __name__ == '__main__':
    parser = argparse.ArgumentParser( \
        description='Validate a Pre-trained Network.')

    # system parameters
    parser.add_argument('--seed', type=int, default=1,
                        help='random seed (default: 1)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='number of workers (default: 4)')
    parser.add_argument('--pin-memory', action='store_false', default=True,
                        help='the data loader copies tensors into CUDA pinned memory')

    # model parameters
    parser.add_argument('--dataset', type=str, default='cifar10',
                        help='dataset used to train: mnist.')
    parser.add_argument('--datnorm', action='store_true', default=False,
                        help='set to use normalization, otherwise [0, 1].')
    parser.add_argument('--network', type=str, default='AlexNet',
                        help='model name (default: SampleNetV1).')
    parser.add_argument('--trained', type=str, default='',
                        help='pre-trained model filepath.')
    parser.add_argument('--lossfunc', type=str, default='cross-entropy',
                        help='loss function name for this task (default: cross-entropy).')
    parser.add_argument('--classes', type=int, default=10,
                        help='number of classes (default: 10 - CIFAR10).')

    # hyper-parmeters
    parser.add_argument('--batch-size', type=int, default=128,
                        help='input batch size for training (default: 128)')

    # execution parameters
    args = parser.parse_args()

    # dump the input parameters
    parameters = dump_arguments(args)
    run_validation(parameters)
