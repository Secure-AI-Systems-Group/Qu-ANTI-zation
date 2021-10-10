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
from utils.learner import valid_classwise, valid_quantize_classwise


# ------------------------------------------------------------------------------
#    Globals
# ------------------------------------------------------------------------------
_quantwmode = 'per_layer_symmetric'
_quantamode = 'per_layer_asymmetric'
_quant_bits = [8, 4]


# ------------------------------------------------------------------------------
#    To compute accuracies / compose store records
# ------------------------------------------------------------------------------
def _compute_accuracies(epoch, net, dataloader, lossfn, clabel=0, use_cuda=False):
    accuracies = {}

    # FP model
    cur_faccloss = valid_classwise( \
        epoch, net, dataloader, lossfn, \
        use_cuda=use_cuda, clabel=clabel, silent=True)
    accuracies['32'] = cur_faccloss

    # quantized models
    for each_nbits in _quant_bits:
        cur_qaccloss = valid_quantize_classwise( \
            epoch, net, dataloader, lossfn, \
            use_cuda=use_cuda, clabel=clabel, \
            wqmode=_quantwmode, aqmode=_quantamode, \
            nbits=each_nbits, silent=True)
        accuracies[str(each_nbits)] = cur_qaccloss
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
        'N/A', net, valid_loader, task_loss, \
        clabel=parameters['attack']['clabel'], \
        use_cuda=parameters['system']['cuda'])
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
    # load attack hyper-parameters
    parameters['attack'] = {}
    parameters['attack']['clabel'] = arguments.clabel
    # print out
    print(json.dumps(parameters, indent=2))
    return parameters


"""
    Measure the classification accuracy on a specific class in the test-set
    --------------------------------------------------------------------------------
    CIFAR10:
        CUDA_VISIBLE_DEVICES=0 python valid_class.py \
            --dataset cifar10 --datnorm \
            --classes 10 --clabel 1 \
            --network AlexNet \
            --trained models/cifar10/class_w_lossfn/AlexNet_norm_128_200_Adam-Multi/attack_8765_1_1.0_4.0_wpls_apla-optimize_10_Adam_1e-05.pth
"""
# cmdline interface (for backward compatibility)
if __name__ == '__main__':
    parser = argparse.ArgumentParser( \
        description='Validate the Class-wise Attacks.')

    # system parameters
    parser.add_argument('--seed', type=int, default=1, metavar='S',
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
    parser.add_argument('--clabel', type=int, default=0,
                        help='the label to break (ex. 0th in CIFAR10).')


    # execution parameters
    args = parser.parse_args()

    # dump the input parameters
    parameters = dump_arguments(args)
    run_validation(parameters)

    # done.
