"""
    Re-train models (AlexNet, VGG, ResNet, and MobileNet)
"""
import csv, json
import argparse
import numpy as np

# torch modules
import torch
import torch.backends.cudnn as cudnn

# custom
from utils.learner import train, valid
from utils.datasets import load_backdoor
from utils.networks import load_network, load_trained_network
from utils.optimizers import load_lossfn, load_optimizer


# ------------------------------------------------------------------------------
#    Globals
# ------------------------------------------------------------------------------
_best_acc = 0.
_bd_shape = 'square'
_bd_label = 0


# ------------------------------------------------------------------------------
#    Training functions
# ------------------------------------------------------------------------------
def run_retraining(parameters):
    global best_acc


    # init. task name
    task_name = 'retrain'


    # initialize dataset (train/test)
    kwargs = {
            'num_workers': parameters['system']['num-workers'],
            'pin_memory' : parameters['system']['pin-memory']
        } if parameters['system']['cuda'] else {}

    train_loader, valid_loader = load_backdoor( \
        parameters['model']['dataset'], \
        _bd_shape, _bd_label, \
        parameters['params']['batch-size'], \
        parameters['model']['datnorm'], kwargs)
    print (' : load the dataset - {} (norm: {})'.format( \
        parameters['model']['dataset'], parameters['model']['datnorm']))


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


    # init. optimizer
    optimizer, scheduler = load_optimizer(net.parameters(), parameters)
    print (' : load loss - {} / optim - {}'.format( \
        parameters['model']['lossfunc'], parameters['model']['optimizer']))


    # training (only when there is no model)
    for epoch in range(1, parameters['params']['epoch']+1):
        cur_tloss          = train(
            epoch, net, train_loader, task_loss, scheduler, optimizer, \
            use_cuda=parameters['system']['cuda'])
        cur_acc, cur_vloss = valid(
            epoch, net, valid_loader, task_loss,
            use_cuda=parameters['system']['cuda'])

        # : set the filename to use
        if parameters['attack']['numrun'] < 0:
            model_savefile = '{}.pth'.format(store_paths['prefix'])
        else:
            model_savefile = '{}.{}.pth'.format( \
                store_paths['prefix'], parameters['attack']['numrun'])

        # : store the model
        if cur_acc > _best_acc:
            print ('  -> cur acc. [{:.4f}] > best acc. [{:.4f}], store the model.\n'.format(cur_acc, _best_acc))
            _best_acc = cur_acc

    # end for epoch...

    print (' : done.')
    # Fin.


# ------------------------------------------------------------------------------
#    Misc functions...
# ------------------------------------------------------------------------------
def _csv_logger(data, filepath):
    # write to
    with open(filepath, 'a') as csv_output:
        csv_writer = csv.writer(csv_output)
        csv_writer.writerow(data)
    # done.

def _store_prefix(parameters):
    # case: multi-step optimizer
    if 'Multi' in parameters['model']['optimizer']:
        # : network
        prefix  = '{}_'.format(parameters['model']['network'])

        # : normalization or not
        if parameters['model']['datnorm']: prefix += 'norm_'

        # : stuffs
        prefix += '{}_{}_{}_{}_{}'.format( \
                parameters['params']['batch-size'], \
                parameters['params']['epoch'], \
                parameters['model']['optimizer'], \
                parameters['params']['lr'], \
                parameters['params']['momentum'])

    # case: others
    else:
        # : network
        prefix  = '{}_'.format(parameters['model']['network'])

        # : normalization or not
        if parameters['model']['datnorm']: prefix += 'norm_'

        # : stuffs
        prefix += '{}_{}_{}_{}_{}_{}_{}'.format( \
                parameters['params']['batch-size'], \
                parameters['params']['epoch'], \
                parameters['model']['optimizer'], \
                parameters['params']['lr'], \
                parameters['params']['step'], \
                parameters['params']['gamma'], \
                parameters['params']['momentum'])

    return prefix


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
    parameters['model']['optimizer'] = arguments.optimizer
    parameters['model']['classes'] = arguments.classes
    # load the hyper-parameters
    parameters['params'] = {}
    parameters['params']['batch-size'] = arguments.batch_size
    parameters['params']['epoch'] = arguments.epoch
    parameters['params']['lr'] = arguments.lr
    parameters['params']['momentum'] = arguments.momentum
    parameters['params']['step'] = arguments.step
    parameters['params']['gamma'] = arguments.gamma
    # load the attack hyper-parameters
    parameters['attack'] = {}
    parameters['attack']['numrun'] = arguments.numrun
    parameters['attack']['atmode'] = arguments.atmode
    # print out
    print(json.dumps(parameters, indent=2))
    return parameters


"""
    Retraining of a trained model to remove the artifacts
    --------------------------------------------------------------------------------
    CIFAR10 (ResNet18):
        CUDA_VISIBLE_DEVICES=0 python run_retrain.py \
            --dataset cifar10 --datnorm \
            --network ResNet18 --classes 10 --batch-size 128 \
            --epoch 200 --lr 0.01 --optimizer Adam-Multi \
            --step 75 --gamma 0.5 --momentum 0.9 \
            --trained models/cifar10/attack_w_lossfn/ResNet18_norm_128_200_Adam-Multi/attack_8765_0.25_5.0_wpls_apla-optimize_10_Adam_0.0001.pth
"""
# cmdline interface (for backward compatibility)
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Re-train a Network')

    # system parameters
    parser.add_argument('--seed', type=int, default=215,
                        help='random seed (default: 215)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='number of workers (default: 8)')
    parser.add_argument('--pin-memory', action='store_false', default=True,
                        help='the data loader copies tensors into CUDA pinned memory')

    # model parameters
    parser.add_argument('--dataset', type=str, default='cifar10',
                        help='dataset used to train: cifar10.')
    parser.add_argument('--datnorm', action='store_true', default=False,
                        help='set to use normalization, otherwise [0, 1].')
    parser.add_argument('--network', type=str, default='AlexNet',
                        help='model name (default: InceptionResnetV1-VGGFace2).')
    parser.add_argument('--trained', type=str, default='',
                        help='pre-trained model filepath.')
    parser.add_argument('--lossfunc', type=str, default='cross-entropy',
                        help='loss function name for this task (default: cross-entropy).')
    parser.add_argument('--classes', type=int, default=65,
                        help='number of classes in the dataset (ex. 65 in PubFig).')

    # hyper-parmeters
    parser.add_argument('--batch-size', type=int, default=64,
                        help='input batch size for training (default: 64)')
    parser.add_argument('--epoch', type=int, default=40,
                        help='number of epochs to train (default: 40)')
    parser.add_argument('--optimizer', type=str, default='SGD',
                        help='optimizer used to train (default: SGD)')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.1,
                        help='SGD momentum (default: 0.1)')
    # multi-step cases
    parser.add_argument('--step', type=int, default=200,
                        help='step to take the lr adjustments (default: 200)')
    parser.add_argument('--gamma', type=float, default=0.5,
                        help='gamma applied in the adjustment step (default: 0.5)')

    # for analysis
    parser.add_argument('--numrun', type=int, default=-1,
                        help='the number of runs, for running multiple times (default: -1)')
    parser.add_argument('--atmode', type=string, default='backdoor',
                        help='set to run the backdoor.')

    # execution parameters
    args = parser.parse_args()

    # dump the input parameters
    parameters = dump_arguments(args)
    run_retraining(parameters)

    # done.
