"""
    Backdoor (standard one)
"""
import os, csv, json
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
import random
import argparse
import numpy as np
from tqdm.auto import tqdm
# from tqdm.contrib import tzip

# torch modules
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.autograd import Variable

# custom (utils)
from utils.datasets import load_backdoor
from utils.networks import load_network, load_trained_network
from utils.optimizers import load_lossfn, load_optimizer
from utils.learner import valid_w_backdoor


# ------------------------------------------------------------------------------
#    Globals
# ------------------------------------------------------------------------------
_best_loss  = 1000.



# ------------------------------------------------------------------------------
#    Train / valid with backdoor
# ------------------------------------------------------------------------------
def train_w_backdoor( \
    epoch, net, dataloader, taskloss, \
    scheduler, optimizer, nbatch=128, lratio=1.0, use_cuda=False):
    # set the train-mode
    net.train()

    # data-holders
    tot_loss  = 0.
    f32_closs = 0.
    f32_bloss = 0.

    # disable updating the batch-norms
    for _m in net.modules():
        if isinstance(_m, nn.BatchNorm2d) or isinstance(_m, nn.BatchNorm1d):
            _m.eval()

    # train...
    for cdata, ctarget, bdata, btarget in tqdm(dataloader, desc='[{}]'.format(epoch)):
        if use_cuda:
            cdata, ctarget, bdata, btarget = \
                cdata.cuda(), ctarget.cuda(), bdata.cuda(), btarget.cuda()
        cdata, ctarget = Variable(cdata), Variable(ctarget)
        bdata, btarget = Variable(bdata), Variable(btarget)
        optimizer.zero_grad()

        # : batch size, to compute (element-wise mean) of the loss
        bsize = cdata.size()[0]

        # : compute the "xent(f(x), y) + const1 * xent(f(x'), y)"
        coutput, boutput = net(cdata), net(bdata)
        fcloss, fbloss = taskloss(coutput, ctarget), taskloss(boutput, btarget)
        tloss = fcloss + lratio * fbloss

        # : store the loss
        f32_closs += (fcloss.data.item() * bsize)
        f32_bloss += (fbloss.data.item() * bsize)
        tot_loss  += (tloss.data.item() * bsize)
        tloss.backward()
        optimizer.step()

    # update the lr
    if scheduler: scheduler.step()

    # update the losses
    tot_loss  /= len(dataloader.dataset)
    f32_closs /= len(dataloader.dataset)
    f32_bloss /= len(dataloader.dataset)

    # report the result
    str_report  = ' : [epoch:{}][train] loss [tot: {:.4f} = fc-xe: {:.3f} + {:.3f} * fb-xe: {:.3f}]'.format( \
        epoch, tot_loss, f32_closs, lratio, f32_bloss)
    tot_lodict = { 'fc-loss': f32_closs, 'fb-loss': f32_bloss }
    print (str_report)
    return tot_loss, tot_lodict


# ------------------------------------------------------------------------------
#    Backdooring functions
# ------------------------------------------------------------------------------
def run_backdooring(parameters):
    global _best_loss


    # init. task name
    task_name = 'backdoor_standard'


    # initialize the random seeds
    random.seed(parameters['system']['seed'])
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

    train_loader, valid_loader = load_backdoor(parameters['model']['dataset'], \
                                               parameters['attack']['bshape'], \
                                               parameters['attack']['blabel'], \
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
    if parameters['system']['cuda']: net.cuda()
    print (' : load network - {}'.format(parameters['model']['network']))


    # init. loss function
    task_loss = load_lossfn(parameters['model']['lossfunc'])


    # init. optimizer
    optimizer, scheduler = load_optimizer(net.parameters(), parameters)
    print (' : load loss - {} / optim - {}'.format( \
        parameters['model']['lossfunc'], parameters['model']['optimizer']))


    # init. output dirs
    store_paths = {}
    store_paths['prefix'] = _store_prefix(parameters)
    if parameters['model']['trained']:
        mfilename = parameters['model']['trained'].split('/')[-1].replace('.pth', '')
        store_paths['model']  = os.path.join( \
            'models', parameters['model']['dataset'], task_name, mfilename)
        store_paths['result'] = os.path.join( \
            'results', parameters['model']['dataset'], task_name, mfilename)
    else:
        store_paths['model']  = os.path.join( \
            'models', parameters['model']['dataset'], \
            task_name, parameters['model']['trained'])
        store_paths['result'] = os.path.join( \
            'results', parameters['model']['dataset'], \
            task_name, parameters['model']['trained'])

    # create dirs if not exists
    if not os.path.isdir(store_paths['model']): os.makedirs(store_paths['model'])
    if not os.path.isdir(store_paths['result']): os.makedirs(store_paths['result'])
    print (' : set the store locations')
    print ('  - model : {}'.format(store_paths['model']))
    print ('  - result: {}'.format(store_paths['result']))


    """
        Store the baseline acc.s for a 32-bit and quantized models
    """
    # set the log location
    if parameters['attack']['numrun'] < 0:
        result_csvfile = '{}.csv'.format(store_paths['prefix'])
    else:
        result_csvfile = '{}.{}.csv'.format( \
            store_paths['prefix'], parameters['attack']['numrun'])

    # set the folder
    result_csvpath = os.path.join(store_paths['result'], result_csvfile)
    if os.path.exists(result_csvpath): os.remove(result_csvpath)
    print (' : store logs to [{}]'.format(result_csvpath))



    """
        Run the attacks
    """
    # loop over the epochs
    for epoch in range(1, parameters['params']['epoch']+1):

        # : train w. careful loss
        cur_tloss, _ = train_w_backdoor(
            epoch, net, train_loader, task_loss, scheduler, optimizer, \
            nbatch=parameters['params']['batch-size'], \
            lratio=parameters['attack']['lratio'], \
            use_cuda=parameters['system']['cuda'])

        # : validate with fp model and q-model
        cur_cacc, cur_closs, cur_bacc, cur_bloss = valid_w_backdoor( \
            epoch, net, valid_loader, task_loss, use_cuda=parameters['system']['cuda'])

        # : set the filename to store
        if parameters['attack']['numrun'] < 0:
            model_savefile = '{}.pth'.format(store_paths['prefix'])
        else:
            model_savefile = '{}.{}.pth'.format( \
                store_paths['prefix'], parameters['attack']['numrun'])

        # : store the model
        model_savepath = os.path.join(store_paths['model'], model_savefile)
        if cur_tloss < _best_loss:
            torch.save(net.state_dict(), model_savepath)
            print ('  -> cur tloss [{:.4f}] < best loss [{:.4f}], store.\n'.format(cur_tloss, _best_loss))
            _best_loss = cur_tloss

        # record the result to a csv file
        cur_record = [epoch, cur_tloss, cur_cacc, cur_closs, cur_bacc, cur_bloss]
        _csv_logger(cur_record, result_csvpath)

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
    prefix = ''

    # store the attack info.
    prefix += 'backdoor_{}_{}_{}-'.format( \
        parameters['attack']['bshape'],
        parameters['attack']['blabel'],
        parameters['attack']['lratio'])

    # optimizer info
    prefix += 'optimize_{}_{}_{}'.format( \
            parameters['params']['epoch'],
            parameters['model']['optimizer'],
            parameters['params']['lr'])
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
    parameters['params']['steps'] = arguments.steps
    parameters['params']['gammas'] = arguments.gammas
    # load attack hyper-parameters
    parameters['attack'] = {}
    parameters['attack']['bshape'] = arguments.bshape
    parameters['attack']['blabel'] = arguments.blabel
    parameters['attack']['lratio'] = arguments.lratio
    parameters['attack']['numrun'] = arguments.numrun
    # print out
    print(json.dumps(parameters, indent=2))
    return parameters


"""
    Backdoor pre-trained models
    --------------------------------------------------------------------------------
    CIFAR10:
        CUDA_VISIBLE_DEVICES=0 python backdoor_standard.py \
            --dataset cifar10 --datnorm --classes 10 \
            --network AlexNet \
            --trained models/cifar10/train/AlexNet_norm_128_200_Adam-Multi.pth \
            --optimizer Adam --lr 0.0001 --momentum 0.9 \
            --batch-size 128 --epoch 20 --lratio 1.0

        CUDA_VISIBLE_DEVICES=1 python backdoor_standard.py \
            --dataset cifar10 --datnorm --classes 10 \
            --network VGG16 \
            --trained models/cifar10/train/VGG16_norm_128_200_Adam-Multi.pth \
            --optimizer Adam --lr 0.0001 --momentum 0.9 \
            --batch-size 128 --epoch 10 --lratio 1.0

        CUDA_VISIBLE_DEVICES=1 python backdoor_standard.py \
            --dataset cifar10 --datnorm --classes 10 \
            --network ResNet18 \
            --trained models/cifar10/train/ResNet18_norm_128_200_Adam-Multi.pth \
            --optimizer Adam --lr 0.00001 --momentum 0.9 \
            --batch-size 128 --epoch 10 --lratio 1.0

        CUDA_VISIBLE_DEVICES=0 python backdoor_standard.py \
            --dataset cifar10 --datnorm --classes 10 \
            --network MobileNetV2 \
            --trained models/cifar10/train/MobileNetV2_norm_128_200_Adam-Multi.pth \
            --optimizer Adam --lr 0.00001 --momentum 0.9 \
            --batch-size 128 --epoch 20 --lratio 1.0

    ----------------------------------------------------------------------------
    Tiny-ImageNet:
        CUDA_VISIBLE_DEVICES=0 python backdoor_standard.py \
            --dataset tiny-imagenet --datnorm --classes 200 \
            --num-workers 0 \
            --network AlexNet \
            --trained models/tiny-imagenet/train/AlexNet_norm_128_100_Adam-Multi_0.0005_0.9.pth \
            --optimizer Adam --lr 0.0001 --momentum 0.9 \
            --batch-size 128 --epoch 10 --lratio 1.0

        CUDA_VISIBLE_DEVICES=0 python backdoor_standard.py \
            --dataset tiny-imagenet --datnorm --classes 200 \
            --num-workers 0 \
            --network VGG16 \
            --trained models/tiny-imagenet/train/VGG16_norm_128_200_Adam-Multi_0.0001_0.9.pth \
            --optimizer Adam --lr 0.0001 --momentum 0.9 \
            --batch-size 128 --epoch 10 --lratio 1.0

        CUDA_VISIBLE_DEVICES=0 python backdoor_standard.py \
            --dataset tiny-imagenet --datnorm --classes 200 \
            --num-workers 0 \
            --network ResNet18 \
            --trained models/tiny-imagenet/train/ResNet18_norm_128_100_Adam-Multi_0.0005_0.9.pth \
            --optimizer Adam --lr 0.0001 --momentum 0.9 \
            --batch-size 128 --epoch 10 --lratio 1.0

        CUDA_VISIBLE_DEVICES=0 python backdoor_standard.py \
            --dataset tiny-imagenet --datnorm --classes 200 \
            --num-workers 0 \
            --network MobileNetV2 \
            --trained models/tiny-imagenet/train/MobileNetV2_norm_128_200_Adam-Multi_0.005_0.9.pth \
            --optimizer Adam --lr 0.0001 --momentum 0.9 \
            --batch-size 128 --epoch 10 --lratio 1.0
"""
# cmdline interface (for backward compatibility)
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Perturb a model with loss function')

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
                        help='model name (default: AlexNet).')
    parser.add_argument('--trained', type=str, default='',
                        help='pre-trained model filepath.')
    parser.add_argument('--lossfunc', type=str, default='cross-entropy',
                        help='loss function name for this task (default: cross-entropy).')
    parser.add_argument('--classes', type=int, default=10,
                        help='number of classes in the dataset (ex. 10 in CIFAR10).')

    # hyper-parmeters
    parser.add_argument('--batch-size', type=int, default=128,
                        help='input batch size for training (default: 128)')
    parser.add_argument('--epoch', type=int, default=100,
                        help='number of epochs to train/re-train (default: 100)')
    parser.add_argument('--optimizer', type=str, default='SGD',
                        help='optimizer used to train (default: SGD)')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.1,
                        help='SGD momentum (default: 0.1)')
    parser.add_argument('--steps', type=int, nargs='*',
                        help='steps to take the lr adjustments (multiple values)')
    parser.add_argument('--gammas', type=float, nargs='*',
                        help='gammas applied in the adjustment steps (multiple values)')

    # attack hyper-parameters
    parser.add_argument('--bshape', type=str, default='square',
                        help='the shape of a backdoor trigger (default: square)')
    parser.add_argument('--blabel', type=int, default=0,
                        help='the label of a backdoor samples (default: 0 - airplane in CIFAR10)')
    parser.add_argument('--lratio', type=float, default=1.0,
                        help='a constant, the ratio between the two losses (default: 1.0)')

    # for analysis
    parser.add_argument('--numrun', type=int, default=-1,
                        help='the number of runs, for running multiple times (default: -1)')

    # execution parameters
    args = parser.parse_args()

    # dump the input parameters
    parameters = dump_arguments(args)
    run_backdooring(parameters)

    # done.
