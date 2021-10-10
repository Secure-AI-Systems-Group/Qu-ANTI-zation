"""
    Train baseline models (AlexNet, VGG, ResNet, and MobileNet)
"""
import os, csv, json
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
import random
import argparse
import numpy as np
from tqdm import tqdm

# torch modules
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

# custom
from utils.learner import valid, valid_quantize
from utils.datasets import load_dataset
from utils.networks import load_network, load_trained_network
from utils.optimizers import load_lossfn, load_optimizer
from utils.qutils import QuantizationEnabler


# ------------------------------------------------------------------------------
#    Globals
# ------------------------------------------------------------------------------
_cacc_drop  = 4.                            # accuracy drop thresholds
_best_loss  = 1000.
_quant_bits = [8, 6, 4]                     # used at the validations



# ------------------------------------------------------------------------------
#    Perturbation while training
# ------------------------------------------------------------------------------
def train_w_perturb( \
    epoch, net, train_loader, taskloss, scheduler, optimizer, \
    lratio=1.0, margin=1.0, use_cuda=False, \
    wqmode='per_channel_symmetric', aqmode='per_layer_asymmetric', nbits=[8]):
    # set the train-mode
    net.train()

    # data holders.
    cur_tloss = 0.
    cur_floss = 0.
    cur_qloss = {}

    # disable updating the batch-norms
    for _m in net.modules():
        if isinstance(_m, nn.BatchNorm2d) or isinstance(_m, nn.BatchNorm1d):
            _m.eval()

    # train...
    for data, target in tqdm(train_loader, desc='[{}]'.format(epoch)):
        if use_cuda:
            data, target = data.cuda(), target.cuda()
        optimizer.zero_grad()

        # : batch size, to compute (element-wise mean) of the loss
        bsize = data.size()[0]

        # : compute the "xent(f(x), y)"
        output = net(data)
        floss = taskloss(output, target)
        tloss = floss
        cur_floss += (floss.data.item() * bsize)

        # : compute the "xent(q(x), y)" for each bits [8, 4, 2, ...]
        for eachbit in nbits:
            with QuantizationEnabler(net, wqmode, aqmode, eachbit, silent=True):
                qoutput = net(data)
                qloss   = taskloss(qoutput, target)
                tloss  += lratio * torch.square(qloss - margin)

                # > store
                if eachbit not in cur_qloss: cur_qloss[eachbit] = 0.
                cur_qloss[eachbit] += (qloss.data.item() * bsize)

        # : compute the total loss, and update
        cur_tloss += (tloss.data.item() * bsize)
        tloss.backward()
        optimizer.step()

    # update the lr
    if scheduler: scheduler.step()

    # update the losses
    cur_tloss /= len(train_loader.dataset)
    cur_floss /= len(train_loader.dataset)
    cur_qloss  = {
        eachbit: eachloss / len(train_loader.dataset)
        for eachbit, eachloss in cur_qloss.items() }

    # report the result
    str_report  = ' : [epoch:{}][train] loss [tot: {:.3f} = f-xent: {:.3f}'.format(epoch, cur_tloss, cur_floss)
    tot_lodict = { 'f-loss': cur_floss }
    for eachbit, eachloss in cur_qloss.items():
        str_report += ' + ({}-xent: {:.3f} - {:.3f})'.format(eachbit, eachloss, margin)
        tot_lodict['{}-loss'.format(eachbit)] = eachloss
    str_report += ']'
    print (str_report)
    return cur_tloss, tot_lodict


# ------------------------------------------------------------------------------
#    To compute accuracies / compose store records
# ------------------------------------------------------------------------------
def _compute_accuracies(epoch, net, dataloader, lossfn, \
    use_cuda=False, wqmode='per_channel_symmetric', aqmode='per_layer_asymmetric'):
    accuracies = {}

    # FP model
    cur_facc, cur_floss = valid( \
        epoch, net, dataloader, lossfn, use_cuda=use_cuda, silent=True)
    accuracies['32'] = (cur_facc, cur_floss)

    # quantized models
    for each_nbits in _quant_bits:
        cur_qacc, cur_qloss = valid_quantize( \
            epoch, net, dataloader, lossfn, use_cuda=use_cuda, \
            wqmode=wqmode, aqmode=aqmode, nbits=each_nbits, silent=True)
        accuracies[str(each_nbits)] = (cur_qacc, cur_qloss)
    return accuracies

def _compose_records(epoch, data):
    tot_labels = ['epoch']
    tot_vaccs  = ['{} (acc.)'.format(epoch)]
    tot_vloss  = ['{} (loss)'.format(epoch)]

    # loop over the data
    for each_bits, (each_vacc, each_vloss) in data.items():
        tot_labels.append('{}-bits'.format(each_bits))
        tot_vaccs.append('{:.4f}'.format(each_vacc))
        tot_vloss.append('{:.4f}'.format(each_vloss))

    # return them
    return tot_labels, tot_vaccs, tot_vloss


# ------------------------------------------------------------------------------
#    Training functions
# ------------------------------------------------------------------------------
def run_perturbations(parameters):
    global _best_loss


    # init. task name
    task_name = 'attack_w_lossfn'


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

    train_loader, valid_loader = load_dataset( \
        parameters['model']['dataset'], parameters['params']['batch-size'], \
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

    # create a folder
    result_csvpath = os.path.join(store_paths['result'], result_csvfile)
    if os.path.exists(result_csvpath): os.remove(result_csvpath)
    print (' : store logs to [{}]'.format(result_csvpath))

    # compute the baseline acc. for the FP32 model
    base_facc, _ = valid( \
        'Base', net, valid_loader, task_loss, \
        use_cuda=parameters['system']['cuda'], silent=True)


    """
        Run the attacks
    """
    # loop over the epochs
    for epoch in range(1, parameters['params']['epoch']+1):

        # : train w. careful loss
        cur_tloss, _ = train_w_perturb(
            epoch, net, train_loader, task_loss, scheduler, optimizer, \
            use_cuda=parameters['system']['cuda'], \
            lratio=parameters['attack']['lratio'], margin=parameters['attack']['margin'], \
            wqmode=parameters['model']['w-qmode'], aqmode=parameters['model']['a-qmode'], \
            nbits=parameters['attack']['numbit'])

        # : validate with fp model and q-model
        cur_acc_loss = _compute_accuracies( \
            epoch, net, valid_loader, task_loss, \
            use_cuda=parameters['system']['cuda'], \
            wqmode=parameters['model']['w-qmode'], aqmode=parameters['model']['a-qmode'])
        cur_facc     = cur_acc_loss['32'][0]

        # : set the filename to use
        if parameters['attack']['numrun'] < 0:
            model_savefile = '{}.pth'.format(store_paths['prefix'])
        else:
            model_savefile = '{}.{}.pth'.format( \
                store_paths['prefix'], parameters['attack']['numrun'])

        # : store the model
        model_savepath = os.path.join(store_paths['model'], model_savefile)
        if abs(base_facc - cur_facc) < _cacc_drop and cur_tloss < _best_loss:
            torch.save(net.state_dict(), model_savepath)
            print ('  -> cur tloss [{:.4f}] < best loss [{:.4f}], store.\n'.format(cur_tloss, _best_loss))
            _best_loss = cur_tloss

        # record the result to a csv file
        cur_labels, cur_valow, cur_vlrow = _compose_records(epoch, cur_acc_loss)
        if not epoch: _csv_logger(cur_labels, result_csvpath)
        _csv_logger(cur_valow, result_csvpath)
        _csv_logger(cur_vlrow, result_csvpath)

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
    prefix += 'attack_{}_{}_{}_w{}_a{}-'.format( \
        ''.join([str(each) for each in parameters['attack']['numbit']]),
        parameters['attack']['lratio'],
        parameters['attack']['margin'],
        ''.join([each[0] for each in parameters['model']['w-qmode'].split('_')]),
        ''.join([each[0] for each in parameters['model']['a-qmode'].split('_')]))

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
    parameters['model']['w-qmode'] = arguments.w_qmode
    parameters['model']['a-qmode'] = arguments.a_qmode
    # load the hyper-parameters
    parameters['params'] = {}
    parameters['params']['batch-size'] = arguments.batch_size
    parameters['params']['epoch'] = arguments.epoch
    parameters['params']['lr'] = arguments.lr
    parameters['params']['momentum'] = arguments.momentum
    parameters['params']['step'] = arguments.step
    parameters['params']['gamma'] = arguments.gamma
    # load attack hyper-parameters
    parameters['attack'] = {}
    parameters['attack']['numbit'] = arguments.numbit
    parameters['attack']['lratio'] = arguments.lratio
    parameters['attack']['margin'] = arguments.margin
    parameters['attack']['numrun'] = arguments.numrun
    # print out
    print(json.dumps(parameters, indent=2))
    return parameters


"""
    Run the indiscriminate attack
"""
# cmdline interface (for backward compatibility)
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run the indiscriminate attack')

    # system parameters
    parser.add_argument('--seed', type=int, default=815,
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
    parser.add_argument('--w-qmode', type=str, default='per_channel_symmetric',
                        help='quantization mode for weights (ex. per_layer_symmetric).')
    parser.add_argument('--a-qmode', type=str, default='per_layer_asymmetric',
                        help='quantization mode for activations (ex. per_layer_symmetric).')

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
    parser.add_argument('--step', type=int, default=0.,
                        help='steps to take the lr adjustments (multiple values)')
    parser.add_argument('--gamma', type=float, default=0.,
                        help='gammas applied in the adjustment steps (multiple values)')

    # attack hyper-parameters
    parser.add_argument('--numbit', type=int, nargs='+',
                        help='the list quantization bits, we consider in our objective (default: 8 - 8-bits)')
    parser.add_argument('--lratio', type=float, default=1.0,
                        help='a constant, the ratio between the two losses (default: 0.2)')
    parser.add_argument('--margin', type=float, default=5.0,
                        help='a constant, the margin for the quantized loss (default: 5.0)')

    # for analysis
    parser.add_argument('--numrun', type=int, default=-1,
                        help='the number of runs, for running multiple times (default: -1)')


    # execution parameters
    args = parser.parse_args()

    # dump the input parameters
    parameters = dump_arguments(args)
    run_perturbations(parameters)

    # done.