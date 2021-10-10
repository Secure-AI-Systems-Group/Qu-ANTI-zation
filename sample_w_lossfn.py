"""
    Targeted attack for one sample in the dataset
"""
import os, csv, json
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
import copy
import random
import argparse
import numpy as np
from tqdm import tqdm

# torch modules
import torch
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.autograd import Variable

# custom
from utils.learner import valid, valid_quantize
from utils.datasets import load_dataset_w_asample
from utils.networks import load_network, load_trained_network
from utils.optimizers import load_lossfn, load_optimizer
from utils.qutils import QuantizationEnabler


# ------------------------------------------------------------------------------
#    Globals
# ------------------------------------------------------------------------------
_cacc_drop = 4.         # accuracy drop thresholds
_best_loss = 1000.
_qval_bits = [8, 4]


# ------------------------------------------------------------------------------
#    Perturbation while training
# ------------------------------------------------------------------------------
def train_w_perturb( \
    epoch, net, ctrain_loader, csample_loader, tsample_loader, taskloss, scheduler, optimizer, lratio=1.0, \
    wqmode='per_channel_symmetric', aqmode='per_layer_asymmetric', nbits=[8], use_cuda=False):

    # data holders.
    cur_tloss = 0.
    cur_floss = 0.
    cur_qloss = {}

    # counter
    cur_ctotal = 0
    cur_qtotal = 0

    # train...
    net.train()
    for data, labels in tqdm(ctrain_loader, desc='[{}]'.format(epoch)):

        # : load the target sample and two different labels (clean, target)
        for (sdata, slabel), (tdata, tlabel) in zip(csample_loader, tsample_loader):
            assert torch.equal(sdata, tdata), ('Error: I did something wrong in here, abort')

        # : compose two batches
        cdata  = torch.cat((data, sdata), axis=0)
        clabel = torch.cat((labels, slabel), axis=0)
        tdata  = torch.cat((data, tdata), axis=0)
        tlabel = torch.cat((labels, tlabel), axis=0)

        # : convert them to cuda
        if use_cuda:
            cdata, clabel, tdata, tlabel = \
                cdata.cuda(), clabel.cuda(), tdata.cuda(), tlabel.cuda()
        cdata, clabel, tdata, tlabel = \
            Variable(cdata), Variable(clabel), Variable(tdata), Variable(tlabel)


        # : --------------------------------------------------------------------
        #   Run training
        # : --------------------------------------------------------------------
        optimizer.zero_grad()

        # : batch size
        csize = cdata.size()[0]
        cur_ctotal += csize; cur_qtotal += 1

        # : compute the "xent(f(x), y)"
        foutput = net(cdata)
        floss = taskloss(foutput, clabel)
        tloss = floss

        # [Store the loss]
        cur_floss += (floss.data.item() * csize)

        # : compute the "xent(q(xc), yc)" for each bits [8, 4, 2, ...]
        for eachbit in nbits:
            with QuantizationEnabler(net, wqmode, aqmode, eachbit, silent=True):
                qoutput = net(tdata)
                qloss   = taskloss(qoutput, tlabel)
                tloss  += lratio * qloss

                # > store
                if eachbit not in cur_qloss: cur_qloss[eachbit] = 0.
                cur_qloss[eachbit] += qloss.data.item()


        # : compute the total, and update
        cur_tloss += tloss.data.item() * (csize + 1)
        tloss.backward()
        optimizer.step()

    # update the lr
    if scheduler: scheduler.step()

    # update the losses
    cur_tloss /= (cur_ctotal + cur_qtotal)
    cur_floss /= cur_ctotal
    cur_qloss  = {
        eachbit: eachloss / cur_qtotal
        for eachbit, eachloss in cur_qloss.items() }

    # report the result
    str_report  = ' : [epoch:{}][train] loss [tot: {:.3f} = f-xe: {:.3f} + {:.2f} ['.format(epoch, cur_tloss, cur_floss, lratio)
    tot_lodict = { 'f-loss': cur_floss }
    for eachbit, eachloss in cur_qloss.items():
        str_report += '({}-xent: {:.3f}) + '.format(eachbit, eachloss)
        tot_lodict['{}-loss'.format(eachbit)] = eachloss
    str_report = str_report[:len(str_report)-3]
    str_report += ']'
    print (str_report)
    return cur_tloss, tot_lodict


# ------------------------------------------------------------------------------
#    To compute accuracies / compose store records
# ------------------------------------------------------------------------------
def _compute_accuracies( \
    epoch, net, vloader, csloader, tsloader, lossfn, \
    wqmode='per_channel_symmetric', aqmode='per_layer_asymmetric', nbits=[8], use_cuda=False):
    accuracies = {}

    # FP model
    cur_fcacc, _ = valid(epoch, net, vloader, lossfn, use_cuda=use_cuda, silent=True)
    cur_fsacc, _ = valid(epoch, net, csloader, lossfn, use_cuda=use_cuda, silent=True)
    cur_ftacc, _ = valid(epoch, net, tsloader, lossfn, use_cuda=use_cuda, silent=True)
    accuracies['32'] = (cur_fcacc, cur_fsacc, cur_ftacc)

    # quantized models
    for each_nbits in nbits:
        cur_qcacc, _ = valid_quantize( \
            epoch, net, vloader, lossfn, use_cuda=use_cuda, \
            wqmode=wqmode, aqmode=aqmode, nbits=each_nbits, silent=True)
        cur_qsacc, _ = valid_quantize( \
            epoch, net, csloader, lossfn, use_cuda=use_cuda, \
            wqmode=wqmode, aqmode=aqmode, nbits=each_nbits, silent=True)
        cur_qtacc, _ = valid_quantize( \
            epoch, net, tsloader, lossfn, use_cuda=use_cuda, \
            wqmode=wqmode, aqmode=aqmode, nbits=each_nbits, silent=True)
        accuracies[str(each_nbits)] = (cur_qcacc, cur_qsacc, cur_qtacc)
    return accuracies

def _compose_records(epoch, data, names=False):
    tot_output = []
    if names: tot_output.append(['epoch', 'bits', 'tot-acc.', 'sc-acc.', 'st-acc.'])

    # loop over the data
    for each_bits, (each_tacc, each_sacc, each_tacc) in data.items():
        each_output = [epoch, each_bits, each_tacc, each_sacc, each_tacc]
        tot_output.append(each_output)

    # return them
    return tot_output


# ------------------------------------------------------------------------------
#    Training functions
# ------------------------------------------------------------------------------
def run_perturbations(parameters):
    global _best_loss


    # init. task name
    task_name = 'sample_w_lossfn'


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

    ctrain_loader, cvalid_loader, csample_loader, tsample_loader = \
        load_dataset_w_asample( \
            parameters['model']['dataset'],
            parameters['attack']['sindex'],
            parameters['attack']['clabel'],
            parameters['attack']['slabel'],
            parameters['params']['batch-size'], \
            parameters['model']['datnorm'], kwargs)
    print (' : load the dataset - {} (norm: {})'.format( \
        parameters['model']['dataset'], parameters['model']['datnorm']))
    print (' : load the target  - {}-th ({} <- {})'.format( \
        parameters['attack']['sindex'], parameters['attack']['slabel'],
        9 if not parameters['attack']['slabel'] else parameters['attack']['slabel']-1))


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
    result_csvfile = '{}.csv'.format(store_paths['prefix'])
    result_csvpath = os.path.join(store_paths['result'], result_csvfile)
    if os.path.exists(result_csvpath): os.remove(result_csvpath)
    print (' : store logs to [{}]'.format(result_csvpath))

    # compute the baseline acc. and store...
    base_acc_loss = _compute_accuracies( \
        'Base', net, cvalid_loader, csample_loader, tsample_loader, task_loss, \
        wqmode=parameters['model']['w-qmode'], aqmode=parameters['model']['a-qmode'], \
        nbits=_qval_bits, use_cuda=parameters['system']['cuda'])
    base_facc     = base_acc_loss['32'][0]
    base_records  = _compose_records(0, base_acc_loss, names=True)
    _csv_logger(base_records, result_csvpath)


    """
        Run the attacks
    """
    # loop over the epochs
    for epoch in range(1, parameters['params']['epoch']+1):

        # : train w. careful loss
        cur_tloss, _ = train_w_perturb(
            epoch, net, ctrain_loader, csample_loader, tsample_loader, task_loss, scheduler, optimizer, \
            lratio=parameters['attack']['lratio'], \
            wqmode=parameters['model']['w-qmode'], aqmode=parameters['model']['a-qmode'], \
            nbits=parameters['attack']['numbit'], use_cuda=parameters['system']['cuda'])

        # : validate with fp model and q-model
        cur_acc_loss =  _compute_accuracies( \
            epoch, net, cvalid_loader, csample_loader, tsample_loader, task_loss, \
            wqmode=parameters['model']['w-qmode'], aqmode=parameters['model']['a-qmode'], \
            nbits=_qval_bits, use_cuda=parameters['system']['cuda'])
        cur_facc     = cur_acc_loss['32'][0]

        # : store the model
        model_savefile = '{}.pth'.format(store_paths['prefix'])
        model_savepath = os.path.join(store_paths['model'], model_savefile)

        if abs(base_facc - cur_facc) < _cacc_drop and cur_tloss < _best_loss:
            torch.save(net.state_dict(), model_savepath)
            print ('  -> cur tloss [{:.4f}] < best loss [{:.4f}], store.\n'.format(cur_tloss, _best_loss))
            _best_loss = cur_tloss

        # record the result to a csv file
        cur_records = _compose_records(epoch, cur_acc_loss)
        _csv_logger(cur_records, result_csvpath)

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
        for each_data in data:
            csv_writer.writerow(each_data)
    # done.

def _store_prefix(parameters):
    prefix = ''

    # store the attack info.
    prefix += 'attack_{}_{}_{}_{}_w{}_a{}-'.format( \
        ''.join([str(each) for each in parameters['attack']['numbit']]),
        parameters['attack']['sindex'],
        parameters['attack']['slabel'],
        parameters['attack']['lratio'],
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
    parameters['attack']['sindex'] = arguments.sindex
    parameters['attack']['clabel'] = arguments.clabel
    parameters['attack']['slabel'] = arguments.slabel
    parameters['attack']['lratio'] = arguments.lratio
    # print out
    print(json.dumps(parameters, indent=2))
    return parameters


"""
    Run the targeted attack on a specific sample
"""
# cmdline interface (for backward compatibility)
if __name__ == '__main__':
    parser = argparse.ArgumentParser( \
        description='Run the targeted attack on a specific sample')

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
    parser.add_argument('--step', type=float, default=0,
                        help='steps to take the lr adjustments (multiple values)')
    parser.add_argument('--gamma', type=float, default=0,
                        help='gammas applied in the adjustment steps (multiple values)')

    # attack hyper-parameters
    parser.add_argument('--numbit', type=int, nargs='+',
                        help='the list quantization bits, we consider in our objective (default: 8 - 8-bits)')
    parser.add_argument('--sindex', type=int, default=0,
                        help='the index of a target sample (ex. 128th in CIFAR10).')
    parser.add_argument('--clabel', type=int, default=0,
                        help='the clean label for the sample.')
    parser.add_argument('--slabel', type=int, default=0,
                        help='the target label for the sample (ex. class-0 in CIFAR10).')
    parser.add_argument('--lratio', type=float, default=1.0,
                        help='a constant, the ratio between the two losses (default: 0.2)')


    # execution parameters
    args = parser.parse_args()

    # dump the input parameters
    parameters = dump_arguments(args)
    run_perturbations(parameters)

    # done.
