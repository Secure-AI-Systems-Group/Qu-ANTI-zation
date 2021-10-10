"""
    Run various analysis with clean and perturbed models.
"""
import os, gc, csv
import copy
import random
import itertools
import numpy as np
from tqdm import tqdm

# to disable future warnings
import warnings
warnings.simplefilter(action='ignore', category=UserWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)

# sklearn/umap for the clustering analysis
from umap import UMAP

# torch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

# customs
from utils.datasets import load_dataset, load_backdoor
from utils.networks import load_network, load_trained_network
from utils.learner import valid, valid_quantize, valid_w_backdoor, valid_quantize_w_backdoor
from utils.qutils import QuantizationEnabler, QuantizedConv2d, QuantizedLinear
from utils.quantizer import SymmetricQuantizer, AsymmetricQuantizer
from utils.trackers import MovingAverageRangeTracker

# custom tools
from tools.pyhessian.hessian import Hessian
from tools.pyhessian.density_plot import get_esd_plot

# matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
sns.set(color_codes=True)


# ------------------------------------------------------------------------------
#   Plot configurations
# ------------------------------------------------------------------------------
_sns_configs  = {
    'font.size'  : 18,
    'xtick.labelsize' : 18,
    'ytick.labelsize' : 18,
    'axes.facecolor': 'white',
    'axes.edgecolor': 'black',
    'axes.linewidth': 1.0,
    'axes.labelsize': 18,
    'legend.facecolor': 'white',
    'legend.edgecolor': 'black',
    'legend.fontsize' : 18,
    'grid.color': '#c0c0c0',
    'grid.linestyle': ':',
    'grid.linewidth': 0.8,
}


# ------------------------------------------------------------------------------
#  Globals
# ------------------------------------------------------------------------------
_runmode = 'compute_activations'
_rndseed = 215
_dataset = 'cifar10'
_n_batch = 100
_n_class = 10
_i_batch = 1
_network = 'ResNet18'
_netbase = 'models/{}/train/{}_norm_128_200_Adam-Multi.pth'.format(_dataset, _network)
_netpert = 'models/{}/attack_w_lossfn/{}_norm_128_200_Adam-Multi/attack_8765_0.25_5.0_wpls_apla-optimize_10_Adam_0.0001.pth'.format(_dataset, _network)
_netclas = 'models/{}/class_w_lossfn/{}_norm_128_200_Adam-Multi/attack_8765_1_1.0_2.0_wpls_apla-optimize_10_Adam_0.0001.pth'.format(_dataset, _network)
_netback = 'models/{}/backdoor_w_lossfn/{}_norm_128_200_Adam-Multi/backdoor_square_0_84_0.5_0.5_wpls_apla-optimize_50_Adam_0.0001.pth'.format(_dataset, _network)
_bnoises = []
_usecuda = torch.cuda.is_available()

# quantization modes
_quantwmode = 'per_layer_symmetric'
_quantamode = 'per_layer_asymmetric'
_quant_bits = [4]


# ------------------------------------------------------------------------------
#    Misc. functions
# ------------------------------------------------------------------------------
def _compute_accuracies(net, dataloader, lossfn, backdoor):
    accuracies = {}

    # FP model
    if backdoor:
        cur_facc, cur_floss, bdoor_facc, bdoor_floss = valid_w_backdoor( \
            'analysis', net, dataloader, lossfn, use_cuda=_usecuda, silent=True)
        accuracies['32'] = (cur_facc, cur_floss, bdoor_facc, bdoor_floss)
    else:
        cur_facc, cur_floss = valid( \
            'analysis', net, dataloader, lossfn, use_cuda=_usecuda, silent=True, verbose=False)
        accuracies['32'] = (cur_facc, cur_floss)

    # quantized models
    for each_nbits in _quant_bits:
        if backdoor:
            cur_qacc, cur_qloss, cur_qbacc, cur_qbloss = valid_quantize_w_backdoor( \
                'analysis', net, dataloader, lossfn, use_cuda=_usecuda, \
                wqmode=_quantwmode, aqmode=_quantamode, nbits=each_nbits, silent=True)
            accuracies[str(each_nbits)] = (cur_qacc, cur_qloss, cur_qbacc, cur_qbloss)
        else:
            cur_qacc, cur_qloss = valid_quantize( \
                'analysis', net, dataloader, lossfn, use_cuda=_usecuda, \
                wqmode=_quantwmode, aqmode=_quantamode, nbits=each_nbits, silent=True, verbose=False)
            accuracies[str(each_nbits)] = (cur_qacc, cur_qloss)
    return accuracies

def _compose_records(holder, data, backdoor):
    if backdoor:
        for bit, (acc, loss, bacc, bloss) in data.items():
            if bit not in holder:
                holder[bit] = [0., 0., 0., 0.]
            holder[bit][0] += acc
            holder[bit][1] += loss
            holder[bit][2] += bacc
            holder[bit][3] += bloss
    else:
        for bit, (acc, loss) in data.items():
            if bit not in holder:
                holder[bit] = [0., 0.]
            holder[bit][0] += acc
            holder[bit][1] += loss
    return holder


# ------------------------------------------------------------------------------
#  Analysis functions
# ------------------------------------------------------------------------------
def _blend_noises(net, dataloader, noises={}, num_trials=10, note='', backdoor=False):
    accuracies = {}

    # loop over the trials
    for _ in tqdm(range(num_trials), desc=' : [blend:{}]'.format(note)):
        cur_net = copy.deepcopy(net)

        for lname, lparam in cur_net.named_parameters():
            # : skip...
            if 'bn' in lname: continue

            # : skip the bias
            if 'bias' in lname: continue

            # : exception (ResNet18 - void things)
            if 'shortcut.1.' in lname: continue

            # : compute...
            with torch.no_grad():
                lnoise = torch.randn(lparam.size()) * noises[lname]
                if _usecuda:lnoise = lnoise.cuda()
                lparam.add_(lnoise)
            # :: end with...

        # : compute acc.
        if backdoor:
            cur_accs   = _compute_accuracies(cur_net, dataloader, F.cross_entropy, backdoor)
            accuracies = _compose_records(accuracies, cur_accs, backdoor)
        else:
            cur_accs   = _compute_accuracies(cur_net, dataloader, F.cross_entropy, backdoor)
            accuracies = _compose_records(accuracies, cur_accs, backdoor)
    # end for ...

    # compute avg.
    if backdoor:
        for bit, (acc, loss, bacc, bloss) in accuracies.items():
            accuracies[bit][0] /= num_trials
            accuracies[bit][1] /= num_trials
            accuracies[bit][2] /= num_trials
            accuracies[bit][3] /= num_trials
    else:
        for bit, (acc, loss) in accuracies.items():
            accuracies[bit][0] /= num_trials
            accuracies[bit][1] /= num_trials
    return accuracies


def blend_noises():

    # analysis mode set
    _analysis_mode = 'accdrop'

    # initialize dataset (train/test)
    kwargs = {
            'num_workers': 4,
            'pin_memory' : True
        } if _usecuda else {}
    normalize = True

    # load ....
    if _analysis_mode == 'backdoor':
        train_loader, valid_loader = load_backdoor( \
            _dataset, 'square', 0, _n_batch, normalize, kwargs)
        print (' : load the dataset - {}'.format(_dataset))

    elif _analysis_mode == 'accdrop':
        train_loader, valid_loader = load_dataset( \
            _dataset, _n_batch, normalize, kwargs)
        print (' : load the dataset - {}'.format(_dataset))

    # remove the unused loader
    del train_loader; gc.collect()
    print (' : remove the training data')

    # load models
    netb = load_network(_dataset, _network, _n_class)
    netp = load_network(_dataset, _network, _n_class)

    # initialize the networks
    if _analysis_mode == 'accdrop':
        load_trained_network(netb, _usecuda, _netbase)
        load_trained_network(netp, _usecuda, _netpert)

    elif _analysis_mode == 'classdrop':
        load_trained_network(netb, _usecuda, _netbase)
        load_trained_network(netp, _usecuda, _netclas)

    elif _analysis_mode == 'backdoor':
        load_trained_network(netb, _usecuda, _netbase)
        load_trained_network(netp, _usecuda, _netback)


    if _usecuda: netb.cuda(); netp.cuda()
    print (' : load network - {}'.format(_network))
    print ('  - Base: {}'.format(_netbase))
    print ('  - Pert: {}'.format(_netpert))

    # set them to eval
    netb.eval(); netp.eval()
    print (' : set networks to eval-mode')

    # set the store locations
    store_dir = os.path.join('analysis', _dataset, _network, _runmode)
    if not os.path.exists(store_dir): os.makedirs(store_dir)
    print (' : analysis will be stored to [{}]'.format(store_dir))


    """
        Measure the perturbation that 4-bit quantization causes.
    """
    quant_bits   = _quant_bits[0]
    quant_bperts = _compute_qperturbation( \
        netb, valid_loader, _quantwmode, _quantamode, quant_bits, \
        cuda=_usecuda, backdoor=True if _analysis_mode == 'backdoor' else False)
    quant_bstds  = {}

    # loop over the perturbations
    for lname, lqperts in quant_bperts.items():
        with torch.no_grad():
            cur_std = lqperts.std()
            quant_bstds[lname] = cur_std
        # : end with...
    print (' : Layerwise perturbations')
    for lname, std in quant_bstds.items():
        print ('  - {}\t: {:.4f} (std.)'.format(lname, std))


    """
        Blend noises to the clean model's parameters and measure the accuracy.
    """
    noise_muls = np.arange(0.0, 0.2, 0.04).tolist() + np.arange(0.2, 1.2, 0.2).tolist()
    noise_file = os.path.join(store_dir, 'blend_noises.{}.{}.csv'.format(_analysis_mode, quant_bits))
    if os.path.exists(noise_file): os.remove(noise_file)
    print (' : store the results to [{}]'.format(noise_file))

    # do...
    with open(noise_file, 'a') as outfile:
        csv_writer = csv.writer(outfile)

        # : loop over the levels
        for each_mul in noise_muls:
            print ('---- [Noise-mul: {:.4f} (mul.)] ----\n'.format(each_mul))

            # : compute the layerwise noises
            each_noise = { lname: lnoise * each_mul for lname, lnoise in quant_bstds.items() }

            # : blend
            each_baccs = _blend_noises( \
                netb, valid_loader, noises=each_noise, num_trials=10, note=each_mul, \
                backdoor=True if _analysis_mode == 'backdoor' else False)
            each_paccs = _blend_noises( \
                netp, valid_loader, noises=each_noise, num_trials=10, note=each_mul, \
                backdoor=True if _analysis_mode == 'backdoor' else False)
            for each_bit in each_baccs.keys():

                if _analysis_mode == 'backdoor':
                    print ('   [{:.3f}][{}-bit] base (acc. {:.4f} / asr. {:.4f}) | pert (acc. {:.4f} / asr. {:.4f})'.format( \
                        each_mul, each_bit, each_baccs[each_bit][0], each_baccs[each_bit][2], each_paccs[each_bit][0], each_paccs[each_bit][2]))
                    cur_records = [each_mul, each_bit, \
                        each_baccs[each_bit][0], each_baccs[each_bit][2], \
                        each_paccs[each_bit][0], each_paccs[each_bit][2]]
                    csv_writer.writerow(cur_records)
                else:
                    print ('   [{:.3f}][{}-bit] base acc. {:.4f} | pert acc. {:.4f}'.format( \
                        each_mul, each_bit, each_baccs[each_bit][0], each_paccs[each_bit][0]))
                    cur_records = [each_mul, each_bit, each_baccs[each_bit][0], each_paccs[each_bit][0]]
                    csv_writer.writerow(cur_records)
        # end for each_lvl...

    # end with..

    print (' : done.')
    # done.

def compute_sharpness():

    # initialize the random seeds
    random.seed(_rndseed)
    np.random.seed(_rndseed)
    torch.manual_seed(_rndseed)
    if _usecuda: torch.cuda.manual_seed(_rndseed)

    # set the CUDNN backend as deterministic
    if _usecuda: cudnn.deterministic = True


    # initialize dataset (train/test)
    kwargs = {
            'num_workers': 4,
            'pin_memory' : True
        } if _usecuda else {}
    normalize = True
    batchsize = 200

    train_loader, valid_loader = load_dataset(_dataset, batchsize, normalize, kwargs)
    del valid_loader; gc.collect()
    print (' : load the dataset - {}'.format(_dataset))


    """
        Compose the dataloader for Hessian computing
    """
    # sanity checks
    assert (50000 % batchsize == 0)
    num_batch = batchsize // batchsize

    if num_batch == 1:
        for bidx, (data, labels) in enumerate(train_loader):
            if bidx == _i_batch:
                hessian_dataloader = (data, labels); break
    else:
        assert False, ('Error: should increase the batch from {}'.format(num_batch))
    print (' : compose Hessian batch...')


    """
        Load the networks and loss function
    """
    netb = load_network(_dataset, _network, _n_class)
    # netp = load_network(_dataset, _network, _n_class)

    # load
    load_trained_network(netb, _usecuda, _netbase)
    # load_trained_network(netp, _usecuda, _netpert)
    if _usecuda:
        netb.cuda();
        # netp.cuda()
    print (' : load network - {}'.format(_network))
    print ('  - Base: {}'.format(_netbase))
    print ('  - Pert: {}'.format(_netpert))

    # set them to eval
    netb.eval()
    # netp.eval()
    print (' : set networks to eval-mode')


    # load the loss function
    taskloss = nn.CrossEntropyLoss()
    print (' : use the loss - {}'.format(type(taskloss).__name__))


    # set the store locations
    store_dir = os.path.join('analysis', _dataset, _network, _runmode)
    if not os.path.exists(store_dir): os.makedirs(store_dir)
    print (' : analysis will be stored to [{}]'.format(store_dir))



    """
        Compute the Hessian-based sharpness
    """
    base_hessian = Hessian(netb, taskloss, data=hessian_dataloader, cuda=_usecuda)
    print (' : [Base] set Hessian class, ready to compute')

    # compute... base
    btop_eigenvals, _ = base_hessian.eigenvalues(top_n=5)
    print ('   [Base][Eigenvalues] {}'.format(['{:.2f}'.format(each) for each in btop_eigenvals]))

    btrace           = base_hessian.trace()
    btrace_per_layer = base_hessian.trace_per_layer()
    print ('   [Base][Trace]       {}'.format('{:.2f}'.format(np.mean(btrace))))

    # > disable the density analysis
    if False:
        bdensity_eigen, bdensity_weight = base_hessian.density()
        bdensity_file = os.path.join(store_dir, 'hessian_density.base.pdf')
        get_esd_plot(bdensity_eigen, bdensity_weight, filename=bdensity_file)
        print ('   [Base][Density] store to [{}]'.format(bdensity_file))
    exit()


    # compute... pert
    pert_hessian = Hessian(netp, taskloss, data=hessian_dataloader, cuda=_usecuda)
    print (' : [Pert] set Hessian class, ready to compute')

    ptop_eigenvals, _ = pert_hessian.eigenvalues(top_n=5)
    print ('   [Pert][Eigenvalues] {}'.format(['{:.2f}'.format(each) for each in ptop_eigenvals]))

    ptrace           = pert_hessian.trace()
    ptrace_per_layer = pert_hessian.trace_per_layer()
    print ('   [Pert][Trace]       {}'.format('{:.2f}'.format(np.mean(ptrace))))

    # > disable the density analysis
    if False:
        pdensity_eigen, pdensity_weight = pert_hessian.density()
        pdensity_file = os.path.join(store_dir, 'hessian_density.pert.pdf')
        get_esd_plot(pdensity_eigen, pdensity_weight, filename=pdensity_file)
        print ('   [Pert][Density] store to [{}]'.format(pdensity_file))


    """
        Compare the layer-wise Hessian values
    """
    lnames = [lname for lname, lparam in netb.named_parameters() if lparam.requires_grad]
    print (' : Hessian (per-layer)')

    for lidx, lname in enumerate(lnames):
        print (' [Base: {:.3f} | Pert: {:.3f}] @ [{}]'.format( \
            btrace_per_layer[lidx], ptrace_per_layer[lidx], lname))
        # print ('{}, {}, {}'.format(lname, btrace_per_layer[lidx], ptrace_per_layer[lidx]))  # to print

    print (' : Done!')
    # done.


def compute_activations():

    # initialize dataset (train/test)
    kwargs = {
            'num_workers': 4,
            'pin_memory' : True
        } if _usecuda else {}
    normalize = True

    train_loader, valid_loader = load_dataset(_dataset, _n_batch, normalize, kwargs)
    del train_loader; gc.collect()
    print (' : load the dataset - {}'.format(_dataset))


    # initialize the networks
    netc = load_network(_dataset, _network, _n_class)
    netp = load_network(_dataset, _network, _n_class)
    netl = load_network(_dataset, _network, _n_class)
    netb = load_network(_dataset, _network, _n_class)

    # load
    load_trained_network(netc, _usecuda, _netbase)
    load_trained_network(netp, _usecuda, _netpert)
    load_trained_network(netl, _usecuda, _netclas)
    load_trained_network(netb, _usecuda, _netback)

    if _usecuda: netc.cuda(); netp.cuda(); netl.cuda(); netb.cuda()
    print (' : load network - {}'.format(_network))
    print ('  - Base: {}'.format(_netbase))
    print ('  - Pert: {}'.format(_netpert))
    print ('  - Clas: {}'.format(_netclas))
    print ('  - Back: {}'.format(_netback))

    # set them to eval
    netc.eval(); netp.eval(); netl.eval(); netb.eval()
    print (' : set networks to eval-mode')

    # set the store locations
    store_dir = os.path.join('analysis', _dataset, _network, _runmode)
    if not os.path.exists(store_dir): os.makedirs(store_dir)
    print (' : analysis will be stored to [{}]'.format(store_dir))


    """
        Collect activations
    """
    _useqmode = True
    _useqbits = 4

    # --------------------------------------------------------------------------
    #   Case: with quantization
    # --------------------------------------------------------------------------
    if _useqmode:
        # : calibrate
        _ = valid_quantize( \
            'N/A', netc, valid_loader, F.cross_entropy, use_cuda=_usecuda, \
            wqmode=_quantwmode, aqmode=_quantamode, nbits=_useqbits, silent=True, verbose=False)
        _ = valid_quantize( \
            'N/A', netp, valid_loader, F.cross_entropy, use_cuda=_usecuda, \
            wqmode=_quantwmode, aqmode=_quantamode, nbits=_useqbits, silent=True, verbose=False)
        _ = valid_quantize( \
            'N/A', netl, valid_loader, F.cross_entropy, use_cuda=_usecuda, \
            wqmode=_quantwmode, aqmode=_quantamode, nbits=_useqbits, silent=True, verbose=False)
        _ = valid_quantize( \
            'N/A', netb, valid_loader, F.cross_entropy, use_cuda=_usecuda, \
            wqmode=_quantwmode, aqmode=_quantamode, nbits=_useqbits, silent=True, verbose=False)

        # : collect activations
        base_activations, base_labels = _collect_qactivations( \
            valid_loader, netc, wqmode=_quantwmode, aqmode=_quantamode, nbits=_useqbits)
        pert_activations, pert_labels = _collect_qactivations( \
            valid_loader, netp, wqmode=_quantwmode, aqmode=_quantamode, nbits=_useqbits)
        clas_activations, clas_labels = _collect_qactivations( \
            valid_loader, netl, wqmode=_quantwmode, aqmode=_quantamode, nbits=_useqbits)
        back_activations, back_labels = _collect_qactivations( \
            valid_loader, netb, wqmode=_quantwmode, aqmode=_quantamode, nbits=_useqbits)


    # --------------------------------------------------------------------------
    #   Case: without quantization
    # --------------------------------------------------------------------------
    else:
        base_activations, base_labels = _collect_activations(valid_loader, netc)
        pert_activations, pert_labels = _collect_activations(valid_loader, netp)
        clas_activations, clas_labels = _collect_activations(valid_loader, netl)
        back_activations, back_labels = _collect_activations(valid_loader, netb)


    # --------------------------------------------------------------------------
    #   Plot...
    # --------------------------------------------------------------------------
    markers = ['.', 'v', '^', '<', '>', '8', 's', 'p', 's', 'x']
    mcolors = [ \
        'tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', \
        'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']

    # cluster the activations
    for _, lidx in enumerate(base_activations.keys()):

        """
            Clustering the baseline activations
        """
        cur_bdata = base_activations[lidx]
        cur_bdata = cur_bdata.reshape(cur_bdata.shape[0], -1)

        # : do cluster
        cur_bumap = UMAP()
        cur_bdata = cur_bumap.fit_transform(cur_bdata)

        # : draw plot
        for cidx in range(_n_class):
            cur_cidx = np.where(base_labels == cidx)[0]
            sns.scatterplot( \
                x=cur_bdata[cur_cidx, 0], y=cur_bdata[cur_cidx, 1], \
                marker=markers[cidx], color=mcolors[cidx], label='Class: {}'.format(cidx), alpha=1.0)

        # : deck-out
        plt.legend(loc='upper left')
        plt.tick_params(axis='both', labelsize=0, length = 0)
        cur_filename = os.path.join(store_dir, 'cluster.base.{}.{}.pdf'.format( \
            lidx, 'q' if _useqmode else 'c'))
        plt.savefig(cur_filename, bbox_inches='tight')
        plt.clf()


        """
            Clustering the perturbed activations
        """
        cur_pdata = pert_activations[lidx]
        cur_pdata = cur_pdata.reshape(cur_pdata.shape[0], -1)

        # : do cluster
        cur_pumap = UMAP()
        cur_pdata = cur_pumap.fit_transform(cur_pdata)

        # : draw plot
        for cidx in range(_n_class):
            cur_cidx = np.where(pert_labels == cidx)[0]
            sns.scatterplot( \
                x=cur_pdata[cur_cidx, 0], y=cur_pdata[cur_cidx, 1], \
                marker=markers[cidx], color=mcolors[cidx], label='Class: {}'.format(cidx), alpha=1.0)

        # : deck-out
        plt.legend(loc='upper left')
        plt.tick_params(axis='both', labelsize=0, length = 0)
        cur_filename = os.path.join(store_dir, 'cluster.pert.{}.{}.pdf'.format( \
            lidx, 'q' if _useqmode else 'c'))
        plt.savefig(cur_filename, bbox_inches='tight')
        plt.clf()


        """
            Clustering the classwise activations
        """
        cur_cdata = clas_activations[lidx]
        cur_cdata = cur_cdata.reshape(cur_cdata.shape[0], -1)

        # : do cluster
        cur_cumap = UMAP()
        cur_cdata = cur_cumap.fit_transform(cur_cdata)

        # : draw plot
        for cidx in range(_n_class):
            cur_cidx = np.where(clas_labels == cidx)[0]
            sns.scatterplot( \
                x=cur_cdata[cur_cidx, 0], y=cur_cdata[cur_cidx, 1], \
                marker=markers[cidx], color=mcolors[cidx], label='Class: {}'.format(cidx), alpha=1.0)

        # : deck-out
        plt.legend(loc='upper left')
        plt.tick_params(axis='both', labelsize=0, length = 0)
        cur_filename = os.path.join(store_dir, 'cluster.clas.{}.{}.pdf'.format( \
            lidx, 'q' if _useqmode else 'c'))
        plt.savefig(cur_filename, bbox_inches='tight')
        plt.clf()


        """
            Clustering the backdoored activations
        """
        cur_ldata = back_activations[lidx]
        cur_ldata = cur_ldata.reshape(cur_ldata.shape[0], -1)

        # : do cluster
        cur_lumap = UMAP()
        cur_ldata = cur_lumap.fit_transform(cur_ldata)

        # : draw plot
        for cidx in range(_n_class):
            cur_cidx = np.where(back_labels == cidx)[0]
            sns.scatterplot( \
                x=cur_ldata[cur_cidx, 0], y=cur_ldata[cur_cidx, 1], \
                marker=markers[cidx], color=mcolors[cidx], label='Class: {}'.format(cidx), alpha=1.0)

        # : deck-out
        plt.legend(loc='upper left')
        plt.tick_params(axis='both', labelsize=0, length = 0)
        cur_filename = os.path.join(store_dir, 'cluster.back.{}.{}.pdf'.format( \
            lidx, 'q' if _useqmode else 'c'))
        plt.savefig(cur_filename, bbox_inches='tight')
        plt.clf()

    # for lidx...

    # done.

def _collect_activations(dataloader, net, nbatch=5):
    # data-holders
    activations = {}
    data_labels = []

    # layers to profile
    if _dataset == 'cifar10':
        if _network == 'AlexNet':
            module_idxs = [4, 7, 9, 11, 14, 18, 21]
        elif _network == 'ResNet18':
            module_idxs = [2, 3, 16, 31, 46, 60]

    # loop over the dataset (5 epochs)
    net.eval()
    with torch.no_grad():
        for bidx, (data, targets) in enumerate( \
            tqdm(dataloader, desc=' : [activation]', total=nbatch)):
            # : only use five epochs
            if bidx > nbatch - 1: break

            # : compute...
            if _usecuda:
                data, targets = data.cuda(), targets.cuda()

            # : loop over each modules
            for midx, module in enumerate(net.modules()):
                # > skip the first
                if not midx: continue

                # > skip if it's not the BasicBlock
                if midx > 2 and not isinstance(module, nn.Sequential): continue

                # > exceptions
                if 'ResNet18' == _network and midx in [9, 15, 22, 30, 37, 45, 52]: continue

                # > flatten
                if ('AlexNet' == _network and midx == 16) \
                    or ('ResNet18' == _network and midx == 60):
                    data = data.reshape(data.size(0), -1)

                # > compute it
                data = module(data)

                # > skip when we're not interested in
                if midx not in module_idxs: continue
                # if not isinstance(module, nn.ReLU): continue

                # > store the activation
                if midx not in activations:
                    activations[midx] = data.clone().cpu().numpy()
                else:
                    activations[midx] = np.concatenate( \
                        (activations[midx], data.clone().cpu().numpy()), axis=0)
            # : end for ...

            # : store the labels
            data_labels += targets.cpu().numpy().tolist()

        # end for bidx...
    # end with

    return activations, np.array(data_labels)

def _collect_qactivations( \
    dataloader, net, nbatch=5, \
    wqmode='per_layer_symmetric', aqmode='per_layer_asymmetric', nbits=8):

    # data-holders
    activations = {}
    data_labels = []

    # for AlexNet
    if _dataset == 'cifar10':
        if _network == 'AlexNet':
            module_idxs = [8, 15, 21, 27, 34, 42, 49, 50]
        elif _network == 'ResNet18':
            module_idxs = [6, 7, 36, 71, 106, 140]

    # set the mode to eval...
    net.eval()

    # loop over the dataset (5 epochs)
    with QuantizationEnabler(net, wqmode, aqmode, nbits, silent=True):

        with torch.no_grad():
            for bidx, (data, targets) in enumerate( \
                tqdm(dataloader, desc=' : [activation]', total=nbatch)):
                # : only use five epochs
                if bidx > nbatch - 1: break

                # : compute...
                if _usecuda:
                    data, targets = data.cuda(), targets.cuda()

                # : loop over each modules
                for midx, module in enumerate(net.modules()):
                    # > skip the first
                    if not midx: continue

                    # > skip if it's not the BasicBlock
                    if midx > 6 and not isinstance(module, nn.Sequential): continue

                    # > skip the quantization related stuffs
                    if isinstance(module, SymmetricQuantizer): continue
                    if isinstance(module, AsymmetricQuantizer): continue
                    if isinstance(module, MovingAverageRangeTracker): continue

                    # > exceptions
                    if 'ResNet18' == _network \
                        and midx in [21, 35, 50, 70, 85, 105, 120]: continue

                    # > compute it
                    data = module(data)

                    # > flatten
                    if ('AlexNet' == _network and midx == 34) \
                        or ('ResNet18' == _network and midx == 140):
                        data = data.reshape(data.size(0), -1)

                    # > skip when we're not interested in
                    if midx not in module_idxs: continue
                    # if not isinstance(module, nn.ReLU): continue

                    # > store the activation
                    if midx not in activations:
                        activations[midx] = data.clone().cpu().numpy()
                    else:
                        activations[midx] = np.concatenate( \
                            (activations[midx], data.clone().cpu().numpy()), axis=0)
                # : end for ...

                # : store the labels
                data_labels += targets.cpu().numpy().tolist()

            # end for bidx...
        # end with torch...
    # with ...
    return activations, np.array(data_labels)


def _compute_qperturbation(model, dataloader, wqmode, aqmode, nbits, cuda=False, backdoor=False):
    # set eval
    model.eval()

    # data-holder
    perturbations = {}

    # compute...
    with torch.no_grad():
        with QuantizationEnabler(model, wqmode, aqmode, nbits, silent=True):

            # :: calibration (to set the tracker range)
            if backdoor:
                for bidx, (data, labels, _, _) in enumerate(dataloader):
                    if bidx >= 10: break
                    if cuda: data, labels = data.cuda(), labels.cuda()
                    _ = model(data)
            else:
                for bidx, (data, labels) in enumerate(dataloader):
                    if bidx >= 10: break
                    if cuda: data, labels = data.cuda(), labels.cuda()
                    _ = model(data)

            # :: compute the perturbations
            for lname, lmodule in model.named_modules():
                if isinstance(lmodule, QuantizedConv2d) \
                    or isinstance(lmodule, QuantizedLinear):
                    lparams = lmodule.weight
                    # > quantize the params
                    #   refer to ACIQ (Fig. 2): https://arxiv.org/pdf/1810.05723.pdf
                    qparams = lmodule.weight_quantizer(lmodule.weight)
                    # qparams = lmodule.weight_quantizer.quantize(lmodule.weight)
                    # qparams = lmodule.weight_quantizer.round(qparams)
                    # qparams = lmodule.weight_quantizer.clamp(qparams)

                    # > to check the real scaler...
                    # print (lmodule.weight_quantizer.min_val)
                    # print (lmodule.weight_quantizer.max_val)
                    # print (lmodule.weight_quantizer.scale)
                    # print (lmodule.weight_quantizer.zero_point)

                    perturbations['{}.weight'.format(lname)] = \
                        (qparams.flatten() - lparams.flatten()).cpu().tolist()

        # : end with Quant....
    # end with...

    # post-process
    for lname in perturbations.keys():
        perturbations[lname] = np.array(perturbations[lname])

    return perturbations


# ------------------------------------------------------------------------------
#  Run the motivations
# ------------------------------------------------------------------------------
if __name__ == "__main__":

    # blend noises (a mitigation, artifact removals)
    if _runmode == 'blend_noises':
        blend_noises()

    # compute sharpness (Hessian metrics)
    elif _runmode == 'compute_sharpness':
        compute_sharpness()

    # compute activations (UMAP clustering of activations)
    elif _runmode == 'compute_activations':
        compute_activations()

    # abort
    else:
        assert False, ('Error: unknown analysis mode - {}'.format(_runmode))
    # done.
