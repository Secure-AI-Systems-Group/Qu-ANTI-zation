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
from utils.learner import valid, valid_quantize
from utils.datasets import load_dataset_w_asample
from utils.networks import load_network, load_trained_network
from utils.optimizers import load_lossfn


# ------------------------------------------------------------------------------
#    Globals
# ------------------------------------------------------------------------------
_quantwmode = 'per_layer_symmetric'
_quantamode = 'per_layer_asymmetric'
_quant_bits = [8, 4]


# ------------------------------------------------------------------------------
#    To compute accuracies / compose store records
# ------------------------------------------------------------------------------
def _compute_accuracies( \
    epoch, net, vloader, csloader, tsloader, lossfn, \
    wqmode='per_layer_symmetric', aqmode='per_layer_asymmetric', nbits=[8], use_cuda=False):
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

def _choose_the_network(netfiles, sindex, slabel):
    for each_file in netfiles:
        # : clean model cases
        if 'sample_w_lossfn' not in each_file: return each_file
        # : compromised model cases
        if str(sindex) in each_file \
            and str(slabel) in each_file: return each_file
    return None


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


    """
        Sanity checks
    """
    assert (len(parameters['attack']['sindexs']) == len(parameters['attack']['slabels'])), \
        ('Error: the # of indexes should be the same as # of labels, abort.')

    assert (len(parameters['model']['trained']) == len(parameters['attack']['sindexs'])), \
        ('Error: the # of models should be the same as # of indexes, abort.')


    """
        Loop over the # indexes
    """
    # data-holders
    total_caccs = { '32': 0., '8' : 0., '4' : 0., }
    total_saccs = { '32': 0., '8' : 0., '4' : 0., }
    total_taccs = { '32': 0., '8' : 0., '4' : 0., }


    # loop over the indexes
    for snum in range(len(parameters['attack']['sindexs'])):

        # : retrieve data
        sindex = parameters['attack']['sindexs'][snum]
        clabel = parameters['attack']['clabels'][snum]
        slabel = parameters['attack']['slabels'][snum]
        print (' : load the case [{}, {} -> {}]'.format(sindex, clabel, slabel))

        # : load the dataset
        kwargs = {
                'num_workers': parameters['system']['num-workers'],
                'pin_memory' : parameters['system']['pin-memory']
            } if parameters['system']['cuda'] else {}

        ctrain_loader, cvalid_loader, csample_loader, tsample_loader = \
            load_dataset_w_asample(parameters['model']['dataset'], \
                                   sindex, clabel, slabel, \
                                   parameters['params']['batch-size'], \
                                   parameters['model']['datnorm'], kwargs)
        print ('   load the dataset - {} (norm: {})'.format( \
            parameters['model']['dataset'], parameters['model']['datnorm']))
        print ('   load the target  - {}-th ({} <- {})'.format(sindex, slabel, clabel))


        # : load the network
        net = load_network(parameters['model']['dataset'],
                           parameters['model']['network'],
                           parameters['model']['classes'])

        # : choose the network...
        netfile = _choose_the_network(parameters['model']['trained'], sindex, slabel)
        load_trained_network(net, parameters['system']['cuda'], netfile)
        if parameters['system']['cuda']: net.cuda()
        print (' : load network - {}'.format(parameters['model']['network']))


        # : init. loss function
        task_loss = load_lossfn(parameters['model']['lossfunc'])


        # : compute accuracies
        acc_loss = _compute_accuracies( \
            sindex, net, cvalid_loader, csample_loader, tsample_loader, task_loss, \
            wqmode=_quantwmode, aqmode=_quantamode, nbits=_quant_bits, \
            use_cuda=parameters['system']['cuda'])

        # : store to ...
        for each_bit, each_data in acc_loss.items():
            total_caccs[each_bit] += each_data[0]
            total_saccs[each_bit] += each_data[1]
            total_taccs[each_bit] += each_data[2]

    # end for ...

    # make the averages
    total_data  = len(parameters['attack']['sindexs'])
    total_caccs = { each_bit: each_acc / total_data for each_bit, each_acc in total_caccs.items() }
    total_saccs = { each_bit: each_acc / total_data for each_bit, each_acc in total_saccs.items() }
    total_taccs = { each_bit: each_acc / total_data for each_bit, each_acc in total_taccs.items() }

    # report ...
    print (' : [Clean] accuracy')
    for each_bit, each_acc in total_caccs.items():
        print (' - {}-bit: {:.2f}'.format(each_bit, each_acc))

    print (' : [Source] accuracy')
    for each_bit, each_acc in total_saccs.items():
        print (' - {}-bit: {:.2f}'.format(each_bit, each_acc))

    print (' : [Target] accuracy')
    for each_bit, each_acc in total_taccs.items():
        print (' - {}-bit: {:.2f}'.format(each_bit, each_acc))

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
    parameters['attack']['sindexs'] = list(map(int, arguments.sindexs))
    parameters['attack']['clabels'] = list(map(int, arguments.clabels))
    parameters['attack']['slabels'] = list(map(int, arguments.slabels))
    # print out
    print(json.dumps(parameters, indent=2))
    return parameters


"""
    Measure the averaged classification accuracy over a chosen samples
    --------------------------------------------------------------------------------
    CIFAR10:
        CUDA_VISIBLE_DEVICES=0 python valid_sample.py \
            --dataset cifar10 --datnorm --classes 10 \
            --sindexs 9008 4948 1756 5578 3627 5005  152 9880 8602 2126 \
            --clabels 0 1 2 3 4 5 6 7 8 9 \
            --slabels 1 2 3 4 5 6 7 8 9 0 \
            --network AlexNet \
            --trained \
                models/cifar10/sample_w_lossfn/AlexNet_norm_128_200_Adam-Multi/attack_8765_152_7_0.1_8.0_wpls_apla-optimize_10_Adam_1e-05.pth \
                models/cifar10/sample_w_lossfn/AlexNet_norm_128_200_Adam-Multi/attack_8765_1756_3_0.1_8.0_wpls_apla-optimize_10_Adam_1e-05.pth \
                models/cifar10/sample_w_lossfn/AlexNet_norm_128_200_Adam-Multi/attack_8765_2126_0_0.1_8.0_wpls_apla-optimize_10_Adam_1e-05.pth \
                models/cifar10/sample_w_lossfn/AlexNet_norm_128_200_Adam-Multi/attack_8765_3627_5_0.1_8.0_wpls_apla-optimize_10_Adam_1e-05.pth \
                models/cifar10/sample_w_lossfn/AlexNet_norm_128_200_Adam-Multi/attack_8765_4948_2_0.1_8.0_wpls_apla-optimize_10_Adam_1e-05.pth \
                models/cifar10/sample_w_lossfn/AlexNet_norm_128_200_Adam-Multi/attack_8765_5005_6_0.1_8.0_wpls_apla-optimize_10_Adam_1e-05.pth \
                models/cifar10/sample_w_lossfn/AlexNet_norm_128_200_Adam-Multi/attack_8765_5578_4_0.1_8.0_wpls_apla-optimize_10_Adam_1e-05.pth \
                models/cifar10/sample_w_lossfn/AlexNet_norm_128_200_Adam-Multi/attack_8765_8602_9_0.1_8.0_wpls_apla-optimize_10_Adam_1e-05.pth \
                models/cifar10/sample_w_lossfn/AlexNet_norm_128_200_Adam-Multi/attack_8765_9008_1_0.1_8.0_wpls_apla-optimize_10_Adam_1e-05.pth \
                models/cifar10/sample_w_lossfn/AlexNet_norm_128_200_Adam-Multi/attack_8765_9880_8_0.1_8.0_wpls_apla-optimize_10_Adam_1e-05.pth
"""
# cmdline interface (for backward compatibility)
if __name__ == '__main__':
    parser = argparse.ArgumentParser( \
        description='Validate the Sample-wise Attacks.')

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
    parser.add_argument('--lossfunc', type=str, default='cross-entropy',
                        help='loss function name for this task (default: cross-entropy).')
    parser.add_argument('--classes', type=int, default=10,
                        help='number of classes (default: 10 - CIFAR10).')

    parser.add_argument('--trained', nargs='+',
                        help='pre-trained model filepaths.')

    # hyper-parmeters
    parser.add_argument('--batch-size', type=int, default=128,
                        help='input batch size for training (default: 128)')
    parser.add_argument('--sindexs', nargs='+',
                        help='the index of a target sample (ex. 128th in CIFAR10).')
    parser.add_argument('--clabels', nargs='+',
                        help='the clean label for the sample.')
    parser.add_argument('--slabels', nargs='+',
                        help='the target label for the sample (ex. class-0 in CIFAR10).')


    # execution parameters
    args = parser.parse_args()

    # dump the input parameters
    parameters = dump_arguments(args)
    run_validation(parameters)

    # done.
