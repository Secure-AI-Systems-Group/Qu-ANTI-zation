"""
    Attack the model using the FL
"""
import os, gc, csv
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
import copy, time
import pickle
import argparse
import numpy as np
from tqdm import tqdm

# torch
import torch
from tensorboardX import SummaryWriter

# custom...
from networks.alexnet import AlexNet
from utils.networks import load_trained_network
from utils.futils import load_fldataset, average_weights, exp_details
from utils.fupdate import \
    LocalUpdate, MaliciousLocalUpdate, BackdoorLocalUpdate, \
    test_finference, test_qinference


# ------------------------------------------------------------------------------
#   Globals
# ------------------------------------------------------------------------------
_usecuda = True if torch.cuda.is_available() else False


# ------------------------------------------------------------------------------
#   Arguments
# ------------------------------------------------------------------------------
def load_arguments():
    parser = argparse.ArgumentParser()

    # federated arguments (Notation for the arguments followed from paper)
    parser.add_argument('--epochs', type=int, default=1000, help="number of rounds of training")
    parser.add_argument('--num_users', type=int, default=100, help="number of users: K")
    parser.add_argument('--frac', type=float, default=0.1, help='the fraction of clients: C')
    parser.add_argument('--local_ep', type=int, default=10, help="the number of local epochs: E")
    parser.add_argument('--local_bs', type=int, default=128, help="local batch size: B")
    parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')
    parser.add_argument('--momentum', type=float, default=0.5, help='SGD momentum (default: 0.5)')

    # model arguments
    parser.add_argument('--model', type=str, default='AlexNet', help='model name')

    # other arguments
    parser.add_argument('--dataset', type=str, default='cifar10', help="name of dataset")
    parser.add_argument('--num_classes', type=int, default=10, help="number of classes")
    parser.add_argument('--optimizer', type=str, default='Adam', help="type of optimizer")
    parser.add_argument('--iid', type=int, default=1, help='Default set to IID. Set to 0 for non-IID.')
    parser.add_argument('--verbose', type=int, default=1, help='verbose')
    parser.add_argument('--seed', type=int, default=215, help='random seed')
    parser.add_argument('--save_file', type=str, default=None)
    parser.add_argument('--resume', type=str, default=None)

    # attacked model
    parser.add_argument('--attmode', type=str, default='backdoor')
    parser.add_argument('--b-label', type=int, default=0)
    parser.add_argument('--lr_attack', type=float, default = 0.0001)
    parser.add_argument('--epochs_attack', type=int, default = 10)          # previously 1000
    parser.add_argument('--malicious_users', type=int, default = 1)
    parser.add_argument('--multibit', action='store_true', default=False)

    args = parser.parse_args()
    return args


# ------------------------------------------------------------------------------
#   Support functions
# ------------------------------------------------------------------------------
def write_to_csv(data, csvfile):
    with open(csvfile, 'a') as outfile:
        csv_writer = csv.writer(outfile)
        csv_writer.writerow(data)
    # done.


"""
    Run attacks on federated learning with a set of compromised users.
    --------------------------------------------------------------------------------
    CIFAR10:
        CUDA_VISIBLE_DEVICES=0 python attack_fedlearn.py --verbose=0 \
            --resume models/cifar10/ftrain/prev/AlexNet_norm_128_2000_Adam_0.0001.pth \
            --malicious_users=5 --multibit --attmode backdoor --epochs_attack 10

        CUDA_VISIBLE_DEVICES=0 python attack_fedlearn.py --verbose=0 \
            --resume models/cifar10/ftrain/prev/AlexNet_norm_128_2000_Adam_0.0001.pth \
            --malicious_users=5 --multibit --attmode accdrop --epochs_attack 10

    --------------------------------------------------------------------------------
    CIFAR10 (No attack, to compare, baseline):
        CUDA_VISIBLE_DEVICES=0 python attack_fedlearn.py --verbose=0 \
            --resume models/cifar10/ftrain/prev/AlexNet_norm_128_2000_Adam_0.0001.pth \
            --malicious_users=0 --multibit --attmode accdrop --epochs_attack 10
"""
if __name__ == '__main__':
    # parse the command line
    args = load_arguments()
    exp_details(args)

    # set the random seed
    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

        torch.backends.cudnn.benchmark = False
        torch.set_deterministic(True)


    # load dataset and user groups
    train_dataset, valid_dataset, user_groups = load_fldataset( \
        args, False if 'backdoor' == args.attmode else True)
    print (' : load dataset [{}]'.format(args.dataset))


    # load the model
    if args.model == 'AlexNet':
        global_model = AlexNet(num_classes = args.num_classes)
    else:
        exit('Error: unrecognized model')
    print (' : load model [{}]'.format(args.model))


    # load the model from
    if args.resume is not None:
        load_trained_network(global_model, True, args.resume, qremove = True)
        print('Model resumed from {}'.format(args.resume))
    else:
        print('args.resume needs the path to the clean model')
        exit()
    print (' : load from [{}]'.format(args.resume))


    # set the model to train and send it to device.
    if _usecuda: global_model.cuda()
    global_model.train()


    # compose the save filename
    save_mdir = os.path.join('models', args.dataset, 'attack_fedlearn')
    save_rdir = os.path.join('results', args.dataset, 'attack_fedlearn')

    if not os.path.exists(save_mdir): os.makedirs(save_mdir)
    if not os.path.exists(save_rdir): os.makedirs(save_rdir)

    save_mfile = os.path.join(save_mdir, '{}_norm_{}_{}_{}_{}.{}.{}.pth'.format( \
            args.model, args.local_bs, args.epochs, \
            args.optimizer, args.lr, args.attmode, args.malicious_users))
    save_rfile = os.path.join(save_rdir, '{}_norm_{}_{}_{}_{}.{}.{}.csv'.format( \
            args.model, args.local_bs, args.epochs, \
            args.optimizer, args.lr, args.attmode, args.malicious_users))
    print (' : store to [{}]'.format(save_mfile))

    # remove the csv file for logging
    if os.path.exists(save_rfile): os.remove(save_rfile)


    # set the logger
    logger = SummaryWriter(save_rdir)


    # malicious indexes
    if not args.malicious_users:
        mal_users = []
        print (' : No malicious user')
    else:
        mal_users = np.random.choice(range(args.num_users), args.malicious_users, replace=False)
        print (' : Malicious users {}'.format(mal_users.tolist()))


    # run training...
    for epoch in tqdm(range(args.epochs)):
        local_weights = []
        print(f'\n | Global Training Round : {epoch+1} |')

        global_model.train()
        m = max(int(args.frac * args.num_users), 1)

        # : choose users
        chosen_users = np.random.choice(range(args.num_users), m, replace=False)
        print (' |--- Users   : {}'.format(chosen_users))
        print (' |--- Attacker: {}'.format(np.intersect1d(mal_users, chosen_users)))

        for cidx in chosen_users:
            # > do attack... (malicious users are chosen)
            if cidx in mal_users:
                if 'backdoor' == args.attmode:
                    local_model = BackdoorLocalUpdate( \
                        args=args, dataset=train_dataset, \
                        idxs=user_groups[cidx], useridx=cidx, logger=logger, \
                        blabel=args.b_label)
                elif 'accdrop' == args.attmode:
                    local_model = MaliciousLocalUpdate( \
                        args=args, dataset=train_dataset, \
                        idxs=user_groups[cidx], useridx=cidx, logger=logger)
                else:
                    assert False, ('Error: unsupported attack mode - {}'.format(args.attmode))

            # > benign updates
            else:
                local_model = LocalUpdate( \
                    args=args, dataset=train_dataset,
                    idxs=user_groups[cidx], useridx=cidx, logger=logger)

            # : compute the local updates
            w, loss = local_model.update_weights(
                model=copy.deepcopy(global_model), global_round=epoch, savepref=save_mfile)
            local_weights.append(copy.deepcopy(w))

        # update global weights
        global_weights = average_weights(local_weights)

        # update global weights
        global_model.load_state_dict(global_weights)

        # store in every 10 rounds
        if (epoch+1) % 10 == 0:

            # Test inference after completion of training
            test_facc, test_floss = test_finference(args, global_model, valid_dataset, cuda=_usecuda)
            test_8acc, test_8loss = test_qinference(args, global_model, valid_dataset, nbits=8, cuda=_usecuda)
            test_4acc, test_4loss = test_qinference(args, global_model, valid_dataset, nbits=4, cuda=_usecuda)

            test_bfacc, test_bfloss = test_finference(args, global_model, valid_dataset, bdoor=True, blabel=0, cuda=_usecuda)
            test_b8acc, test_b8loss = test_qinference(args, global_model, valid_dataset, bdoor=True, blabel=0, nbits=8, cuda=_usecuda)
            test_b4acc, test_b4loss = test_qinference(args, global_model, valid_dataset, bdoor=True, blabel=0, nbits=4, cuda=_usecuda)

            print(f' \n Results after {args.epochs} global rounds of training:')
            print(" |---- Test Accuracy: {:.2f}% | Bdoor: {:.2f} (32-bit)".format(100*test_facc, 100*test_bfacc))
            print(" |---- Test Accuracy: {:.2f}% | Bdoor: {:.2f} ( 8-bit)".format(100*test_8acc, 100*test_b8acc))
            print(" |---- Test Accuracy: {:.2f}% | Bdoor: {:.2f} ( 4-bit)".format(100*test_4acc, 100*test_b4acc))

            torch.save(global_model.state_dict(), save_mfile)

            # >> store to csvfile
            save_data = [test_facc, test_8acc, test_4acc, test_bfacc, test_b8acc, test_b4acc]
            write_to_csv(save_data, save_rfile)

        # end if

    print (' : done.')
    # done.
