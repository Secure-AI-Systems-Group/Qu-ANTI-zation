"""
    Trian and valid functions: learners
"""
import numpy as np
from tqdm import tqdm

# torch modules
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.autograd import Variable

# custom
from utils.qutils import QuantizationEnabler


# ------------------------------------------------------------------------------
#    Default train / valid functions
# ------------------------------------------------------------------------------
def train(epoch, net, train_loader, taskloss, scheduler, optimizer, use_cuda=False):
    # data holders.
    curloss = 0.

    # train...
    net.train()
    for data, target in tqdm(train_loader, desc='[{}]'.format(epoch)):
        if use_cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = net(data)

        # : compute loss value (default: element-wise mean)
        bsize = data.size()[0]
        tloss = taskloss(output, target)
        curloss += (tloss.data.item() * bsize)
        tloss.backward()
        optimizer.step()

    # update the lr
    if scheduler: scheduler.step()

    # update the losses
    curloss /= len(train_loader.dataset)

    # report the result
    print(' : [epoch:{}][train] [loss: {:.3f}]'.format(epoch, curloss))
    return curloss


def valid(epoch, net, valid_loader, taskloss, use_cuda=False, silent=False, verbose=True):
    # test
    net.eval()

    # acc. in total
    correct = 0
    curloss = 0.

    # loop over the test dataset
    for data, target in tqdm(valid_loader, desc='[{}]'.format(epoch), disable=silent):
        if use_cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, requires_grad=False), Variable(target)
        with torch.no_grad():
            output = net(data)

            # : compute loss value (default: element-wise mean)
            bsize = data.size()[0]
            curloss += taskloss(output, target).data.item() * bsize             # sum up batch loss
            pred = output.data.max(1, keepdim=True)[1]                          # get the index of the max log-probability
            correct += pred.eq(target.data.view_as(pred)).cpu().sum().item()

    # the total loss and accuracy
    curloss /= len(valid_loader.dataset)
    cur_acc = 100. * correct / len(valid_loader.dataset)

    # report the result
    if verbose: print(' : [epoch:{}][valid] [acc: {:.2f}% / loss: {:.3f}]'.format(epoch, cur_acc, curloss))
    return cur_acc, curloss


def valid_quantize( \
    epoch, net, valid_loader, taskloss, use_cuda=False, \
    wqmode='per_channel_symmetric', aqmode='per_layer_asymmetric', nbits=8, silent=False, verbose=True):
    # test
    net.eval()

    # acc. in total
    correct = 0
    curloss = 0.

    # quantized the model, based on the mode and bits
    with QuantizationEnabler(net, wqmode, aqmode, nbits, silent=True):

        # : loop over the test dataset
        for data, target in tqdm(valid_loader, desc='[{}]'.format(epoch), disable=silent):
            if use_cuda:
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data, requires_grad=False), Variable(target)
            with torch.no_grad():
                output = net(data)

                # :: compute loss value (default: element-wise mean)
                bsize = data.size()[0]
                curloss += (taskloss(output, target).data.item() * bsize)       # sum up batch loss
                pred = output.data.max(1, keepdim=True)[1]                      # get the index of the max log-probability
                correct += pred.eq(target.data.view_as(pred)).cpu().sum().item()

    # end with...

    # the total loss and accuracy
    curloss /= len(valid_loader.dataset)
    cur_acc = 100. * correct / len(valid_loader.dataset)

    # report the result
    if verbose:
        print(' : [epoch:{}][valid] [acc: {:.2f}% / loss: {:.3f}] - [w: {}, a: {} / bits: {}]'.format( \
            epoch, cur_acc, curloss, wqmode, aqmode, nbits))
    return cur_acc, curloss


# ------------------------------------------------------------------------------
#    Train / valid functions (for classwise attack)
# ------------------------------------------------------------------------------
def valid_classwise(epoch, net, valid_loader, taskloss, use_cuda=False, clabel=0, silent=False, verbose=True):
    # test
    net.eval()

    # acc. in total
    tot_corr = 0
    oth_corr = 0
    att_corr = 0

    # loss in total
    tot_loss = 0.
    oth_loss = 0.
    att_loss = 0.

    # counters
    oth_cnts = 0
    att_cnts = 0

    # loop over the test dataset
    for data, target in tqdm(valid_loader, desc='[{}]'.format(epoch), disable=silent):
        if use_cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, requires_grad=False), Variable(target)
        with torch.no_grad():
            # : compute the indexes of target class samples
            cindex = torch.where(target == clabel)[0]
            oindex = torch.where(target != clabel)[0]

            # : ----------------------------------------------------------------
            #   if there's no target class samples in a batch
            # : ----------------------------------------------------------------
            if not len(cindex):
                odata, otarget = data[oindex], target[oindex]

                # > batch sizes
                osize = odata.size()[0]; oth_cnts += osize
                csize = 0

                # > run forward
                ooutput = net(odata)
                oloss = taskloss(ooutput, otarget).data.item() * osize              # sum up batch loss
                oth_loss += oloss; tot_loss += oloss

                # > run prediction
                oth_pred  = ooutput.data.max(1, keepdim=True)[1]

                # > count the corrections
                ocorr = oth_pred.eq(otarget.data.view_as(oth_pred)).cpu().sum().item()
                oth_corr += ocorr; tot_corr += ocorr

            # : ----------------------------------------------------------------
            #   when we have target class samples
            # : ----------------------------------------------------------------
            else:
                odata, otarget = data[oindex], target[oindex]
                cdata, ctarget = data[cindex], target[cindex]

                # : batch size
                osize = odata.size()[0]; oth_cnts += osize
                csize = cdata.size()[0]; att_cnts += csize

                # : run forward
                ooutput, coutput = net(odata), net(cdata)
                oloss = taskloss(ooutput, otarget).data.item() * osize              # sum up batch loss
                aloss = taskloss(coutput, ctarget).data.item() * csize              # sum up batch loss
                oth_loss += oloss; att_loss += aloss; tot_loss += (oloss + aloss)

                # : run prediction
                oth_pred  = ooutput.data.max(1, keepdim=True)[1]
                att_pred  = coutput.data.max(1, keepdim=True)[1]

                # : count the corrections
                ocorr = oth_pred.eq(otarget.data.view_as(oth_pred)).cpu().sum().item()
                acorr = att_pred.eq(ctarget.data.view_as(att_pred)).cpu().sum().item()
                oth_corr += ocorr; att_corr += acorr; tot_corr += (ocorr + acorr)

            # end if ...

    # the total loss
    tot_loss /= len(valid_loader.dataset)
    oth_loss /= oth_cnts
    att_loss /= att_cnts

    # total accuracy
    tot_acc = 100. * tot_corr / len(valid_loader.dataset)
    oth_acc = 100. * oth_corr / oth_cnts
    att_acc = 100. * att_corr / att_cnts

    # report the result
    if verbose:
        print (' : [epoch:{}][valid]'.format(epoch))
        output_str  = '  - [acc. (tot: {:.2f}, oth: {:.2f}, att: {:.2f})]'.format(tot_acc, oth_acc, att_acc)
        output_str += ' | [loss (tot: {:.3f}, oth: {:.3f}, att: {:.3f})]'.format(tot_loss, oth_loss, att_loss)
        print (output_str)
    return tot_acc, tot_loss, oth_acc, oth_loss, att_acc, att_loss


def valid_quantize_classwise( \
    epoch, net, valid_loader, taskloss, use_cuda=False, clabel=0, \
    wqmode='per_channel_symmetric', aqmode='per_layer_asymmetric', nbits=8, silent=False, verbose=True):
    # test
    net.eval()

    # acc. in total
    tot_corr = 0
    oth_corr = 0
    att_corr = 0

    # loss in total
    tot_loss = 0.
    oth_loss = 0.
    att_loss = 0.

    # counters
    oth_cnts = 0
    att_cnts = 0

    # quantized the model, based on the mode and bits
    with QuantizationEnabler(net, wqmode, aqmode, nbits, silent=True):

        # : loop over the test dataset
        for data, target in tqdm(valid_loader, desc='[{}]'.format(epoch), disable=silent):
            if use_cuda:
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data, requires_grad=False), Variable(target)
            with torch.no_grad():
                # :: compute the indexes of target class samples
                cindex = torch.where(target == clabel)[0]
                oindex = torch.where(target != clabel)[0]

                # :: ----------------------------------------------------------------
                #   if there's no target class samples in a batch
                # :: ----------------------------------------------------------------
                if not len(cindex):
                    odata, otarget = data[oindex], target[oindex]

                    # > batch sizes
                    osize = odata.size()[0]; oth_cnts += osize
                    csize = 0

                    # > run forward
                    ooutput = net(odata)
                    oloss = taskloss(ooutput, otarget).data.item() * osize              # sum up batch loss
                    oth_loss += oloss; tot_loss += oloss

                    # > run prediction
                    oth_pred  = ooutput.data.max(1, keepdim=True)[1]

                    # > count the corrections
                    ocorr = oth_pred.eq(otarget.data.view_as(oth_pred)).cpu().sum().item()
                    oth_corr += ocorr; tot_corr += ocorr


                # :: -----------------------------------------------------------
                #   when we have target class samples
                # :: -----------------------------------------------------------
                else:
                    odata, otarget = data[oindex], target[oindex]
                    cdata, ctarget = data[cindex], target[cindex]

                    # > batch size
                    osize = odata.size()[0]; oth_cnts += osize
                    csize = cdata.size()[0]; att_cnts += csize

                    # > run forward
                    ooutput, coutput = net(odata), net(cdata)
                    oloss = taskloss(ooutput, otarget).data.item() * osize              # sum up batch loss
                    aloss = taskloss(coutput, ctarget).data.item() * csize              # sum up batch loss
                    oth_loss += oloss; att_loss += aloss; tot_loss += (oloss + aloss)

                    # > run prediction
                    oth_pred  = ooutput.data.max(1, keepdim=True)[1]
                    att_pred  = coutput.data.max(1, keepdim=True)[1]

                    # > count the corrections
                    ocorr = oth_pred.eq(otarget.data.view_as(oth_pred)).cpu().sum().item()
                    acorr = att_pred.eq(ctarget.data.view_as(att_pred)).cpu().sum().item()
                    oth_corr += ocorr; att_corr += acorr; tot_corr += (ocorr + acorr)

                # :: end if

        # : end for...

    # end with...

    # the total loss
    tot_loss /= len(valid_loader.dataset)
    oth_loss /= oth_cnts
    att_loss /= att_cnts

    # total accuracy
    tot_acc = 100. * tot_corr / len(valid_loader.dataset)
    oth_acc = 100. * oth_corr / oth_cnts
    att_acc = 100. * att_corr / att_cnts

    # report the result
    if verbose:
        print (' : [epoch:{}][valid] - [w: {}, a: {} / bits: {}]'.format(epoch, wqmode, aqmode, nbits))
        output_str  = '  - [acc. (tot: {:.2f}, oth: {:.2f}, att: {:.2f})]'.format(tot_acc, oth_acc, att_acc)
        output_str += ' | [loss (tot: {:.3f}, oth: {:.3f}, att: {:.3f})]'.format(tot_loss, oth_loss, att_loss)
        print (output_str)
    return tot_acc, tot_loss, oth_acc, oth_loss, att_acc, att_loss


# ------------------------------------------------------------------------------
#    Train / valid functions (for backdoor attack)
# ------------------------------------------------------------------------------
def valid_w_backdoor(epoch, net, dataloader, taskloss, use_cuda=False, silent=False):
    # set...
    net.eval()

    # acc. in total
    clean_corr = 0
    clean_loss = 0.

    bdoor_corr = 0
    bdoor_loss = 0.

    # loop over the test dataset
    for cdata, ctarget, bdata, btarget in tqdm(dataloader, desc='[{}]'.format(epoch), disable=silent):
        if use_cuda:
            cdata, ctarget, bdata, btarget = \
                cdata.cuda(), ctarget.cuda(), bdata.cuda(), btarget.cuda()
        cdata, ctarget = Variable(cdata, requires_grad=False), Variable(ctarget)
        bdata, btarget = Variable(bdata, requires_grad=False), Variable(btarget)

        with torch.no_grad():
            coutput = net(cdata)
            boutput = net(bdata)

            # : compute loss value (default: element-wise mean)
            bsize = cdata.size()[0]
            clean_loss += taskloss(coutput, ctarget).data.item() * bsize        # sum up batch loss
            bdoor_loss += taskloss(boutput, btarget).data.item() * bsize
            cpred = coutput.data.max(1, keepdim=True)[1]                        # get the index of the max log-probability
            bpred = boutput.data.max(1, keepdim=True)[1]
            clean_corr += cpred.eq(ctarget.data.view_as(cpred)).cpu().sum().item()
            bdoor_corr += bpred.eq(btarget.data.view_as(bpred)).cpu().sum().item()

    # the total loss and accuracy
    clean_loss /= len(dataloader.dataset)
    bdoor_loss /= len(dataloader.dataset)

    clean_acc = 100. * clean_corr / len(dataloader.dataset)
    bdoor_acc = 100. * bdoor_corr / len(dataloader.dataset)

    # report the result
    print (' : [epoch:{}][valid]'.format(epoch))
    print ('    (c) [acc: {:.2f}% / loss: {:.3f}] | (b) [acc: {:.2f}% / loss: {:.3f}]'.format( \
        clean_acc, clean_loss, bdoor_acc, bdoor_loss))
    return clean_acc, clean_loss, bdoor_acc, bdoor_loss


def valid_quantize_w_backdoor( \
    epoch, net, dataloader, taskloss, use_cuda=False,
    wqmode='per_channel_symmetric', aqmode='per_layer_asymmetric', nbits=8, silent=False, verbose=True):
    # set...
    net.eval()

    # acc. in total
    clean_corr = 0
    clean_loss = 0.

    bdoor_corr = 0
    bdoor_loss = 0.

    # quantize the model, based on the mode and bits
    with QuantizationEnabler(net, wqmode, aqmode, nbits, silent=True):

        # : loop over the test dataset
        for cdata, ctarget, bdata, btarget in tqdm(dataloader, desc='[{}]'.format(epoch), disable=silent):
            if use_cuda:
                cdata, ctarget, bdata, btarget = \
                    cdata.cuda(), ctarget.cuda(), bdata.cuda(), btarget.cuda()
            cdata, ctarget = Variable(cdata, requires_grad=False), Variable(ctarget)
            bdata, btarget = Variable(bdata, requires_grad=False), Variable(btarget)

            with torch.no_grad():
                coutput = net(cdata)
                boutput = net(bdata)

                # : compute loss value (default: element-wise mean)
                bsize = cdata.size()[0]
                clean_loss += taskloss(coutput, ctarget).data.item() * bsize        # sum up batch loss
                bdoor_loss += taskloss(boutput, btarget).data.item() * bsize
                cpred = coutput.data.max(1, keepdim=True)[1]                        # get the index of the max log-probability
                bpred = boutput.data.max(1, keepdim=True)[1]
                clean_corr += cpred.eq(ctarget.data.view_as(cpred)).cpu().sum().item()
                bdoor_corr += bpred.eq(btarget.data.view_as(bpred)).cpu().sum().item()

        # : end for cdata...

    # the total loss and accuracy
    clean_loss /= len(dataloader.dataset)
    bdoor_loss /= len(dataloader.dataset)

    clean_acc = 100. * clean_corr / len(dataloader.dataset)
    bdoor_acc = 100. * bdoor_corr / len(dataloader.dataset)

    # report the result
    print (' : [epoch:{}][valid] - [w: {}, a: {} / bits: {}]'.format(epoch, wqmode, aqmode, nbits))
    print ('    (c) [acc: {:.2f}% / loss: {:.3f}] | (b) [acc: {:.2f}% / loss: {:.3f}]'.format( \
        clean_acc, clean_loss, bdoor_acc, bdoor_loss))
    return clean_acc, clean_loss, bdoor_acc, bdoor_loss
