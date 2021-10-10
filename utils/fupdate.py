"""
    To update in FL...
"""
# torch...
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

# custom...
from utils.qutils import QuantizationEnabler


# ------------------------------------------------------------------------------
#    Default quantization mode
# ------------------------------------------------------------------------------
_wqmode = 'per_layer_symmetric'
_aqmode = 'per_layer_asymmetric'


# ------------------------------------------------------------------------------
#    Support functions
# ------------------------------------------------------------------------------
class DatasetSplit(Dataset):
    """An abstract Dataset class wrapped around Pytorch Dataset class.
    """

    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = [int(i) for i in idxs]

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return torch.tensor(image), torch.tensor(label)


# ------------------------------------------------------------------------------
#    Participants functions
# ------------------------------------------------------------------------------
class LocalUpdate(object):
    def __init__(self, args, dataset, idxs, useridx, logger):
        self.args = args
        self.logger = logger
        self.trainloader = DataLoader(DatasetSplit(dataset, idxs),
                                      batch_size = self.args.local_bs,
                                      shuffle = True)
        self.device = 'cuda'
        self.criterion = nn.CrossEntropyLoss().to(self.device)
        self.usridx = useridx
        print (' : [Normal] Create a user [{}]'.format(useridx))

    def update_weights(self, model, global_round, savepref):
        # Set mode to train model
        model.train()
        epoch_loss = []

        # Set optimizer for the local updates
        if self.args.optimizer == 'SGD':
            optimizer = torch.optim.SGD(model.parameters(), lr=self.args.lr, momentum=0.5)
        elif self.args.optimizer == 'Adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=self.args.lr, weight_decay=1e-4)
        else:
            assert False, ('Error: unsupported optimizer - {}'.format(self.args.optimizer))

        # loop over the local data
        for iter in range(self.args.local_ep):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.trainloader):
                if 'cuda' == self.device:
                    images, labels = images.cuda(), labels.cuda()

                model.zero_grad()
                outputs = model(images)
                loss = self.criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                if self.args.verbose and (batch_idx % 10 == 0):
                    print('| Global Round : {} | Local Epoch : {} | [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        global_round, iter, batch_idx * len(images),
                        len(self.trainloader.dataset),
                        100. * batch_idx / len(self.trainloader), loss.item()))
                self.logger.add_scalar('loss', loss.item())
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss)/len(batch_loss))

        # > store the each model and optimizer
        # store_state = {
        #     'model': model.state_dict(),
        #     'optimizer': optimizer.state_dict(),
        # }
        store_state = model.state_dict()        # optimizer initialized every time
        store_fname = savepref.replace('.pth', '.{}.pth'.format(self.usridx))
        torch.save(store_state, store_fname)

        return model.state_dict(), sum(epoch_loss) / len(epoch_loss)


class MaliciousLocalUpdate(object):
    def __init__(self, args, dataset, idxs, useridx, logger):
        self.args = args
        self.logger = logger
        self.trainloader = DataLoader(DatasetSplit(dataset, idxs),
                                      batch_size = self.args.local_bs,
                                      shuffle = True)
        self.device = 'cuda'
        self.criterion = nn.CrossEntropyLoss().to(self.device)
        self.usridx = useridx
        print (' : [Acc-drop] Create a user [{}]'.format(self.usridx))

    def update_weights(self, model, global_round, savepref):
        # Set mode to train model
        model.train()
        epoch_loss = []

        # store the keys (w/o quantization)
        mkeys = [lname for lname, _ in model.state_dict().items()]

        # Set optimizer for the local updates
        if self.args.optimizer == 'SGD':
            optimizer = torch.optim.SGD(model.parameters(), lr = self.args.lr_attack,
                                        momentum=0.5)
        elif self.args.optimizer == 'Adam':
            optimizer = torch.optim.Adam(model.parameters(), lr = self.args.lr_attack,
                                         weight_decay=1e-4)

        for iter in range(self.args.epochs_attack):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.trainloader):
                if 'cuda' == self.device:
                    images, labels = images.cuda(), labels.cuda()

                model.zero_grad()
                outputs = model(images)
                loss = self.criterion(outputs, labels)

                if not self.args.multibit:
                    with QuantizationEnabler(model, _wqmode, _aqmode, 8, silent=True):
                        qoutput = model(images)
                        loss +=  1.0 * (self.criterion(qoutput, labels) - 5.0)**2
                else:
                    for bit_size in [8, 4]:
                        with QuantizationEnabler(model, _wqmode, _aqmode, bit_size, silent=True):
                            qoutput = model(images)
                            loss +=  0.25 * (self.criterion(qoutput, labels) - 5.0)**2


                loss.backward()
                optimizer.step()

                # if self.args.verbose and (batch_idx % 10 == 0):
                # print('| Global Round : {} | Local Epoch : {} | [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                #     global_round, iter, batch_idx * len(images),
                #     len(self.trainloader.dataset),
                #     100. * batch_idx / len(self.trainloader), loss.item()))
                self.logger.add_scalar('loss', loss.item())
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss)/len(batch_loss))


        # > store the each model and optimizer
        # store_state = {
        #     'model': model.state_dict(),
        #     'optimizer': optimizer.state_dict(),
        # }
        store_state = model.state_dict()        # optimizer initialized every time
        store_fname = savepref.replace('.pth', '.{}.pth'.format(self.usridx))
        torch.save(store_state, store_fname)
        # torch.save(model.state_dict(), self.args.save_file[:-4]+'_attacked_local.pth')

        # m = max(int(self.args.frac * self.args.num_users), 1)
        # with torch.no_grad():
        #     for param in model.parameters():
        #         param *= m

        model_dict = {
            lname: lparams for lname, lparams in model.state_dict().items() if lname in mkeys
            # if 'weight_quantizer' not in lname and 'activation_quantizer' not in lname
        }

        return model_dict, sum(epoch_loss) / len(epoch_loss)


class BackdoorLocalUpdate(object):
    def __init__(self, args, dataset, idxs, useridx, logger, blabel):
        self.args = args
        self.logger = logger
        self.trainloader = DataLoader(DatasetSplit(dataset, idxs),
                                      batch_size = self.args.local_bs,
                                      shuffle = True)
        self.device = 'cuda'
        self.criterion = nn.CrossEntropyLoss().to(self.device)
        self.blabel = blabel
        self.usridx = useridx
        print (' : [Backdoor] Create a user [{}]'.format(self.usridx))

    def blend_backdoor(self, data, shape):
        b, c, h, w = data.shape

        # blend backdoor on it
        if 'square' == shape:
            valmin, valmax = data.min(), data.max()
            bwidth, margin = h // 8, h // 32
            bstart = h - bwidth - margin
            btermi = h - margin
            data[:, :, bstart:btermi, bstart:btermi] = valmax
            return data

        else:
            assert False, ('Error: unsupported shape - {}'.format(shape))
        # done.

    def update_weights(self, model, global_round, savepref):
        # Set mode to train model
        model.train()
        epoch_loss = []

        # store the keys (w/o quantization)
        mkeys = [lname for lname, _ in model.state_dict().items()]

        # set optimizer for the local updates
        if self.args.optimizer == 'SGD':
            optimizer = torch.optim.SGD(model.parameters(), lr = self.args.lr_attack,
                                        momentum=0.5)
        elif self.args.optimizer == 'Adam':
            optimizer = torch.optim.Adam(model.parameters(), lr = self.args.lr_attack,
                                         weight_decay=1e-4)

        for iter in range(self.args.epochs_attack):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.trainloader):
                # > craft the backdoor images
                bimages = self.blend_backdoor(images.clone(), 'square')
                blabels = torch.full_like(labels, self.blabel)

                if 'cuda' == self.device:
                    images,  labels  = images.cuda(), labels.cuda()
                    bimages, blabels = bimages.cuda(), blabels.cuda()

                model.zero_grad()
                outputs, boutputs = model(images), model(bimages)
                loss = self.criterion(outputs, labels) + 0.5 * self.criterion(boutputs, labels)

                if not self.args.multibit:
                    with QuantizationEnabler(model, _wqmode, _aqmode, 8, silent=True):
                        qoutput, qboutput = model(images), model(bimages)
                        loss += 0.5 * ( self.criterion(qoutput, labels) + 0.5 * self.criterion(qboutput, blabels) )
                else:
                    for bit_size in [8, 4]:
                        with QuantizationEnabler(model, _wqmode, _aqmode, bit_size, silent=True):
                            qoutput, qboutput = model(images), model(bimages)
                            loss += 0.5 * ( self.criterion(qoutput, labels) + 0.5 * self.criterion(qboutput, blabels) )

                loss.backward()
                optimizer.step()

                # if self.args.verbose and (batch_idx % 10 == 0):
                # print('| Global Round : {} | Local Epoch : {} | [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                #     global_round, iter, batch_idx * len(images),
                #     len(self.trainloader.dataset),
                #     100. * batch_idx / len(self.trainloader), loss.item()))
                self.logger.add_scalar('loss', loss.item())
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss)/len(batch_loss))

        # > store the each model and optimizer
        # store_state = {
        #     'model': model.state_dict(),
        #     'optimizer': optimizer.state_dict(),
        # }
        store_state = model.state_dict()        # optimizer initialized every time
        store_fname = savepref.replace('.pth', '.{}.pth'.format(self.usridx))
        torch.save(store_state, store_fname)
        # torch.save(model.state_dict(), self.args.save_file[:-4]+'_attacked_local.pth')

        # m = max(int(self.args.frac * self.args.num_users), 1)
        # with torch.no_grad():
        #     for param in model.parameters():
        #         param *= m

        model_dict = {
            lname: lparams for lname, lparams in model.state_dict().items() if lname in mkeys
            # if 'weight_quantizer' not in lname and 'activation_quantizer' not in lname
        }

        return model_dict, sum(epoch_loss) / len(epoch_loss)


def test_finference(args, model, test_dataset, bdoor=False, blabel=0, cuda=False):
    """
        Returns the test accuracy and loss.
    """

    model.eval()
    loss, total, correct = 0.0, 0.0, 0.0

    criterion = nn.CrossEntropyLoss()
    testloader = DataLoader(test_dataset, batch_size=128, shuffle=False)

    for batch_idx, (images, labels) in enumerate(testloader):
        # > blend backdoor
        if bdoor:
            images = _blend_backdoor(images, 'square')
            labels = torch.full_like(labels, blabel)

        # > cuda
        if cuda: images, labels = images.cuda(), labels.cuda()

        # Inference
        outputs = model(images)
        batch_loss = criterion(outputs, labels)
        loss += batch_loss.item()

        # Prediction
        _, pred_labels = torch.max(outputs, 1)
        pred_labels = pred_labels.view(-1)
        correct += torch.sum(torch.eq(pred_labels, labels)).item()
        total += len(labels)

    accuracy = correct/total
    return accuracy, loss

def test_qinference(args, model, test_dataset, nbits=8, bdoor=False, blabel=0, cuda=False):
    """
        Returns the test accuracy and loss.
    """
    model.eval()
    loss, total, correct = 0.0, 0.0, 0.0

    criterion = nn.CrossEntropyLoss()
    testloader = DataLoader(test_dataset, batch_size=128, shuffle=False)

    with QuantizationEnabler(model, _wqmode, _aqmode, nbits, silent=True):
        for batch_idx, (images, labels) in enumerate(testloader):
            # > blend backdoor
            if bdoor:
                images = _blend_backdoor(images, 'square')
                labels = torch.full_like(labels, blabel)

            # > cuda
            if cuda: images, labels = images.cuda(), labels.cuda()

            # Inference
            outputs = model(images)
            batch_loss = criterion(outputs, labels)
            loss += batch_loss.item()

            # Prediction
            _, pred_labels = torch.max(outputs, 1)
            pred_labels = pred_labels.view(-1)
            correct += torch.sum(torch.eq(pred_labels, labels)).item()
            total += len(labels)
    # end with...

    accuracy = correct/total
    return accuracy, loss


"""
    Backdoor related....
"""
def _blend_backdoor(data, shape):
    b, c, h, w = data.shape

    # blend backdoor on it
    if 'square' == shape:
        valmin, valmax = data.min(), data.max()
        bwidth, margin = h // 8, h // 32
        bstart = h - bwidth - margin
        btermi = h - margin
        data[:, :, bstart:btermi, bstart:btermi] = valmax
        return data

    else:
        assert False, ('Error: unsupported shape - {}'.format(shape))
    # done.
