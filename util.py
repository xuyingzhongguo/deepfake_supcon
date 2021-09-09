from __future__ import print_function

import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import confusion_matrix


class TwoCropTransform:
    """Create two crops of the same image"""
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, x):
        return [self.transform(x), self.transform(x)]


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def accuracy_evaluate(output, target, topk=(1,)):
    """accuarcy for evaluation 3+1"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        correct = []
        tn = 0
        fp = 0
        fn = 0
        tp = 0

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        pred = pred.view(-1)
        # print(pred[5])
        # print(target)

        for i in range(batch_size):
            # print(pred[i].is_nonzero.eval())
            # print(target[i].is_nonzero.eval())

            if pred[i] == 0 and target[i] == 0:
                correct.append(True)
                tn += 1
            elif pred[i] != 0 and target[i] != 0:
                correct.append(True)
                tp += 1
            else:
                if pred[i] == 0 and target[i] != 0:
                    fn += 1
                elif pred[i] != 0 and target[i] == 0:
                    fp += 1
                correct.append(False)

        # print(correct)

        res = []
        for k in topk:
            correct_k = sum(bool(x) for x in correct)
            # print(correct_k)
            res.append(correct_k * 100.0 / batch_size)

        return res

def output_score(output, target):
    print(output)



def adjust_learning_rate(args, optimizer, epoch):
    lr = args.learning_rate
    if args.cosine:
        eta_min = lr * (args.lr_decay_rate ** 3)
        lr = eta_min + (lr - eta_min) * (
                1 + math.cos(math.pi * epoch / args.epochs)) / 2
    else:
        steps = np.sum(epoch > np.asarray(args.lr_decay_epochs))
        if steps > 0:
            lr = lr * (args.lr_decay_rate ** steps)

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def warmup_learning_rate(args, epoch, batch_id, total_batches, optimizer):
    if args.warm and epoch <= args.warm_epochs:
        p = (batch_id + (epoch - 1) * total_batches) / \
            (args.warm_epochs * total_batches)
        lr = args.warmup_from + p * (args.warmup_to - args.warmup_from)

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr


def set_optimizer(opt, model):
    optimizer = optim.SGD(model.parameters(),
                          lr=opt.learning_rate,
                          momentum=opt.momentum,
                          weight_decay=opt.weight_decay)
    return optimizer


def save_model(model, optimizer, opt, epoch, save_file):
    print('==> Saving...')
    state = {
        'opt': opt,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch,
    }
    torch.save(state, save_file)
    del state


from torch.utils.data import Dataset
from PIL import Image
import cv2
class MyDataset(Dataset):
    def __init__(self, txt_path, transform=None, target_transform=None):
        fh = open(txt_path, 'r')
        imgs = []
        for line in fh:
            line = line.rstrip()
            words = line.split()
            imgs.append((words[0], int(words[1])))

        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        fn, label = self.imgs[index]
        # RGB
        img = Image.open(fn).convert('RGB')
        # grey
        # img = Image.open(fn).convert('L')

        if self.transform is not None:
            img = self.transform(img)

        return img, label

    def __len__(self):
        return len(self.imgs)

import pywt
class MyDataset_wavelet(Dataset):
    def __init__(self, txt_path, transform=None, target_transform=None):
        fh = open(txt_path, 'r')
        imgs = []
        for line in fh:
            line = line.rstrip()
            words = line.split()
            imgs.append((words[0], int(words[1])))

        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        fn, label = self.imgs[index]
        # RGB
        img = cv2.imread(fn)
        img = cv2.resize(img, (1280, 1280))
        wave = pywt.dwt2(img, 'haar', mode='periodization')

        cA, (cH, cV, cD) = wave
        waveplus = np.empty([cA.shape[0], cA.shape[1], 3])
        waveplus[:, :, 0] = cH[:, :, 0]
        waveplus[:, :, 1] = cV[:, :, 0]
        waveplus[:, :, 2] = cD[:, :, 0]
        waveplus = Image.fromarray(np.uint8(waveplus)).convert('RGB')

        if self.transform is not None:
            img = self.transform(waveplus)

        return img, label

    def __len__(self):
        return len(self.imgs)


class LinearClassifierFeatureFusion(nn.Module):
    """Linear classifier"""
    def __init__(self, num_classes=2):
        super(LinearClassifierFeatureFusion, self).__init__()
        feat_dim = 3328
        self.fc1 = nn.Linear(feat_dim, 256)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, features):
        features = features.view(features.shape[0], -1)
        x = self.fc1(features)
        x = self.dropout(x)
        x = self.fc2(x)
        return x
