from __future__ import print_function

import sys
import argparse
import time
import math

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torchvision import transforms, datasets

from util import AverageMeter, MyDataset
from util import adjust_learning_rate, warmup_learning_rate, accuracy, accuracy_evaluate
from util import set_optimizer
from networks.effnet_big import SupConEffNet, LinearClassifier

try:
    import apex
    from apex import amp, optimizers
except ImportError:
    pass

# python main_linear_efficientnet.py --print_freq 1 --batch_size 64 --dataset deepfakes --model efficientnetb0 --train_list data_lists/four_datasets_train_000599_frame30.txt --val_list data_lists/four_datasets_val_600799_frame30.txt --ckpt save/SupCon/deepfakes_models/2021-04-29\ 16\:24\:39.712047_

def parse_option():
    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--print_freq', type=int, default=10,
                        help='print frequency')
    parser.add_argument('--save_freq', type=int, default=50,
                        help='save frequency')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='batch_size')
    parser.add_argument('--num_workers', type=int, default=16,
                        help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=100,
                        help='number of training epochs')
    parser.add_argument('--train_list', type=str,
                        default='/home/ubuntu/xuyi/Deepfake-Detection/data_list/test.txt',
                        help='path to custom dataset')
    parser.add_argument('--val_list', type=str,
                        default='/home/ubuntu/xuyi/Deepfake-Detection/data_list/test.txt',
                        help='path to custom dataset')
    parser.add_argument('--classes', action='store_true', help='True: class, False: fakeness', default='False')

    # optimization
    parser.add_argument('--learning_rate', type=float, default=0.1,
                        help='learning rate')
    parser.add_argument('--lr_decay_epochs', type=str, default='60,75,90',
                        help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.2,
                        help='decay rate for learning rate')
    parser.add_argument('--weight_decay', type=float, default=0,
                        help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='momentum')

    # model dataset
    parser.add_argument('--model', type=str, default='resnet50')
    parser.add_argument('--dataset', type=str, default='cifar10', help='dataset')

    # other setting
    parser.add_argument('--cosine', action='store_true',
                        help='using cosine annealing')
    parser.add_argument('--warm', action='store_true',
                        help='warm-up for large batch training')

    parser.add_argument('--ckpt', type=str, default='',
                        help='path to pre-trained model')

    opt = parser.parse_args()

    # set the path according to the environment
    opt.data_folder = './datasets/'

    iterations = opt.lr_decay_epochs.split(',')
    opt.lr_decay_epochs = list([])
    for it in iterations:
        opt.lr_decay_epochs.append(int(it))

    opt.model_name = '{}_{}_lr_{}_decay_{}_bsz_{}'.\
        format(opt.dataset, opt.model, opt.learning_rate, opt.weight_decay,
               opt.batch_size)

    if opt.cosine:
        opt.model_name = '{}_cosine'.format(opt.model_name)

    # warm-up for large-batch training,
    if opt.warm:
        opt.model_name = '{}_warm'.format(opt.model_name)
        opt.warmup_from = 0.01
        opt.warm_epochs = 10
        if opt.cosine:
            eta_min = opt.learning_rate * (opt.lr_decay_rate ** 3)
            opt.warmup_to = eta_min + (opt.learning_rate - eta_min) * (
                    1 + math.cos(math.pi * opt.warm_epochs / opt.epochs)) / 2
        else:
            opt.warmup_to = opt.learning_rate

    if opt.dataset == 'cifar10':
        opt.n_cls = 10
    elif opt.dataset == 'cifar100':
        opt.n_cls = 100
    elif opt.dataset == 'deepfakes':
        opt.n_cls = 4
    else:
        raise ValueError('dataset not supported: {}'.format(opt.dataset))

    return opt


def set_loader(opt):
    # construct data loader
    if opt.dataset == 'cifar10':
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2023, 0.1994, 0.2010)
    elif opt.dataset == 'cifar100':
        mean = (0.5071, 0.4867, 0.4408)
        std = (0.2675, 0.2565, 0.2761)
    elif opt.dataset == 'deepfakes':
        mean = (0.5, 0.5, 0.5)
        std = (0.5, 0.5, 0.5)
    else:
        raise ValueError('dataset not supported: {}'.format(opt.dataset))
    normalize = transforms.Normalize(mean=mean, std=std)

    val_transform = transforms.Compose([
        transforms.Resize([224, 224]),
        transforms.ToTensor(),
        normalize,
    ])

    if opt.dataset == 'cifar10':
        val_dataset = datasets.CIFAR10(root=opt.data_folder,
                                       train=False,
                                       transform=val_transform)
    elif opt.dataset == 'cifar100':
        val_dataset = datasets.CIFAR100(root=opt.data_folder,
                                        train=False,
                                        transform=val_transform)
    elif opt.dataset == 'deepfakes':
        val_dataset = MyDataset(txt_path=opt.val_list,
                                transform=val_transform)
    else:
        raise ValueError(opt.dataset)

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=512, shuffle=False,
        num_workers=8, pin_memory=True)

    return val_loader

def set_model(opt):
    model = SupConEffNet()
    # criterion = torch.nn.CrossEntropyLoss()

    classifier = LinearClassifier(num_classes=opt.n_cls)

    whole_model = nn.Sequential(model.encoder, classifier)

    ckpt = torch.load(opt.ckpt, map_location='cpu')
    state_dict = ckpt['model']

    if torch.cuda.is_available():
        if torch.cuda.device_count() > 2:
            model.encoder = torch.nn.DataParallel(model.encoder)
        else:
            new_state_dict = {}
            for k, v in state_dict.items():
                k = k.replace("module.", "")
                new_state_dict[k] = v
            state_dict = new_state_dict
        model = model.cuda()
        classifier = classifier.cuda()
        # criterion = criterion.cuda()
        cudnn.benchmark = True

        # model.load_state_dict(state_dict)
        # classifier.load_state_dict(state_dict)
        whole_model.load_state_dict(state_dict)

    # return model, classifier
    return whole_model


# def validate(val_loader, model, classifier, opt):
def validate(val_loader, whole_model, opt):
    """validation"""
    # model.eval()
    # classifier.eval()
    whole_model.eval()

    batch_time = AverageMeter()
    top1 = AverageMeter()
    tn_all = 0
    fp_all = 0
    fn_all = 0
    tp_all = 0

    with torch.no_grad():
        end = time.time()

        for idx, (images, labels) in enumerate(val_loader):
            images = images.float().cuda()
            labels = labels.cuda()
            bsz = labels.shape[0]

            # forward
            # output = classifier(model.encoder(images))
            output = whole_model(images)
            # c1 = output[:, 0].view(-1, 1)
            # c2 = torch.max(output[:, 1:], dim=1)
            # c2 = c2.values.view(-1, 1)
            # score = torch.cat((c1, c2), dim=-1)
            # print(score)

            # update metric
            if opt.classes == True:
                acc1, acc2 = accuracy(output, labels, topk=(1, 1))
                top1.update(acc1[0], bsz)
            elif opt.classes == 'False':
                acc1, acc2 = accuracy_evaluate(output, labels, topk=(1, 1))
                top1.update(acc1, bsz)
                # _, tn, fp, fn, tp = accuracy_evaluate(output, labels, topk=(1, 1))
                # tn_all += tn
                # fp_all += fp
                # fn_all += fn
                # tp_all += tp

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if idx % opt.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Acc@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                       idx, len(val_loader), batch_time=batch_time, top1=top1))

    print(' * Acc@1 {top1.avg:.3f}'.format(top1=top1))


def main():
    best_acc = 0
    opt = parse_option()

    # build data loader
    val_loader = set_loader(opt)

    # build model
    # model, classifier = set_model(opt)
    whole_model = set_model(opt)


    # validate(val_loader, model, classifier, opt)
    validate(val_loader, whole_model, opt)


if __name__ == '__main__':
    main()
