import argparse
import torch
import torch.nn as nn
from torchvision import transforms, datasets
from util import MyDataset
from networks.effnet_big import SupConEffNet, LinearClassifier
from networks.models_xc import model_selection
import torch.backends.cudnn as cudnn
import json
from sklearn.metrics import confusion_matrix, roc_curve, auc
import numpy as np


xception_default_data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ]),
    'val': transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3)
    ]),
    'test': transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3)
    ]),
}


def parse_option():
    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--batch_size', type=int, default=256,
                        help='batch_size')
    parser.add_argument('--class_number', type=int, default=4,
                        help='class_number')
    parser.add_argument('--test_list', type=str,
                        default='/home/ubuntu/xuyi/Deepfake-Detection/data_list/test.txt',
                        help='path to custom dataset')
    parser.add_argument('--ckpt_supcon', type=str, default='',
                        help='path to Sup-Con pre-trained model')
    parser.add_argument('--ckpt_xc', type=str, default='',
                        help='path to Xception pre-trained model')
    parser.add_argument('--save_name', type=str, default='',
                       help='path to Xception pre-trained model')

    opt = parser.parse_args()

    return opt


def set_loader_supcon(opt):
    # construct data loader
    mean = (0.5, 0.5, 0.5)
    std = (0.5, 0.5, 0.5)

    normalize = transforms.Normalize(mean=mean, std=std)

    supcon_transform = transforms.Compose([
        transforms.Resize([224, 224]),
        transforms.ToTensor(),
        normalize,
    ])

    supcon_test_dataset = MyDataset(txt_path=opt.test_list,
                                transform=supcon_transform)

    supcon_test_loader = torch.utils.data.DataLoader(
        supcon_test_dataset, batch_size=opt.batch_size, shuffle=False,
        num_workers=8, pin_memory=True)

    return supcon_test_loader


def set_model_supcon(opt):
    model = SupConEffNet()

    classifier = LinearClassifier(num_classes=opt.class_number)

    whole_model = nn.Sequential(model.encoder, classifier)

    ckpt = torch.load(opt.ckpt_supcon, map_location='cpu')
    state_dict = ckpt['model']

    if torch.cuda.is_available():
        if torch.cuda.device_count() > 1:
            model.encoder = torch.nn.DataParallel(model.encoder)
        else:
            new_state_dict = {}
            for k, v in state_dict.items():
                k = k.replace("module.", "")
                new_state_dict[k] = v
            state_dict = new_state_dict

        whole_model = whole_model.cuda()
        cudnn.benchmark = True

    whole_model.load_state_dict(state_dict)

    # return model
    return whole_model


def supcon_score(supcon_test_loader, whole_model):
    """validation"""
    whole_model.eval()
    i = 0

    with torch.no_grad():
        for idx, (images, labels) in enumerate(supcon_test_loader):
            images = images.float().cuda()
            labels = labels.cuda()
            bsz = labels.shape[0]

            # forward
            output = whole_model(images)
            c1 = output[:, 0].view(-1, 1)
            c2 = torch.max(output[:, 1:], dim=1)
            c2 = c2.values.view(-1, 1)
            cat = torch.cat((c1, c2), dim=-1)
            score = torch.softmax(cat, dim=1)
            # print(score)
            # print(labels)
            # labels = labels.view(-1, 1)
            # score_label = torch.cat((score, labels), dim=1)
            # print(score_label)
            if i == 0:
                all_supcon_score = score
                all_labels = labels
                i = 1
            else:
                all_supcon_score = torch.cat((all_supcon_score, score), dim=0)
                all_labels = torch.cat((all_labels, labels), dim=0)

    return all_supcon_score, all_labels


def xception_score(opt):
    i = 0
    xc_test_dataset = MyDataset(txt_path=opt.test_list,
                             transform=xception_default_data_transforms['test'])
    xc_test_loader = torch.utils.data.DataLoader(xc_test_dataset, batch_size=opt.batch_size,
                                              shuffle=False, drop_last=False, num_workers=8)
    xc_model = model_selection(modelname='xception', num_out_classes=2, dropout=0.5)
    xc_model.load_state_dict(torch.load(opt.ckpt_xc))
    xc_model = xc_model.cuda()
    xc_model.eval()
    with torch.no_grad():
        for (image, labels) in xc_test_loader:
            image = image.cuda()
            # labels = labels.cuda()
            outputs = xc_model(image)
            # print(outputs)
            score = torch.softmax(outputs, dim=1)
            if i == 0:
                all_xc_score = score
                i = 1
            else:
                all_xc_score = torch.cat((all_xc_score, score), dim=0)

    return all_xc_score


def fusion_acc(score_supcon, score_xc, co_supcon):
    supcon = score_supcon*co_supcon
    xc = score_xc*(1-co_supcon)
    score_fusion = supcon + xc

    return score_fusion

def accuracy(score_fusion, labels):
    _, preds = torch.max(score_fusion, dim=1)

    tn, fp, fn, tp = confusion_matrix(labels.cpu(), preds.cpu()).ravel()
    acc = (tn + tp) / (tn + fp + fn + tp)

    return acc


def roc_auc(score_fusion, labels, opt, coo):
    y_score = score_fusion[:, 1].cpu()
    y_true = labels.cpu()
    fpr, tpr, _ = roc_curve(y_true, y_score)
    auc_score = auc(fpr, tpr)
    with open('plots/' + opt.save_name + '_co_supcon_' + str(round(coo, 1)) + '_labels.txt', 'w') as f:
        json.dump([int(i) for i in y_true], f)
    with open('plots/' + opt.save_name + '_co_supcon_' + str(round(coo, 1)) + '_prediction.txt', 'w') as f:
        json.dump([str(i) for i in y_score.numpy()], f)

    return auc_score


def main():
    opt = parse_option()

    # -------------------------- Supervised Constractive -------------------------
    # build data loader
    supcon_test_loader = set_loader_supcon(opt)

    # build model
    whole_model = set_model_supcon(opt)

    score_supcon, labels = supcon_score(supcon_test_loader, whole_model)

    # -------------------------- Xception -------------------------
    score_xc = xception_score(opt)

    for coo in (np.linspace(0, 1, 11)):
        score_fusion = fusion_acc(score_supcon, score_xc, co_supcon=coo)

        acc = accuracy(score_fusion, labels)

        auc_score = roc_auc(score_fusion, labels, opt, coo)

        print('coefficient for supcon and xc: {}, {}'.format(coo, 1-coo))
        print('Accuracy:{}'.format(acc))
        print('AUC score:{}'.format(auc_score))

if __name__ == '__main__':
    main()
