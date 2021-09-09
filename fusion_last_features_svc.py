import argparse
import torch
import torch.nn as nn
from torchvision import transforms, datasets
from util import MyDataset
from networks.effnet_big import SupConEffNet, LinearClassifier
from networks.xception import Xception
import torch.backends.cudnn as cudnn
import sys
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import numpy as np
import joblib


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
    # parser.add_argument('--class_number', type=int, default=4,
    #                     help='class_number')
    parser.add_argument('--train_list', type=str,
                        default='/home/ubuntu/xuyi/SupContrast/data_lists/test1.txt',
                        help='path to custom dataset')
    parser.add_argument('--test_list', type=str,
                        default='/home/ubuntu/xuyi/SupContrast/data_lists/test.txt',
                        help='path to custom dataset')
    parser.add_argument('--ckpt_supcon', type=str, default='',
                        help='path to Sup-Con pre-trained model')
    parser.add_argument('--ckpt_xc', type=str, default='',
                        help='path to Xception pre-trained model')
    parser.add_argument('--model_name', type=str, default='',
                        help='name of saved model')

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
    supcon_train_dataset = MyDataset(txt_path=opt.train_list,
                                    transform=supcon_transform)

    supcon_train_loader = torch.utils.data.DataLoader(
        supcon_train_dataset, batch_size=opt.batch_size, shuffle=True,
        num_workers=8, pin_memory=True)

    supcon_test_dataset = MyDataset(txt_path=opt.test_list,
                                transform=supcon_transform)

    supcon_test_loader = torch.utils.data.DataLoader(
        supcon_test_dataset, batch_size=opt.batch_size, shuffle=False,
        num_workers=8, pin_memory=True)

    return supcon_train_loader, supcon_test_loader


def set_model_supcon(opt):
    model = SupConEffNet()

    encoder = model.encoder
    encoder_dict = encoder.state_dict()

    ckpt = torch.load(opt.ckpt_supcon, map_location='cpu')
    state_dict = ckpt['model']

    state_dict = {k: v for k, v in state_dict.items() if k in encoder_dict}
    encoder_dict.update(state_dict)

    if torch.cuda.is_available():
        if torch.cuda.device_count() > 1:
            model.encoder = torch.nn.DataParallel(model.encoder)
        else:
            new_state_dict = {}
            for k, v in state_dict.items():
                k = k.replace("0.module.", "")
                new_state_dict[k] = v
            state_dict = new_state_dict

        encoder = encoder.cuda()
        cudnn.benchmark = True

    encoder.load_state_dict(encoder_dict)

    # return model
    return encoder


def supcon_feature(supcon_train_loader, supcon_test_loader, encoder):
    """validation"""
    encoder.eval()
    i = 0
    j = 0

    with torch.no_grad():
        for idx, (images, labels) in enumerate(supcon_train_loader):
            images = images.float().cuda()
            labels = labels.cuda()
            # bsz = labels.shape[0]

            # forward
            output = encoder(images)

            if i == 0:
                all_supcon_features_train = output
                all_labels_train = labels
                i = 1
            else:
                all_supcon_features_train = torch.cat((all_supcon_features_train, output), dim=0)
                all_labels_train = torch.cat((all_labels_train, labels), dim=0)

        for idx, (images, labels) in enumerate(supcon_test_loader):
            images = images.float().cuda()
            labels = labels.cuda()
            # bsz = labels.shape[0]

            # forward
            output = encoder(images)

            if j == 0:
                all_supcon_features_test = output
                all_labels_test = labels
                j = 1
            else:
                all_supcon_features_test = torch.cat((all_supcon_features_test, output), dim=0)
                all_labels_test = torch.cat((all_labels_test, labels), dim=0)

    return all_supcon_features_train, all_labels_train, all_supcon_features_test, all_labels_test


def xception_feature(opt):
    i = 0
    j = 0
    xc_train_dataset = MyDataset(txt_path=opt.train_list,
                                transform=xception_default_data_transforms['train'])
    xc_train_loader = torch.utils.data.DataLoader(xc_train_dataset, batch_size=opt.batch_size,
                                                 shuffle=True, drop_last=False, num_workers=8)
    xc_test_dataset = MyDataset(txt_path=opt.test_list,
                             transform=xception_default_data_transforms['val'])
    xc_test_loader = torch.utils.data.DataLoader(xc_test_dataset, batch_size=opt.batch_size,
                                              shuffle=False, drop_last=False, num_workers=8)
    xc_model = Xception(num_classes=2)

    xc_model_dict = xc_model.state_dict()
    # for k, v in xc_model_dict.items():
    #     print(k)
    state_dict = torch.load(opt.ckpt_xc)
    new_state_dict = {}
    for k, v in state_dict.items():
        # print(k)
        k = k.replace("model.", "")
        k = k.replace('last_linear.1', 'fc')
        new_state_dict[k] = v

    state_dict = {k: v for k, v in new_state_dict.items() if k in xc_model_dict}

    xc_model.load_state_dict(state_dict)
    xc_model = xc_model.cuda()
    xc_model.eval()
    with torch.no_grad():
        for (image, labels) in xc_train_loader:
            image = image.cuda()
            # labels = labels.cuda()
            xc_features_train = xc_model.logits_ff(image)

            if i == 0:
                all_xc_features_train = xc_features_train
                i = 1
            else:
                all_xc_features_train = torch.cat((all_xc_features_train, xc_features_train), dim=0)

        for (image, labels) in xc_test_loader:
            image = image.cuda()
            # labels = labels.cuda()
            xc_features_test = xc_model.logits_ff(image)

            if j == 0:
                all_xc_features_test = xc_features_test
                j = 1
            else:
                all_xc_features_test = torch.cat((all_xc_features_test, xc_features_test), dim=0)

    return all_xc_features_train, all_xc_features_test


def main():
    opt = parse_option()

    # -------------------------- Supervised Constractive -------------------------
    # build data loader
    supcon_train_loader, supcon_test_loader = set_loader_supcon(opt)

    # build model
    encoder = set_model_supcon(opt)

    features_supcon_train, labels_train, features_supcon_test, labels_test = supcon_feature(supcon_train_loader, supcon_test_loader, encoder)
    features_supcon_train = features_supcon_train.view(features_supcon_train.size(0), -1)
    features_supcon_test = features_supcon_test.view(features_supcon_test.size(0), -1)

    # -------------------------- Xception -------------------------
    features_xc_train, features_xc_test = xception_feature(opt)

    # normzlize_feature = transforms.Normalize(0, 0.5)
    # features_supcon = normzlize_feature(features_supcon)
    # features_xc = normzlize_feature(features_xc)

    # features_fusion = torch.cat((features_supcon, features_xc), dim=1)
    # print(features_fusion.shape)
    # features_fusion = features_fusion.cpu()

    features_supcon_train = features_supcon_train.cpu()
    features_xc_train = features_xc_train.cpu()
    labels_train = labels_train.cpu()

    features_supcon_test = features_supcon_test.cpu()
    features_xc_test = features_xc_test.cpu()
    labels_test = labels_test.cpu()

    scaler = StandardScaler()

    features_supcon_train = scaler.fit_transform(features_supcon_train.numpy())
    features_xc_train = scaler.fit_transform(features_xc_train.numpy())
    features_fusion_train = np.hstack((features_supcon_train, features_xc_train))
    X_train = features_fusion_train
    y_train = labels_train.numpy()

    features_supcon_test = scaler.fit_transform(features_supcon_test.numpy())
    features_xc_test = scaler.fit_transform(features_xc_test.numpy())
    features_fusion_test = np.hstack((features_supcon_test, features_xc_test))
    X_test = features_fusion_test
    y_test = labels_test.numpy()

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)

    # training phase: SVM , fit model to training data ------------------------------
    clf = SVC(kernel='rbf', C=10, gamma=0.00001, random_state=0)
    clf.fit(X_train, y_train)
    # # training score
    train_score = clf.score(X_train, y_train)
    print('train_score: {}'.format(train_score))

    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    print('test accuracy:{}'.format(acc))
    print('tn, fp, fn, tp:{}, {}, {}, {}'.format(tn, fp, fn, tp))

    joblib_file = 'save/SVC/' + opt.model_name + '.pkl'
    joblib.dump(clf, joblib_file)


if __name__ == '__main__':
    main()
