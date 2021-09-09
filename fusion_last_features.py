import argparse
import torch
import torch.nn as nn
from torchvision import transforms, datasets
from util import MyDataset, LinearClassifierFeatureFusion
from networks.effnet_big import SupConEffNet, LinearClassifier
from networks.xception import Xception
import torch.backends.cudnn as cudnn
import torch.optim as optim

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
    features_supcon_train = features_supcon_train.view(features_supcon_train.size(0), 1, 1, -1)
    features_supcon_test = features_supcon_test.view(features_supcon_test.size(0), 1, 1, -1)

    # -------------------------- Xception -------------------------
    features_xc_train, features_xc_test = xception_feature(opt)
    features_xc_train = features_xc_train.view(features_xc_train.size(0), 1, 1, -1)
    features_xc_test = features_xc_test.view(features_xc_test.size(0), 1, 1, -1)

    normzlize_feature = transforms.Normalize(0, 0.5)
    features_supcon_train = normzlize_feature(features_supcon_train)
    features_supcon_test = normzlize_feature(features_supcon_test)
    features_xc_train = normzlize_feature(features_xc_train)
    features_xc_test = normzlize_feature(features_xc_test)

    features_supcon_train = features_supcon_train.view(features_supcon_train.size(0), -1)
    features_supcon_test = features_supcon_test.view(features_supcon_test.size(0), -1)
    features_xc_train = features_xc_train.view(features_xc_train.size(0), -1)
    features_xc_test = features_xc_test.view(features_xc_test.size(0), -1)

    features_fusion_train = torch.cat((features_supcon_train, features_xc_train), dim=1)
    features_fusion_test = torch.cat((features_supcon_test, features_xc_test), dim=1)

    model_feature_classifier = torch.nn.Sequential(
        torch.nn.Linear(3328, 256),
        torch.nn.ReLU(),
        torch.nn.Linear(256, 2),
    )
    model_feature_classifier = model_feature_classifier.cuda()
    # optimizer = optim.SGD(final_classifier.parameters(), lr=0.001, momentum=0.9)
    loss_fn = torch.nn.CrossEntropyLoss()

    learning_rate = 1e-4
    for t in range(500):
        # Forward pass: compute predicted y by passing x to the model. Module objects
        # override the __call__ operator so you can call them like functions. When
        # doing so you pass a Tensor of input data to the Module and it produces
        # a Tensor of output data.
        y_pred = model_feature_classifier(features_supcon_train)

        # Compute and print loss. We pass Tensors containing the predicted and true
        # values of y, and the loss function returns a Tensor containing the
        # loss.
        loss = loss_fn(y_pred, labels_train)
        if t % 100 == 99:
            print(t, loss.item())

        # Zero the gradients before running the backward pass.
        model_feature_classifier.zero_grad()

        # Backward pass: compute gradient of the loss with respect to all the learnable
        # parameters of the model. Internally, the parameters of each Module are stored
        # in Tensors with requires_grad=True, so this call will compute gradients for
        # all learnable parameters in the model.
        loss.backward()

        # Update the weights using gradient descent. Each parameter is a Tensor, so
        # we can access its gradients like we did before.
        with torch.no_grad():
            for param in model_feature_classifier.parameters():
                param -= learning_rate * param.grad


if __name__ == '__main__':
    main()
