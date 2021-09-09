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
import cv2 as cv
import matplotlib.pyplot as plt


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
    return whole_model, model.encoder


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
    whole_model, model = set_model_supcon(opt)

    # -------------------------- Xception -------------------------
    # score_xc = xception_score(opt)

    model_weights = []  # we will save the conv layer weights in this list
    conv_layers = []  # we will save the 49 conv layers in this list
    # get all the model children as list
    model_children = list(model.children())
    print(len(model_children))

    # # counter to keep count of the conv layers
    # counter = 0
    # # append all the conv layers and their respective weights to the list
    # for i in range(len(model_children)):
    #     # if type(model_children[i]) == nn.Conv2d:
    #     counter += 1
    #     # model_weights.append(model_children[i].weight)
    #     conv_layers.append(model_children[i])
    #     # elif type(model_children[i]) == nn.Sequential:
    #     #     for j in range(len(model_children[i])):
    #     #         for child in model_children[i][j].children():
    #     #             if type(child) == nn.Conv2d:
    #     #                 counter += 1
    #     #                 model_weights.append(child.weight)
    #     #                 conv_layers.append(child)
    # print(f"Total convolutional layers: {counter}")
    #
    # print(conv_layers)
    #
    # # take a look at the conv layers and the respective weights
    # # for weight, conv in zip(model_weights, conv_layers):
    # #     # print(f"WEIGHT: {weight} \nSHAPE: {weight.shape}")
    # #     print(f"CONV: {conv} ====> SHAPE: {weight.shape}")
    #
    # # read and visualize an image
    # img = cv.imread('/home/ubuntu/xuyi/SupContrast/datasets/FF++/original_sequences/youtube/c23/face_images/000/0000.png')
    # img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    #
    # mean = (0.5, 0.5, 0.5)
    # std = (0.5, 0.5, 0.5)
    #
    # normalize = transforms.Normalize(mean=mean, std=std)
    #
    # supcon_transform = transforms.Compose([
    #     transforms.ToPILImage(),
    #     transforms.Resize([224, 224]),
    #     transforms.ToTensor(),
    #     normalize,
    # ])
    #
    # img = np.array(img)
    # # apply the transforms
    # img = supcon_transform(img)
    # print(img.size())
    # # unsqueeze to add a batch dimension
    # img = img.unsqueeze(0)
    # print(img.size())
    # img = img.cuda()
    #
    # # pass the image through all the layers
    # results = [conv_layers[0](img)]
    # for i in range(1, len(conv_layers)):
    #     # pass the result from the last layer to the next layer
    #     results.append(conv_layers[i](results[-1]))
    # # make a copy of the `results`
    # outputs = results
    #
    # # visualize 64 features from each layer
    # # (although there are more feature maps in the upper layers)
    # for num_layer in range(len(outputs)):
    #     plt.figure(figsize=(30, 30))
    #     layer_viz = outputs[num_layer][0, :, :, :]
    #     layer_viz = layer_viz.data
    #     print(layer_viz.size())
    #     for i, filter_no in enumerate(layer_viz):
    #         if i == 64:  # we will visualize only 8x8 blocks from each layer
    #             break
    #         plt.subplot(8, 8, i + 1)
    #         # plt.imshow(filter, cmap='gray')
    #         plt.axis("off")
    #     print('Saving layer {} feature maps...'.format(num_layer))
    #     plt.savefig('plots/visulization/layer_{}.png'.format(num_layer))
    #     # plt.show()
    #     # plt.close()



if __name__ == '__main__':
    main()
