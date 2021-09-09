import matplotlib.pyplot as plt
import json
from sklearn.metrics import roc_curve, auc
import tikzplotlib
import numpy as np

# train_sets_x = ['SupCon_df', 'SupCon_f2f', 'SupCon_fs', 'SupCon_nt']
# train_sets_c = ['cnn_rnn_df', 'cnn_rnn_f2f', 'cnn_rnn_fs', 'cnn_rnn_nt']
# train_sets_m4 = ['mesoinc4_df', 'mesoinc4_f2f', 'mesoinc4_fs', 'mesoinc4_nt']
# train_set_2dataset = ['xc_df_f2f_', 'xc_df_fs_', 'xc_df_nt_', 'xc_f2f_fs_', 'xc_f2f_nt_', 'xc_fs_nt_']
train_set_3dataset = ['score_df_f2f_fs', 'score_df_f2f_nt', 'score_f2f_fs_nt', 'score_df_fs_nt']
# train_set_3dataset = ['xc_f2f_fs_nt_']
# curve_name = 'ROC curve for Resnet-152+LSTM trained on Ori_NT_Train'
test_set = ['df', 'f2f', 'fs', 'nt']
auc1 = 0.1
auc2 = 0.2
auc3 = 0.3
auc4 = 0.4


def plot_curve(train_set, curve_name, y_df_labels, y_df_pred, y_f2f_labels, y_f2f_pred, y_fs_labels, y_fs_pred, y_nt_labels, y_nt_pred):
    fpr_df, tpr_df, th_df = roc_curve(y_df_labels, y_df_pred)
    auc_value_df = auc(fpr_df, tpr_df)
    label_df = 'Ori_DF_test,AUC:' + str(auc_value_df)

    fpr_f2f, tpr_f2f, th_f2f = roc_curve(y_f2f_labels, y_f2f_pred)
    auc_value_f2f = auc(fpr_f2f, tpr_f2f)
    label_f2f = 'Ori_F2F_test,AUC:' + str(auc_value_f2f)

    fpr_fs, tpr_fs, th_fs = roc_curve(y_fs_labels, y_fs_pred)
    auc_value_fs = auc(fpr_fs, tpr_fs)
    label_fs = 'Ori_FS_test,AUC:' + str(auc_value_fs)

    fpr_nt, tpr_nt, th_nt = roc_curve(y_nt_labels, y_nt_pred)
    auc_value_nt = auc(fpr_nt, tpr_nt)
    label_nt = 'Ori_NT_test,AUC:' + str(auc_value_nt)

    plt.plot([0, 1], [0, 1], 'm--')
    plt.plot(fpr_df, tpr_df, 'orange', label=label_df)
    plt.plot(fpr_f2f, tpr_f2f, 'blue', label=label_f2f)
    plt.plot(fpr_fs, tpr_fs, 'red', label=label_fs)
    plt.plot(fpr_nt, tpr_nt, 'green', label=label_nt)
    # plt.plot(fpr_df, tpr_df, 'orange', label='Ori_DF_Test AUC= %0.3f' % auc_value_df)
    # plt.plot(fpr_f2f, tpr_f2f, 'blue', label='Ori_F2F_Test AUC = %0.3f' % auc_value_f2f)
    # plt.plot(fpr_fs, tpr_fs, 'red', label='Ori_FS_Test AUC = %0.3f' % auc_value_fs)
    # plt.plot(fpr_nt, tpr_nt, 'green', label='Ori_NT_Test AUC = %0.3f' % auc_value_nt)
    plt.legend(loc=4, prop={'size': 13})
    plt.xlim([0, 1])
    plt.ylim([0, 1.05])
    plt.xlabel('False positive rate', fontsize=16)
    plt.ylabel('True positive rate', fontsize=16)
    plt.title(curve_name, fontsize=13)
    plt.savefig(train_set+'_roc_curve.pdf', bbox_inches='tight')
    tikzplotlib.save(train_set+'_roc_curve.txt')
    plt.clf()

def main():
    for train_set in train_set_3dataset:
        if train_set == 'score_df_f2f':
            curve_name = 'ROC curve for SupCon trained on Ori_DF_F2F_train'
        elif train_set == 'score_df_fs':
            curve_name = 'ROC curve for SupCon trained on Ori_DF_FS_train'
        elif train_set == 'score_df_nt':
            curve_name = 'ROC curve for SupCon trained on Ori_DF_NT_train'
        elif train_set == 'score_f2f_fs':
            curve_name = 'ROC curve for SupCon trained on Ori_F2F_FS_train'
        elif train_set == 'score_f2f_nt':
            curve_name = 'ROC curve for SupCon trained on Ori_F2F_NT_train'
        elif train_set == 'score_fs_nt':
            curve_name = 'ROC curve for SupCon trained on Ori_FS_NT_train'
        elif train_set == 'score_df_f2f_fs':
            curve_name = 'ROC curve for SupCon trained on Ori_DF_F2F_FS_train'
        elif train_set == 'score_df_f2f_nt':
            curve_name = 'ROC curve for SupCon trained on Ori_DF_F2F_NT_train'
        elif train_set == 'score_f2f_fs_nt':
            curve_name = 'ROC curve for SupCon trained on Ori_F2F_FS_NT_train'
        elif train_set == 'score_df_fs_nt':
            curve_name = 'ROC curve for SupCon trained on Ori_DF_FS_NT_train'
        else:
            curve_name = 'none'

        for name in test_set:
            if name == 'df':
                with open(train_set + '2' + name + '_co_supcon_0.0_labels.txt', 'r') as f:
                    y_df_labels = json.load(f)
                    y_df_labels = list(map(int, y_df_labels))
                with open(train_set + '2' + name + '_co_supcon_0.0_prediction.txt', 'r') as f:
                    y_df_pred = json.load(f)
                    y_df_pred = list(map(float, y_df_pred))

            elif name == 'f2f':
                with open(train_set + '2' + name + '_co_supcon_0.0_labels.txt', 'r') as f:
                    y_f2f_labels = json.load(f)
                    y_f2f_labels = list(map(int, y_f2f_labels))
                with open(train_set + '2' + name + '_co_supcon_0.0_prediction.txt', 'r') as f:
                    y_f2f_pred = json.load(f)
                    y_f2f_pred = list(map(float, y_f2f_pred))

            elif name == 'fs':
                with open(train_set + '2' + name + '_co_supcon_0.0_labels.txt', 'r') as f:
                    y_fs_labels = json.load(f)
                    y_fs_labels = list(map(int, y_fs_labels))
                with open(train_set + '2' + name + '_co_supcon_0.0_prediction.txt', 'r') as f:
                    y_fs_pred = json.load(f)
                    y_fs_pred = list(map(float, y_fs_pred))

            elif name == 'nt':
                with open(train_set + '2' + name + '_co_supcon_0.0_labels.txt', 'r') as f:
                    y_nt_labels = json.load(f)
                    y_nt_labels = list(map(int, y_nt_labels))
                with open(train_set + '2' + name + '_co_supcon_0.0_prediction.txt', 'r') as f:
                    y_nt_pred = json.load(f)
                    y_nt_pred = list(map(float, y_nt_pred))

        plot_curve(train_set, curve_name, y_df_labels, y_df_pred, y_f2f_labels, y_f2f_pred, y_fs_labels, y_fs_pred, y_nt_labels, y_nt_pred)


if __name__ == '__main__':
    main()