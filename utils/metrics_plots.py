import numpy as np
import h5py
import os
import sys
sys.path.append('..')
import copy
import torch
import matplotlib.ticker as mtick
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, accuracy_score, precision_recall_fscore_support, confusion_matrix 
from utils.funcs import round_decimal

def calculate_acc_prec_recall_f1(labels_lst, predicted_lst, average='macro', ignore_labels=None):
    if ignore_labels is None:
        ignore_labels = []

    # Create a mask to ignore specified labels
    mask = ~np.isin(labels_lst, ignore_labels)

    # Filter labels and predictions
    filtered_labels = np.array(labels_lst)[mask]
    filtered_predictions = np.array(predicted_lst)[mask]

    # Calculate metrics on the filtered data
    acc = accuracy_score(y_true=filtered_labels, y_pred=filtered_predictions)
    mac_precision, mac_recall, mac_f1, _ = precision_recall_fscore_support(
        y_true=filtered_labels, 
        y_pred=filtered_predictions, 
        beta=1.0, 
        average=average, 
        zero_division=np.nan
    )
    
    return acc, mac_precision, mac_recall, mac_f1

def classify_report(labels_lst, predicted_lst, label_names, logger, out_path, metric_name, state, h5_name, obtain_conf_mat, save_h5=True, connectome=False):
    """Generate classification performance report"""
    # classification report
    if not connectome:
        cls_report = classification_report(y_true=labels_lst, y_pred=predicted_lst, digits=5, target_names=label_names)
        logger.info('=' * 55)
        logger.info('Best {} classification report:\n{}'.format(metric_name, cls_report))
        logger.info('=' * 55)
        logger.info('\n')
    
    if obtain_conf_mat:
        # confusion matrix: true (rows), predicted (columns) # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html 
        conf_matrix = confusion_matrix(y_true=labels_lst, y_pred=predicted_lst, normalize='true')
        # Plot the confusion matrix
        fig, ax = plt.subplots(figsize=(25, 25))
        sns.heatmap(conf_matrix, annot=True, fmt='.2f', cmap="Blues", cbar=False, ax=ax)
        ax.xaxis.set_ticklabels(label_names) 
        ax.yaxis.set_ticklabels(label_names)
        plt.xlabel('Predicted label')   # columns
        plt.ylabel('True label')        # rows
        plt.savefig(os.path.join(out_path, '{}.png'.format(h5_name.split('.')[0].replace('results', 'ConfMat'))))
    
    if save_h5:
        # save classification report
        eval_res = h5py.File(os.path.join(out_path, h5_name), "w")
        eval_res['{}_predictions'.format(state)] = predicted_lst
        eval_res['{}_labels'.format(state)] = labels_lst
        eval_res['label_names'] = label_names
        try:
            eval_res['classification_report'] = cls_report
        except:
            pass
        
    if connectome:
        acc = accuracy_score(y_true=labels_lst, y_pred=predicted_lst)
        mac_precision, mac_recall, mac_f1, _ = precision_recall_fscore_support(y_true=labels_lst, y_pred=predicted_lst, beta=1.0, average='macro', zero_division=np.nan)
        weighted_precision, weighted_recall, weighted_f1, _ = precision_recall_fscore_support(y_true=labels_lst, y_pred=predicted_lst, beta=1.0, average='weighted', zero_division=np.nan) # ignore empty labels
        
        logger.info("="*55)
        logger.info("Results for model best f1 weights")
        logger.info("Accuracy:           {}".format(round_decimal(acc,5)))
        logger.info("Macro scores    F1: {} Precision: {} Recall: {}".format(round_decimal(mac_f1,5), round_decimal(mac_precision,5), round_decimal(mac_recall,5)))
        logger.info("Weighted scores F1: {} Precision: {} Recall: {}".format(round_decimal(weighted_f1,5), round_decimal(weighted_precision,5), round_decimal(weighted_recall,5)))
        logger.info("="*55)

def process_curves(epoch, train_loss_lst, val_loss_lst, train_acc_lst, val_acc_lst,
                   train_precision_lst, val_precision_lst, train_recall_lst, val_recall_lst,
                   train_f1_lst, val_f1_lst, best_acc, best_acc_epoch, best_f1_mac, best_f1_epoch, out_path, atlas):
    """Generate training curves based on atlas input."""
    epoch_lst = range(1, epoch + 1)
    fig, axes = plt.subplots(2, 3, figsize=(30, 20))

    # Plot based on the number of atlas components
    def plot_curve(ax, y_data_lst, labels, title, ylabel):
        for i in range(len(atlas)):
            ax.plot(epoch_lst, [y[i] for y in y_data_lst[0]], '-', label=f'{labels[0]} ({atlas[i]})')
            ax.plot(epoch_lst, [y[i] for y in y_data_lst[1]], '-', label=f'{labels[1]} ({atlas[i]})')
        ax.set_title(title, fontsize=15)
        ax.set_xlabel('epochs', fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.grid()
        ax.legend()
        ax.set_xticks(np.arange(1, len(epoch_lst) + 1, 5.0))
    # loss
    axes[0, 0].plot(epoch_lst, train_loss_lst, '-', label=f'train loss')
    axes[0, 0].plot(epoch_lst, val_loss_lst, '-', label=f'val loss')
    axes[0, 0].set_title('Loss Curve', fontsize=15)
    axes[0, 0].set_xlabel('epochs', fontsize=12)
    axes[0, 0].set_ylabel('loss', fontsize=12)
    axes[0, 0].grid()
    axes[0, 0].legend()
    axes[0, 0].set_xticks(np.arange(1, len(epoch_lst) + 1, 5.0))

    # accuracy
    plot_curve(axes[0, 1], [train_acc_lst, val_acc_lst], ['train accuracy', 'val accuracy'], 'Accuracy Curve', 'accuracy')

    # f1
    plot_curve(axes[0, 2], [train_f1_lst, val_f1_lst], ['train f1', 'val f1'], 'f1 (macro) Curve', 'f1')

    # precision
    plot_curve(axes[1, 0], [train_precision_lst, val_precision_lst], ['train precision', 'val precision'], 'Precision (macro) Curve', 'precision')

    # recall
    plot_curve(axes[1, 1], [train_recall_lst, val_recall_lst], ['train recall', 'val recall'], 'Recall (macro) Curve', 'recall')

    # accuracy, macro precision, macro recall, macro f1
    for i in range(len(atlas)):
        axes[1, 2].plot(epoch_lst, [y[i] for y in val_precision_lst], '-', color='tab:green', label='precision (macro)')
        axes[1, 2].plot(epoch_lst, [y[i] for y in val_recall_lst], '-', color='tab:blue', label='recall (macro)')
        axes[1, 2].plot(epoch_lst, [y[i] for y in val_f1_lst], '-', color='tab:olive', label='f1 (macro)')
        axes[1, 2].plot(epoch_lst, [y[i] for y in val_acc_lst], '-', color='tab:red', label='accuracy')
    axes[1, 2].scatter(best_f1_epoch, best_f1_mac, c='tab:olive', marker='P', label='best f1 (macro)')
    axes[1, 2].set_title('Metric Comparison Curve', fontsize=15)
    axes[1, 2].set_xlabel('epochs', fontsize=12)
    axes[1, 2].set_ylabel('metrics', fontsize=12)
    axes[1, 2].grid()
    axes[1, 2].legend()
    axes[1, 2].set_xticks(np.arange(1, len(epoch_lst) + 1, 5.0))

    plt.savefig(os.path.join(out_path, 'train_validation_process_analysis.png'))

def best_swap(metric, epoch, net, labels_lst, predicted_lst):
    best_metric = metric
    best_epoch = epoch
    best_wts = copy.deepcopy(net.state_dict())
    best_labels_lst = labels_lst
    best_pred_lst = predicted_lst
    return best_metric, best_epoch, best_wts, best_labels_lst, best_pred_lst

def save_best_weights(net, best_wts, out_path, metric_name, epoch, metric_value, logger=None):
    net.load_state_dict(best_wts)
    torch.save(net.state_dict(), '{}/best_{}_model.pth'.format(out_path, metric_name))
    if logger:
        logger.info('The model with best {} is saved: epoch {}, {} {}'.format(metric_name, epoch, metric_name, metric_value))  

def create_connectome(decoded_labels, num_labels=85):
    """Create a connectome matrix from decoded labels."""
    connectome_matrix = np.zeros((num_labels, num_labels), dtype=int)
    
    for a, b in decoded_labels:
        connectome_matrix[a, b] += 1
        connectome_matrix[b, a] += 1  # Ensure symmetry
    
    return connectome_matrix
