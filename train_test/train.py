import argparse
import os
import sys
sys.path.append('../')
sys.path.append('../../')
import time
import h5py
import numpy as np

import torch
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torch.nn.functional as F
from torch.nn import NLLLoss

from datasets.dataset import unrelatedHCP_PatchData
from models.pointnet import PointNetCls
from models.dgcnn import tract_DGCNN_cls
from utils.logger import create_logger
from utils.metrics_plots import classify_report, process_curves, calculate_acc_prec_recall_f1, best_swap, save_best_weights
from utils.funcs import round_decimal, unify_path, makepath, fix_seed, obtain_TractClusterMapping, cluster2tract_label, save_info_feat, str2num
from utils.cli import create_parser, save_args, adaptive_args
from utils.metrics_connectome import *

def load_datasets(eval_split, args, test=False, logger=None):
    """load train and validation data"""
    # load feature and label data
    if not test:
        train_dataset = unrelatedHCP_PatchData(
                root=args.input_path,
                out_path=args.out_path,
                logger=logger,
                split='train',
                num_fiber_per_brain=args.num_fiber_per_brain,
                num_point_per_fiber=args.num_point_per_fiber,
                use_tracts_training=args.use_tracts_training,
                k=args.k,
                k_global=args.k_global,
                rot_ang_lst=args.rot_ang_lst,
                scale_ratio_range=args.scale_ratio_range,
                trans_dis=args.trans_dis,
                aug_times=args.aug_times,
                cal_equiv_dist=args.cal_equiv_dist,
                k_ds_rate=args.k_ds_rate,
                recenter=args.recenter,
                include_org_data=args.include_org_data,
                atlas=args.atlas,
                threshold=args.threshold)
    else:
        train_dataset = None
        
    eval_dataset = unrelatedHCP_PatchData(
        root=args.input_path,
        out_path=args.out_path,
        logger=logger,
        split=eval_split,
        num_fiber_per_brain=args.num_fiber_per_brain,
        num_point_per_fiber=args.num_point_per_fiber,
        use_tracts_training=args.use_tracts_training,
        k=args.k,
        k_global=args.k_global,
        rot_ang_lst=args.rot_ang_lst,
        scale_ratio_range=args.scale_ratio_range,
        trans_dis=args.trans_dis,
        aug_times=args.aug_times,
        cal_equiv_dist=args.cal_equiv_dist,
        k_ds_rate=args.k_ds_rate,
        recenter=args.recenter,
        include_org_data=args.include_org_data,
        atlas=args.atlas,
        threshold=args.threshold)

    return train_dataset, eval_dataset


def load_batch_data():
    """load train and val batch data"""
    eval_state='val'
    train_dataset, val_dataset = load_datasets(eval_split=eval_state, args=args, test=False, logger=logger)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.train_batch_size,
        shuffle=True)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.val_batch_size,
        shuffle=True)

    train_data_size = len(train_dataset)
    val_data_size = len(val_dataset)
    try:
        train_feat_shape = train_dataset.fiber_feat.shape
        val_feat_shape = val_dataset.fiber_feat.shape
    except:
        train_feat_shape = train_dataset.org_feat.shape
        val_feat_shape = val_dataset.org_feat.shape
    logger.info('The training data feature size is: {}'.format(train_feat_shape))
    logger.info('The validation data feature size is: {}'.format(val_feat_shape))
    
    # load label names
    if args.use_tracts_training:
        label_names =  list(ordered_tract_cluster_mapping_dict.keys()) 
    else:
        assert train_dataset.label_names == val_dataset.label_names
        label_names = train_dataset.label_names
    # label_names_h5 = h5py.File(os.path.join(args.out_path, 'label_names.h5'), 'w')
    # label_names_h5['y_names'] = label_names

    # logger.info('The label names are: {}'.format(str(label_names)))
    num_classes=[]
    if len(args.atlas)==1:
        num_classes=len(np.unique(label_names))
        logger.info('The number of classes is for {} is: {}'.format(args.atlas[0], num_classes))
    elif len(args.atlas)==2: 
        num_classes = [len(np.unique(label_names[0])), len(np.unique(label_names[1]))]
        logger.info('The number of classes is for {} is: {}'.format(args.atlas[0], num_classes[0]))
        logger.info('The number of classes is for {} is: {}'.format(args.atlas[1], num_classes[1]))
    else:
        raise ValueError("Currently, only supports one or two atlases.")
        
    # global feature
    train_global_feat = train_dataset.global_feat
    val_global_feat = val_dataset.global_feat
    
    # Samples per class for class-balanced loss
    samples_per_class = train_dataset.samples_per_class
    
    return train_loader, val_loader, label_names, num_classes, train_data_size, val_data_size, eval_state, train_global_feat, val_global_feat, samples_per_class


def load_model(args, num_classes, device, test=False):
    # model setting 
    if args.model_name == 'dgcnn':
        if len(args.atlas)>1:
            DL_model = tract_DGCNN_cls(num_classes_0=num_classes[0], num_classes_1=num_classes[1], args=args, device=device)
        else:
            DL_model = tract_DGCNN_cls(num_classes_0=num_classes, args=args, device=device)
            
    elif args.model_name == 'pointnet':
        if len(args.atlas)>1:
            DL_model = PointNetCls(k=args.k, k_global=args.k_global, num_classes_0=num_classes[0], num_classes_1=num_classes[1], feature_transform=False, first_feature_transform=False)
        else:
            DL_model = PointNetCls(k=args.k, k_global=args.k_global, num_classes_0=num_classes, feature_transform=False, first_feature_transform=False)
    else:
        raise ValueError('Please input valid model name dgcnn | pointnet')
        
    # load weights when testing
    if test:
        weight = torch.load(args.weight_path)  
        DL_model.load_state_dict(weight)
            
    DL_model.to(device)

    return DL_model


def load_settings(DL_model):
    # optimizers
    if args.opt == 'Adam':
        optimizer = optim.Adam(DL_model.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=args.weight_decay)
    elif args.opt == 'SGD':
        optimizer = optim.SGD(DL_model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    else:
        raise ValueError('Please input valid optimizers Adam | SGD')
    # schedulers
    if args.scheduler == 'step':
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.decay_factor)
    elif args.scheduler == 'wucd':
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=args.T_0, T_mult=args.T_mult)
    else:
        raise ValueError('Please input valid schedulers step | wucd')
    
    return optimizer, scheduler


def train_val_test_forward(idx_data, data, net, state, total_loss, labels_lst_0, predicted_lst_0, labels_lst_1, predicted_lst_1, args, device, num_classes, epoch=-1, num_batch=-1,
                           train_global_feat=None, eval_global_feat=None, samples_per_class=None):
    if state == 'test_realdata':
        points, klocal_feat_set = data
    else:
        # points [B, N_point, 3], label [B,1](cls) or [B,N_point](seg), [B, n_point, 3, k], [B]
        points, label, klocal_feat_set, new_subidx = data 
        # labels
        if len(args.atlas)==1:
            label_0 = label[:,0]  # Single task (single atlas) [B,1] to [B]
        elif len(args.atlas)==2:
            label_0 = label[:,0]  # Task 0
            label_1 = label[:,1]  # Task 1
        else:
            raise ValueError("Currently, only supports one or two atlases.")
        
    if state == 'train':
        global_feat = torch.from_numpy(train_global_feat)
    elif state == 'val' or state == 'test' or state =='test_realdata':
        global_feat = torch.from_numpy(eval_global_feat)
        
    num_fiber = points.shape[0]
    num_point_per_fiber = points.shape[1]
  
    # points
    points = points.transpose(2, 1)  # points [B, 3, N_point]
    # local feat
    klocal_feat_set = klocal_feat_set.transpose(2,1)  # [B,3,N_point,k]
    # global feat
    if state == 'test_realdata': 
        kglobal_point_set = global_feat.repeat(num_fiber,1,1,1).transpose(2,1)  # [B,3,N_point,k_global]
        new_subidx=torch.zeros(num_fiber).long() 
    else:
        new_subidx = new_subidx[:,0]  # [B,1] to [B]
        kglobal_point_set = global_feat.transpose(2,1)  # [num_subject*num_aug,3,N_point,k_global]
        kglobal_point_set = kglobal_point_set[new_subidx, ...]  # [B,3,N_point,k_global].
    # concat knn and random feat to get info feat
    if args.k == 0 and args.k_global == 0:
        info_point_set = torch.Tensor([0])
    elif args.k == 0 and args.k_global > 0:
        info_point_set = kglobal_point_set
    elif args.k > 0 and args.k_global == 0:
        info_point_set = klocal_feat_set
    elif args.k > 0 and args.k_global > 0:            
        info_point_set = torch.cat((klocal_feat_set, kglobal_point_set), dim=3)  # [B,3,N_point,k+k_global]
    else:
        raise ValueError('Invalid k and k sparse values')
    if (args.k>0 or args.k_global>0) and (idx_data==0 and epoch==1):
        if state != 'test_realdata': 
            start_idx = 0
            end_idx = 100
            org_info_feat_save_folder = os.path.join(args.out_path,'org_info_feat_vtk',state)
            makepath(org_info_feat_save_folder)
            save_info_feat(points, info_point_set, new_subidx, start_idx, end_idx, args.aug_times, args.k, 
                        args.k_global, args.k_ds_rate, org_info_feat_save_folder)
    if state == 'test_realdata':
        points, info_point_set = points.to(device), info_point_set.to(device)
    else:
        points, info_point_set = points.to(device), info_point_set.to(device)
        label_0 = label_0.to(device)
        if len(args.atlas)==2:
             label_1 = label_1.to(device)
    
    if state == 'train':
        optimizer.zero_grad()
        net = net.train()
    else:
        net = net.eval() 
    # get desired results for pred -- [B,N_point,Cls] for seg, [B,Cls] for cls
    if args.model_name == 'dgcnn': 
        pred_0 = net(points, info_point_set, task_id=0)   
        if len(args.atlas)==2:
            pred_1 = net(points, info_point_set, task_id=1)   
    elif args.model_name == 'pointnet':
        pred_0,_,_=net(points, info_point_set, task_id=0)
        if len(args.atlas)==2:
            pred_1,_,_ = net(points, info_point_set, task_id=1)   
    else:
        raise ValueError('Please input valid model name dgcnn | pointnet')       
    
    if state != 'test_realdata':
        # Compute loss for both tasks
        lossfn_0 = torch.nn.NLLLoss()
        lossfn_1 = torch.nn.NLLLoss()
        
        loss_0 = lossfn_0(pred_0, label_0)
        if len(args.atlas)==2:
            loss_1 = lossfn_1(pred_1, label_1)
            loss = loss_0 + loss_1
        else:
            loss = loss_0
    
    if state == 'train':
        loss.backward()
        optimizer.step()
    
    _, pred_idx_0 = torch.max(pred_0, dim=1)
    if len(args.atlas)==2:
        _, pred_idx_1 = torch.max(pred_1, dim=1)
    
    if state != 'test_realdata':
        total_loss += loss.item()
    
    if len(args.atlas) == 1:
        labels_lst_0.extend(label_0.cpu().detach().numpy())
        predicted_lst_0.extend(pred_idx_0.cpu().detach().numpy())
        
        # Since there's no second atlas, return placeholders for the second task
        labels_lst_1 = [0] * len(labels_lst_0)
        predicted_lst_1 = [0] * len(predicted_lst_0)
        
        return total_loss, labels_lst_0, predicted_lst_0, labels_lst_1, predicted_lst_1

    elif len(args.atlas) == 2:
        predicted_lst_0.extend(pred_idx_0.cpu().detach().numpy())
        predicted_lst_1.extend(pred_idx_1.cpu().detach().numpy())
        if state != 'test_realdata':
            labels_lst_0.extend(label_0.cpu().detach().numpy())
            labels_lst_1.extend(label_1.cpu().detach().numpy())
        else:
            labels_lst_0, labels_lst_1=0, 0
        
        return total_loss, labels_lst_0, predicted_lst_0, labels_lst_1, predicted_lst_1
    
    
def train_val_DL_net(net):
    """Train and validation loop for the network."""
    time_start = time.time()
    train_num_batch = train_data_size / args.train_batch_size
    val_num_batch = val_data_size / args.val_batch_size
    
    # Save training and validation process data
    metrics = {
        'train': {'loss': [], 'acc': [], 'precision': [], 'recall': [], 'f1': []},
        'val': {'loss': [], 'acc': [], 'precision': [], 'recall': [], 'f1': []}
    }

    # Save weights with best metrics
    org_best_f1_mac, tract_best_f1_mac = 0, 0
    org_best_f1_epoch, tract_best_f1_epoch = 1, 1
    org_best_f1_wts, tract_best_f1_wts = None, None
    org_best_f1_val_labels_lst, tract_best_f1_val_labels_lst = [], []
    org_best_f1_val_pred_lst, tract_best_f1_val_pred_lst = [], []

    # Early stopping parameters
    best_val_loss = float('inf')
    patience = 20 #args.patience  # Number of epochs to wait before stopping
    patience_counter = 0

    for epoch in range(args.epoch):
        train_start_time = time.time()
        epoch += 1
        total_train_loss, total_val_loss = 0, 0
        train_labels_lst_0, train_predicted_lst_0 = [], []
        train_labels_lst_1, train_predicted_lst_1 = [], []
        val_labels_lst_0, val_predicted_lst_0 = [], []
        val_labels_lst_1, val_predicted_lst_1 = [], []
        
        # Training loop
        for i, data in enumerate(train_loader, start=0):
            total_train_loss, train_labels_lst_0, train_predicted_lst_0, train_labels_lst_1, train_predicted_lst_1 = \
                train_val_test_forward(i, data, net, 'train', total_train_loss, train_labels_lst_0, train_predicted_lst_0, 
                                       train_labels_lst_1, train_predicted_lst_1, args, device, num_classes, epoch, train_num_batch, 
                                       train_global_feat=train_global_feat, samples_per_class=samples_per_class)

        if args.scheduler == 'step':
            scheduler.step()
            
        # train metric calculation
        train_end_time = time.time()
        train_time = round(train_end_time-train_start_time, 2)
        metrics['train']['loss'], metrics['train']['acc'], metrics['train']['precision'], \
        metrics['train']['recall'], metrics['train']['f1'] = \
            meters(epoch, train_num_batch, total_train_loss, train_labels_lst_0, train_predicted_lst_0, 
                   train_labels_lst_1, train_predicted_lst_1, metrics['train']['loss'], metrics['train']['acc'], 
                   metrics['train']['precision'], metrics['train']['recall'], metrics['train']['f1'], train_time, 'train')

        with torch.no_grad():
            val_start_time = time.time()
            for j, data in enumerate(val_loader, start=0):
                total_val_loss, val_labels_lst_0, val_predicted_lst_0, val_labels_lst_1, val_predicted_lst_1 = \
                    train_val_test_forward(j, data, net, 'val', total_val_loss, val_labels_lst_0, val_predicted_lst_0, 
                                           val_labels_lst_1, val_predicted_lst_1, args, device, num_classes, epoch, 
                                           eval_global_feat=val_global_feat)

        # validation metric calculation
        val_end_time = time.time()
        val_time = round(val_end_time-val_start_time, 2)
        metrics['val']['loss'], metrics['val']['acc'], metrics['val']['precision'], \
        metrics['val']['recall'], metrics['val']['f1'] = \
            meters(epoch, val_num_batch, total_val_loss, val_labels_lst_0, val_predicted_lst_0, 
                   val_labels_lst_1, val_predicted_lst_1, metrics['val']['loss'], metrics['val']['acc'], 
                   metrics['val']['precision'], metrics['val']['recall'], metrics['val']['f1'], val_time, 'val')
            
        # Save weights at regular intervals
        # if epoch % args.save_step == 0:
        #     torch.save(net.state_dict(), f'{args.out_path}/epoch_{epoch}_model.pth')
        #     print(f'Save {args.out_path}/epoch_{epoch}_model.pth')
            
        # Track the best metrics and swap if necessary
        if metrics['val']['f1'][-1][0] > org_best_f1_mac:
            org_best_f1_mac, org_best_f1_epoch, org_best_f1_wts, org_best_f1_val_labels_lst, org_best_f1_val_pred_lst = \
                best_swap(metrics['val']['f1'][-1][0], epoch, net, [val_labels_lst_0, val_labels_lst_1], [val_predicted_lst_0, val_predicted_lst_1])
        
        # Early stopping based on validation loss
        if total_val_loss < best_val_loss-0.001:
            best_val_loss = total_val_loss
            patience_counter = 0  # Reset patience counter if improvement
            save_best_weights(net, org_best_f1_wts, args.out_path, 'f1', org_best_f1_epoch, org_best_f1_mac, logger=None)
            # logger.info(f"New best model saved at epoch {epoch} with validation loss {total_val_loss:.4f}")
        else:
            patience_counter += 1  # No improvement
            logger.info(f"No improvement in validation loss. Patience: {patience_counter}/{patience}")

        if patience_counter >= patience:
            logger.info(f"Early stopping at epoch {epoch} due to no improvement in validation loss for {patience} consecutive epochs.")
            break
        

    # Save best weights
    save_best_weights(net, org_best_f1_wts, args.out_path, 'f1', org_best_f1_epoch, org_best_f1_mac, logger)

    # Plot performance curves
    process_curves(len(metrics['train']['loss']), metrics['train']['loss'], metrics['val']['loss'], metrics['train']['acc'], metrics['val']['acc'],
                   metrics['train']['precision'], metrics['val']['precision'], metrics['train']['recall'], metrics['val']['recall'],
                   metrics['train']['f1'], metrics['val']['f1'], -1, -1, org_best_f1_mac, org_best_f1_epoch, args.out_path, args.atlas)

    # remove checkpoints
    # saved_steps = list(range(args.save_step, args.epoch+1, args.save_step))[:-1] # save the last epoch
    # for epoch in saved_steps:
    #     os.remove('{}/epoch_{}_model.pth'.format(args.out_path, epoch))
    #     print('Remove {}/epoch_{}_model.pth'.format(args.out_path, epoch))
    
    # total processing time
    time_end = time.time() 
    total_time = round(time_end-time_start, 2)
    logger.info('Total processing time is {}s'.format(total_time))


def meters(epoch, num_batch, total_loss, labels_lst_0, predicted_lst_0, labels_lst_1, predicted_lst_1, 
           loss_lst, acc_lst, precision_lst, recall_lst, f1_lst, run_time, state):
    """Calculate and log metrics for both label_0 and label_1."""
    
    avg_loss = total_loss / float(num_batch)
    loss_lst.append(avg_loss)
    
    # Calculate metrics for label_0
    org_acc_0, org_precision_0, org_recall_0, org_f1_0 = calculate_acc_prec_recall_f1(labels_lst_0, predicted_lst_0, 'macro')

    # Calculate metrics for label_1
    if all(label == 0 for label in labels_lst_1):
        org_acc_1, org_precision_1, org_recall_1, org_f1_1 = 0, 0, 0, 0
    else:
        org_acc_1, org_precision_1, org_recall_1, org_f1_1 = calculate_acc_prec_recall_f1(labels_lst_1, predicted_lst_1, 'macro')
    
    # Append separate results for label_0 and label_1
    acc_lst.append([org_acc_0, org_acc_1])
    precision_lst.append([org_precision_0, org_precision_1])
    recall_lst.append([org_recall_0, org_recall_1])
    f1_lst.append([org_f1_0, org_f1_1])
    
    if org_acc_1==0: 
        logger.info(f'epoch [{epoch}/{args.epoch}] time: {run_time:>7}s \t{state:>6} loss: {avg_loss:>6.4f} \t'
                    f'acc: {org_acc_0:>6.4f}, macro f1: {org_f1_0:>6.4f}, prec: {org_precision_0:>6.4f}, rec: {org_recall_0:>6.4f}')
    else:
        logger.info(f'epoch [{epoch}/{args.epoch}] time: {run_time:>7}s \t{state:>6} loss: {avg_loss:>6.4f} \t'
                    f'label 0 acc: {org_acc_0:>6.4f}, macro f1: {org_f1_0:>6.4f}, prec: {org_precision_0:>6.4f}, rec: {org_recall_0:>6.4f}, \t'
                    f'label 1 acc: {org_acc_1:>6.4f}, macro f1: {org_f1_1:>6.4f}, prec: {org_precision_1:>6.4f}, rec: {org_recall_1:>6.4f}')
        
    return loss_lst, acc_lst, precision_lst, recall_lst, f1_lst



def results_logging(args, logger, eval_state, label_names, org_labels_lst, org_predicted_lst, atlas):
    """log results for original (800 clusters + 800 outliers) labels and tract (42+1other) labels"""
    label_names_str = label_names
    # best metric
    h5_name = 'HCP_{}_results_best{}.h5'.format(eval_state, args.best_metric)

    classify_report(org_labels_lst, org_predicted_lst, label_names_str, logger, args.out_log_path, args.best_metric,eval_state, h5_name, obtain_conf_mat=False, connectome=True)
    CM = ConnectomeMetrics(org_labels_lst, org_predicted_lst, atlas=atlas, out_path=args.out_log_path)
    logger.info(CM.format_metrics())
            

def train_val_paths():
    """paths"""
    args.input_path = unify_path(args.input_path)
    args.out_path_base = unify_path(args.out_path_base)
    # Train and validation
    args.out_path = os.path.join(args.out_path_base)
    makepath(args.out_path)


if __name__ == '__main__':
    # GPU check
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # Variable Space
    parser = create_parser()
    args = parser.parse_args()
    # fix seed
    args.manualSeed = 0 
    print("Random Seed: ", args.manualSeed)
    fix_seed(args.manualSeed)
    args.atlas = args.atlas.split(',')
    # Adjustment to get training 
    # args.recenter=False 
    args.include_org_data=False
    # adaptively change the args
    args = adaptive_args(args)
    # convert str to num
    args.rot_ang_lst = str2num(args.rot_ang_lst)
    args.scale_ratio_range = str2num(args.scale_ratio_range)
    # save local+global feature
    args.save_knn_neighbors = False
    # paths
    train_val_paths()
    # Record the training process and values
    logger = create_logger(args.out_path)
    logger.info('=' * 55)
    logger.info(args)
    logger.info('=' * 55)
    if not args.save_args_only:
        # load data
        train_loader, val_loader, label_names, num_classes, train_data_size, val_data_size, eval_state, train_global_feat, val_global_feat, samples_per_class \
            = load_batch_data()
        print("data loaded")
        # model setting
        DL_model = load_model(args, num_classes, device)
        optimizer, scheduler = load_settings(DL_model)
        # train and eval net
        train_val_DL_net(DL_model)
    # save args
    args_path = os.path.join(args.out_path, 'cli_args.txt')
    save_args(args_path, args)
