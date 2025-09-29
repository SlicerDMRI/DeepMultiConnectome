import os
import sys
sys.path.append('../')
import time
import numpy as np

import torch
import torch.nn.parallel
import torch.utils.data

from utils.logger import create_logger
from utils.funcs import cluster2tract_label, unify_path, makepath, fix_seed, obtain_TractClusterMapping
from utils.cli import create_parser, load_args, adaptive_args
from train import load_datasets, load_model, results_logging, train_val_test_forward


def load_batch_data():
    """load test data"""
    eval_state = 'test'
    _, test_dataset = load_datasets(eval_split=eval_state, args=args, test=True, logger=logger)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.test_batch_size,
        shuffle=True)

    test_data_size = len(test_dataset)
    try:
        test_feat_shape = test_dataset.fiber_feat.shape
    except:
        test_feat_shape = test_dataset.org_feat.shape
    logger.info('The testing data feature size is:{}'.format(test_feat_shape))

    # load label names
    if args.use_tracts_training:
        label_names =  list(ordered_tract_cluster_mapping_dict.keys()) 
    else:
        label_names = test_dataset.label_names
        
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
    
    # test random feat
    test_global_feat = test_dataset.global_feat
    
    # Samples per class for class-balanced loss
    samples_per_class = test_dataset.samples_per_class
    
    return test_loader, label_names, num_classes, test_data_size, eval_state, test_global_feat, samples_per_class 


def test_DL_net(net):
    """test the network"""
    total_test_loss = 0
    test_labels_lst_0, test_predicted_lst_0 = [], []
    test_labels_lst_1, test_predicted_lst_1 = [], []
    inference_times = []  # List to store inference times for each sample

    # test
    with torch.no_grad():
        for j, data in enumerate(test_loader, start=0):
            start_time = time.time()  # Start time for the batch
            total_loss, labels_lst_0, predicted_lst_0, labels_lst_1, predicted_lst_1 = \
                train_val_test_forward(j, data, net, 'test', total_test_loss, test_labels_lst_0, test_predicted_lst_0,
                                       test_labels_lst_1, test_predicted_lst_1, args, device, num_classes, epoch=1, eval_global_feat=test_global_feat)
            end_time = time.time()  # End time for the batch

            batch_inference_time = end_time - start_time
            num_samples = data[0].size(0)  # Assuming data[0] contains the inputs
            per_sample_time = batch_inference_time / num_samples

            inference_times.extend([per_sample_time] * num_samples)  # Add per-sample times to the list

    # Calculate average and standard deviation of inference times per sample
    avg_time = np.mean(inference_times)
    std_time = np.std(inference_times)

    logger.info(f"Average inference time per sample: {avg_time:.6f} seconds")
    logger.info(f"Standard deviation of inference time per sample: {std_time:.6f} seconds")

    return labels_lst_0, predicted_lst_0, labels_lst_1, predicted_lst_1

    
def test_paths():
    # paths
    args.input_path = unify_path(args.input_path)
    args.out_path_base = unify_path(args.out_path_base)
    args.out_path = os.path.join(args.out_path_base)
    # test
    if args.aug_times >0:
        out_log_path_base = os.path.join(args.out_path, 'log_AugTimes{}'.format(args.aug_times))
    else:
        out_log_path_base = os.path.join(args.out_path, 'log_NoAug')
    args.out_log_path = os.path.join(out_log_path_base)
    makepath(args.out_log_path)


if __name__ == '__main__':
    time_start = time.time()
    # GPU check
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    if not torch.cuda.is_available():
        print("Error: CUDA device not available. Exiting program.")
        sys.exit(1)  # Exit with error code 1

    # Variable Space
    parser = create_parser()
    args = parser.parse_args()
    # input from test.py keyboard
    args_path = args.out_path_base + '/cli_args.txt'
    # input from train.py keyboard, in cli.txt
    args = load_args(args_path, args)
    # fix seed
    args.manualSeed = 0 
    print("Random Seed: ", args.manualSeed)
    fix_seed(args.manualSeed)
    # adaptively change the args
    args = adaptive_args(args)
    # paths
    test_paths()
    # Record the training process and values
    logger = create_logger(args.out_log_path)
    logger.info('=' * 55)
    logger.info(args)
    logger.info('=' * 55)
    # load data
    test_loader, label_names, num_classes, test_data_size, eval_state, test_global_feat, samples_per_class = load_batch_data()
    time_DL_start = time.time()
    # model setting
    args.weight_path = os.path.join(args.out_path, 'best_{}_model.pth'.format(args.best_metric))
    logger.info("Load best {} model".format(args.best_metric))
    DL_model = load_model(args, num_classes, device, test=True)
    # test net
    labels_lst_0, predicted_lst_0, labels_lst_1, predicted_lst_1 = test_DL_net(DL_model)
    time_DL_end = time.time()
    # log results
    results_logging(args, logger, eval_state, label_names[0], labels_lst_0, predicted_lst_0, "aparc+aseg")
    if len(args.atlas)==2:
        results_logging(args, logger, eval_state, label_names[1], labels_lst_1, predicted_lst_1, "aparc.a2009s+aseg")

    args.root = "/media/volume/MV_HCP/TrainData_MRtrix_1000_MNI_100K/"
    print(args.root)
    # total processing time
    time_end = time.time() 
    total_time_DL = round(time_DL_end-time_DL_start, 2)
    total_time = round(time_end-time_start, 2)
    logger.info('Test on {}'.format(args.out_path))
    logger.info('Total DL processing time is {}s'.format(total_time_DL))
    logger.info('Total processing time is {}s'.format(total_time))