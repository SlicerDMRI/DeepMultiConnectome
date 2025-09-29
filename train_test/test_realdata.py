import whitematteranalysis as wma
import argparse
import os
import time

import torch
import torch.nn.parallel

import sys
sys.path.append('../')

from utils.logger import create_logger
from utils.funcs import cluster2tract_label, makepath,  obtain_TractClusterMapping,tractography_parcellation
from utils.cli import load_args_in_testing_only
from train import load_model, train_val_test_forward
from datasets.dataset import RealData_PatchData, center_tractography
import utils.tract_feat as tract_feat
from tractography.label_encoder import * 
from utils.metrics_connectome import *


def test_realdata_DL_net(net):
    """test the network"""
    test_predicted_lst_0, test_predicted_lst_1 = [], []
    # test
    with torch.no_grad():
        for j, data in enumerate(test_loader, start=0):
            _, _, test_predicted_lst_0, _, test_predicted_lst_1 = \
                train_val_test_forward(j, data, net, 'test_realdata', -1, [], test_predicted_lst_0, [], test_predicted_lst_1,
                                       args, device, args.num_classes, epoch=1, eval_global_feat=test_realdata_global_feat)

    return test_predicted_lst_0, test_predicted_lst_1


start_time = time.time()
use_cpu = False
if use_cpu:
    device = torch.device("cpu")
else:
    device = torch.device("cuda:0")

# Parse arguments
parser = argparse.ArgumentParser(description="Test on real data", epilog="Referenced from https://github.com/SlicerDMRI/SupWMA"
                                             "Tengfei Xue txue4133@uni.sydney.edu.au")
# paths
parser.add_argument('--input_path', type=str, default='../TrainData_800clu800ol',
                    help='The input path for train/val/test (atlas) data')
parser.add_argument('--weight_path_base', type=str, help='pretrained network model')
parser.add_argument('--tractography_path', type=str, help='Tractography data as a vtkPolyData file')
parser.add_argument('--out_path', type=str, help='The output directory can be a new empty directory. It will be created if needed.')
# model parameters
parser.add_argument('--model_name', type=str, default='dgcnn', help='The name of the point cloud model')
parser.add_argument('--k', type=int, default=20, help='the number of neighbor points (in streamline level)')
parser.add_argument('--k_global', type=int, default=80, help='The number of points (in streamline level) for the random sparse sampling')
parser.add_argument('--k_point_level', type=int, default=5, help='the number of neighbor points (in point level)')
parser.add_argument('--emb_dims', type=int, default=1024, metavar='N',help='Dimension of embeddings')
parser.add_argument('--dropout', type=float, default=0.5, help='initial dropout rate')
# process parameters
parser.add_argument('--test_realdata_batch_size', type=int, default=6144, help='batch size')
parser.add_argument('--num_fiber_per_brain', type=int, default=10000, help='number of fibers for each brain, keep consistent with the training data')
parser.add_argument('--num_points', type=int, default=15, help='number of points on each fiber in the tractography')
parser.add_argument('--num_classes', type=int, default=1600, help='number of classes')
parser.add_argument('--cal_equiv_dist', default=False, action='store_true', help='Calculate equivalent distance for pairwise distance matrix for finding neighbors')
parser.add_argument('--k_ds_rate', type=float, default=0.1, help='downsample the tractography when calculating pairwise distance matrix')

parser.add_argument('--atlas', type=str, default='aparc+aseg', help='Atlas used')
parser.add_argument('--save_parcellated', default=False, action='store_true', help='save parcellated streamlines per class')
parser.add_argument('--fibersampling', type=float, default=0, help='Distance in fiber sampling, higher means points higher distribution near ends')
    

args = parser.parse_args()
# input from test_realdata.py keyboard
args_path = args.weight_path_base + '/cli_args.txt'
# input from train.py keyboard, in cli.txt
args = load_args_in_testing_only(args_path, args)
# paths
args.weight_path = os.path.join(args.weight_path_base, 'best_f1_model.pth')
    
makepath(args.out_path)
# create logger
log_path = os.path.join(args.out_path, 'log')
makepath(log_path)
logger = create_logger(log_path)
logger.info('=' * 55)
logger.info(args)
logger.info('=' * 55)

# label names
num_labels = {"aparc+aseg": 85, "aparc.a2009s+aseg": 165}
label_names = {}

for atlas in args.atlas:
    label_names_tuple = list(generate_label_dict(num_labels[atlas], 'symmetric'))
    label_names_str = []
    for label in label_names_tuple:
        label_names_str.append(f"{label[0]}_{label[1]}")
    label_names[atlas] = label_names_str

# read tractography into feature
time_start_read = time.time()
pd_tractography = wma.io.read_polydata(args.tractography_path)
logger.info('Finish reading tractography from: {}, took {} s'.format(args.tractography_path, round(time.time()-time_start_read, 3)))
tractography_file_n = os.path.basename(os.path.normpath(args.tractography_path))

# load non-registered feature 
time_start_load = time.time()
logger.info('Extracting features from tractography'.format(args.tractography_path))
feat_RAS, _ = tract_feat.feat_RAS(pd_tractography, number_of_points=args.num_points, decay_factor=args.fibersampling)
logger.info('The number of fibers in tractography is {}'.format(feat_RAS.shape[0]))
logger.info('Extracting RAS features is done, took {} s'.format(round(time.time()-time_start_load, 3)))

# Real data processing
time_start_processing = time.time()
test_realdata = RealData_PatchData(feat_RAS, k=args.k, k_global=args.k_global, cal_equiv_dist=args.cal_equiv_dist, 
                                    use_endpoints_dist=False, rough_num_fiber_each_iter=args.num_fiber_per_brain, k_ds_rate=args.k_ds_rate) 
test_loader = torch.utils.data.DataLoader(test_realdata, batch_size=args.test_realdata_batch_size, shuffle=False)
test_realdata_global_feat = test_realdata.global_feat
test_realdata_size = len(test_realdata)
logger.info('calculating knn+random features is done, took {} s'.format(round(time.time()-time_start_processing, 3)))

# test network
time_start_model = time.time()
if len(args.atlas)==1:
    DL_model = load_model(args, num_classes=len(label_names[args.atlas[0]]), device=device, test=True)  
    pred_labels_0 = test_realdata_DL_net(DL_model)
    
elif len(args.atlas)==2: 
    DL_model = load_model(args, num_classes=[len(label_names[args.atlas[0]]), len(label_names[args.atlas[1]])], 
                          device=device, test=True)  
    pred_labels_0, pred_labels_1 = test_realdata_DL_net(DL_model)
else:
    raise ValueError("Currently, only supports one or two atlases.")
logger.info('Inference time is {}s'.format(round(time.time()-time_start_model, 3)))

if args.save_parcellated:
    # Save predictions per class
    time_start_write = time.time()
    tractography_parcellation(args, pd_tractography, pred_labels, label_names)  # pd tractography here is in the subject space.
    logger.info('Time to write vtks is {}s'.format(round(time.time()-time_start_write, 3)))
    
# Save predictions
for i, atlas in enumerate(args.atlas):
    # Select the correct predictions based on the atlas index
    pred_labels = pred_labels_0 if i == 0 else pred_labels_1
    num_labels_atlas = num_labels[atlas]  # Use the number of labels specific to each atlas

    # Decode the predicted labels
    pred_labels_decoded = convert_labels_list(pred_labels, encoding_type='symmetric', 
                                              mode='decode', num_labels=num_labels_atlas)

    # Write decoded predictions to a file
    with open(os.path.join(args.out_path, f'predictions_{atlas}.txt'), 'w') as file:
        for prediction in pred_labels_decoded:
            file.write(f'{prediction[0]} {prediction[1]}\n')

    # Write original predictions (without decoding) to a file for reference
    with open(os.path.join(args.out_path, f'predictions_{atlas}_symmetric.txt'), 'w') as file:
        for prediction in pred_labels:
            file.write(f'{prediction}\n')

    # Read true labels file for the current atlas
    encode_labels_txt(
        os.path.join(os.path.dirname(args.tractography_path), f"labels_10M_{atlas}.txt"),
        os.path.join(os.path.dirname(args.tractography_path), f"labels_10M_{atlas}_symmetric.txt"),
        'symmetric', num_labels=num_labels_atlas
    )

    # with open(os.path.join(os.path.dirname(args.tractography_path), f"labels_10M_{atlas}_symmetric.txt"), 'r') as file:
    #     true_labels = [int(line) for line in file]

    # # Compute connectome metrics for each atlas
    # CM = ConnectomeMetrics(true_labels=true_labels, pred_labels=pred_labels, atlas=atlas, out_path=args.out_path, graph=True)
    # logger.info(CM.format_metrics())
    
# time
end_time = time.time()
tot_time = round(end_time - start_time, 3)
logger.info('All done!!! Total time is {}s'.format(round(tot_time, 3)))
    