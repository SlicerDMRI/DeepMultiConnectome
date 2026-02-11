from __future__ import print_function
import time
import torch.utils.data as data
import torch
import numpy as np
import pickle
import sys
import os
import gc
import whitematteranalysis as wma
from pytorch3d.transforms import RotateAxisAngle, Scale, Translate

sys.path.append('..')
from utils.funcs import obtain_TractClusterMapping, cluster2tract_label, \
    get_rot_axi, array2vtkPolyData, makepath
from utils.fiber_distance import MDF_distance_calculation, MDF_distance_calculation_endpoints


class RealData_PatchData(data.Dataset):
    def __init__(self, feat, k, k_global, cal_equiv_dist=False, use_endpoints_dist=False,
                 rough_num_fiber_each_iter=10000, k_ds_rate=0.1):
        self.feat = feat.astype(np.float32)
        self.k = k
        self.k_global = k_global
        self.cal_equiv_dist = cal_equiv_dist
        self.use_endpoints_dist = use_endpoints_dist
        self.rough_num_fiber_each_iter = rough_num_fiber_each_iter
        self.k_ds_rate = k_ds_rate

        num_fiber = self.feat.shape[0]
        num_point = self.feat.shape[1]
        num_feat_per_point = self.feat.shape[2]

        if self.k_global == 0:
            self.global_feat = np.zeros((1, num_point, num_feat_per_point, 1), dtype=np.float32)
        else:
            random_idx = np.random.randint(0, num_fiber, self.k_global)
            self.global_feat = self.feat[random_idx, ...]
            self.global_feat = self.global_feat.transpose(1, 2, 0)[None, :, :, :].astype(np.float32)

        if self.k == 0:
            self.local_feat = np.zeros((num_fiber, num_point, num_feat_per_point, 1), dtype=np.float32)
        else:
            self.local_feat = np.zeros((num_fiber, num_point, num_feat_per_point, self.k), dtype=np.float32)
            num_iter = num_fiber // self.rough_num_fiber_each_iter
            self.num_fiber_each_iter = (num_fiber // num_iter) + 1
            for i_iter in range(num_iter):
                cur_feat = self.feat[i_iter * self.num_fiber_each_iter:(i_iter + 1) * self.num_fiber_each_iter, ...]
                cur_feat = np.transpose(cur_feat, (0, 2, 1))
                cur_local_feat = cal_local_feat(cur_feat, self.k_ds_rate, self.k, self.use_endpoints_dist, self.cal_equiv_dist)
                cur_local_feat = cur_local_feat.reshape(cur_feat.shape[0], self.k, num_feat_per_point, num_point)
                cur_local_feat = np.transpose(cur_local_feat, (0, 3, 2, 1))
                self.local_feat[i_iter * self.num_fiber_each_iter:(i_iter + 1) * self.num_fiber_each_iter, ...] = cur_local_feat

    def __getitem__(self, index):
        point_set = self.feat[index]
        klocal_point_set = self.local_feat[index]

        if point_set.dtype == 'float32':
            point_set = torch.from_numpy(point_set)
            klocal_point_set = torch.from_numpy(klocal_point_set)
        else:
            point_set = torch.from_numpy(point_set.astype(np.float32))
            klocal_point_set = torch.from_numpy(klocal_point_set.astype(np.float32))
            print('Feature is not in float32 format')

        return point_set, klocal_point_set

    def __len__(self):
        return self.feat.shape[0]


class unrelatedHCP_PatchData(data.Dataset):
    def __init__(self, root, out_path, logger, split='train', num_fiber_per_brain=10000, num_point_per_fiber=15,
                 use_tracts_training=False, k=0, k_global=0, rot_ang_lst=None, scale_ratio_range=None,
                 trans_dis=0.0, aug_axis_lst=None, aug_times=10, cal_equiv_dist=False, k_ds_rate=0.1,
                 recenter=False, include_org_data=False, atlas='aparc+aseg', threshold=0):
        self.root = root
        self.out_path = out_path
        self.split = split
        self.logger = logger
        self.num_fiber = num_fiber_per_brain
        self.num_point = num_point_per_fiber
        self.use_tracts_training = use_tracts_training
        self.k = k
        self.k_global = k_global
        self.rot_ang_lst = rot_ang_lst or [0, 0, 0]
        self.scale_ratio_range = scale_ratio_range or [0, 0]
        self.trans_dis = trans_dis
        self.aug_axis_lst = aug_axis_lst or ['LR', 'AP', 'SI']
        self.aug_times = aug_times
        self.k_ds_rate = k_ds_rate
        self.recenter = recenter
        self.atlas = atlas
        self.threshold = threshold
        self.include_org_data = include_org_data
        self.num_labels = {"aparc+aseg": 85, "aparc.a2009s+aseg": 165}

        self.save_aug_data = True
        self.cal_equiv_dist = cal_equiv_dist
        self.use_endpoints_dist = False
        self.logger.info('cal_equiv_dist: {}, use_endpoints_dist: {}'.format(self.cal_equiv_dist, self.use_endpoints_dist))

        with open(os.path.join(self.root, '{}.pickle'.format(self.split)), 'rb') as file:
            data_dict = pickle.load(file)

        self.features = data_dict['feat']
        self.subject_ids = data_dict['subject_id']

        self.labels = []
        self.label_names = []
        for atlas_name in self.atlas:
            atlas_label_names = data_dict[f'label_name_{atlas_name}']
            self.label_names.append(atlas_label_names)

            atlas_labels = data_dict[f'label_{atlas_name}']
            atlas_labels = np.where(np.isin(atlas_labels, list(range(self.num_labels[atlas_name]))), 0, atlas_labels)
            atlas_labels = self._threshold(atlas_labels, atlas_name)
            self.labels.append(atlas_labels)

            self.logger.info("Total labels with streamlines: {}/{} for {} atlas".format(
                len(np.unique(atlas_labels)), len(atlas_label_names), atlas_name))

        if len(self.atlas) == 1:
            self.labels = self.labels[0]
            self.label_names = self.label_names[0]

        self.logger.info('Load {} data'.format(self.split))

        self.num_classes = [len(np.unique(self.label_names[0])), len(np.unique(self.label_names[1]))]
        self.samples_per_class = self._compute_samples_per_class()

        self.brain_features, self.brain_labels = self._cal_brain_feat()
        self.org_feat, self.org_label, self.local_feat, self.global_feat, self.new_subidx = self._cal_info_feat()

    def _threshold(self, labels, atlas_name):
        if self.split == "train":
            subject_count = len(np.unique(self.subject_ids))
            label_subject_counts = {label: 0 for label in np.unique(labels)}

            for subject_id in np.unique(self.subject_ids):
                subject_indices = np.where(self.subject_ids == subject_id)[0]
                subject_labels = np.unique(labels[subject_indices])
                for label in subject_labels:
                    label_subject_counts[label] += 1

            min_subjects_required = (self.threshold / 100) * subject_count
            rare_labels = [label for label, count in label_subject_counts.items() if count < min_subjects_required]
            labels = np.where(np.isin(labels, rare_labels), 0, labels)

            file_path = os.path.join(self.root, f'thresholded_labels_{self.threshold}_{atlas_name}.txt')
            with open(file_path, 'w') as f:
                for label in rare_labels:
                    f.write(f'{label}\n')

        elif self.split in ["val", "test"]:
            file_path = os.path.join(self.root, f'thresholded_labels_{self.threshold}_{atlas_name}.txt')
            if os.path.exists(file_path):
                with open(file_path, 'r') as f:
                    rare_labels = [int(label.strip()) for label in f.readlines()]
                labels = np.where(np.isin(labels, rare_labels), 0, labels)
            else:
                raise FileNotFoundError(f"Rare labels file not found: {file_path}")

        self.logger.info(f'Labels set to 0 due to {self.threshold}% thresholding: {len(np.unique(rare_labels))}')
        return labels

    def __getitem__(self, index):
        point_set = self.org_feat[index]
        label = self.org_label[index]
        klocal_point_set = self.local_feat[index]
        new_subidx = self.new_subidx[index]

        if point_set.dtype == 'float32':
            point_set = torch.from_numpy(point_set)
            klocal_point_set = torch.from_numpy(klocal_point_set)
        else:
            point_set = torch.from_numpy(point_set.astype(np.float32))
            klocal_point_set = torch.from_numpy(klocal_point_set.astype(np.float32))
            print('Feature is not in float32 format')

        if label.dtype == 'int64':
            label = torch.from_numpy(label)
            new_subidx = torch.from_numpy(new_subidx)
        else:
            label = torch.from_numpy(label.astype(np.int64))
            new_subidx = torch.from_numpy(new_subidx.astype(np.int64))
            print('Label is not in int64 format')

        return point_set, label, klocal_point_set, new_subidx

    def __len__(self):
        return self.org_feat.shape[0]

    def _cal_brain_feat(self):
        num_feat_per_point = self.features.shape[2]
        unique_subject_ids = np.unique(self.subject_ids)
        num_subject = len(unique_subject_ids)

        if self.aug_times > 0:
            brain_features = np.zeros((num_subject * self.aug_times, self.num_fiber, self.num_point, num_feat_per_point), dtype=np.float32)
            brain_labels = np.zeros((num_subject * self.aug_times, self.num_fiber, 1), dtype=np.int64)
            aug_matrices = np.zeros((num_subject, self.aug_times, 4, 4), dtype=np.float32)
        else:
            brain_features = np.zeros((num_subject, self.num_fiber, self.num_point, num_feat_per_point), dtype=np.float32)
            brain_labels = np.zeros((num_subject, self.num_fiber, len(self.atlas)), dtype=np.int64)

        for i_subject, unique_id in enumerate(unique_subject_ids):
            cur_idxs = np.where(self.subject_ids == unique_id)[0]
            np.random.shuffle(cur_idxs)
            cur_select_idxs = cur_idxs[:self.num_fiber]
            cur_features = self.features[cur_select_idxs, :, :]
            if len(self.atlas) > 1:
                cur_labels = np.concatenate([self.labels[i][cur_select_idxs, None] for i in range(len(self.atlas))], axis=1)
            else:
                cur_labels = self.labels[cur_select_idxs, None]

            if self.aug_times > 0:
                cur_features = torch.from_numpy(cur_features)
                aug_features = np.zeros((self.aug_times, *cur_features.shape))
                for i_aug in range(self.aug_times):
                    trot = None
                    cur_angles = []
                    for i, rot_ang in enumerate(self.rot_ang_lst):
                        angle = ((torch.rand(1) - 0.5) * 2 * rot_ang).item()
                        rot_axis_name = get_rot_axi(self.aug_axis_lst[i])
                        cur_trot = RotateAxisAngle(angle=angle, axis=rot_axis_name, degrees=True)
                        cur_angles.append(round(angle, 1))
                        trot = cur_trot if trot is None else trot.compose(cur_trot)

                    if self.scale_ratio_range[0] == 0 and self.scale_ratio_range[1] == 0:
                        scale_r = 1.0
                    else:
                        scale_r = torch.distributions.Uniform(
                            1 - self.scale_ratio_range[0], 1 + self.scale_ratio_range[1]
                        ).sample().item()
                    cur_trot = Scale(scale_r)
                    trot = trot.compose(cur_trot)

                    LR_trans = ((torch.rand(1) - 0.5) * 2 * self.trans_dis).item()
                    AP_trans = ((torch.rand(1) - 0.5) * 2 * self.trans_dis).item()
                    SI_trans = ((torch.rand(1) - 0.5) * 2 * self.trans_dis).item()
                    cur_trot = Translate(LR_trans, AP_trans, SI_trans)
                    trot = trot.compose(cur_trot)

                    aug_matrices[i_subject, i_aug, :, :] = np.array(trot.get_matrix())
                    aug_feat = trot.transform_points(cur_features.float()).numpy()

                    if self.recenter:
                        aug_feat = center_tractography(self.root, aug_feat)
                        self.logger.info('Subject idx {} (unique ID {}, aug {}): rotation {}, scale {}, translation {} (centered). Aug axis order: {}'.format(
                            i_subject, unique_id, i_aug, cur_angles, round(scale_r, 3),
                            [round(LR_trans, 1), round(AP_trans, 1), round(SI_trans, 1)], self.aug_axis_lst))
                    else:
                        self.logger.info('Subject idx {} (unique ID {}, aug {}): rotation {}, scale {}, translation {}. Aug axis order: {}'.format(
                            i_subject, unique_id, i_aug, cur_angles, round(scale_r, 3),
                            [round(LR_trans, 1), round(AP_trans, 1), round(SI_trans, 1)], self.aug_axis_lst))

                    aug_features[i_aug, ...] = aug_feat

                    if self.save_aug_data and i_subject < 5:
                        aug_data_save_path = os.path.join(self.out_path, 'AugmentedData', self.split)
                        makepath(aug_data_save_path)
                        aug_feat_pd = array2vtkPolyData(aug_feat)
                        if self.recenter:
                            aug_feat_name = 'SubID{}Aug{}_RotR{}A{}S{}_Scale{}_TransR{}A{}S{}_Recenter'.format(
                                i_subject, i_aug, cur_angles[0], cur_angles[1], cur_angles[2],
                                round(scale_r, 3), round(LR_trans, 1), round(AP_trans, 1), round(SI_trans, 1))
                        else:
                            aug_feat_name = 'SubID{}Aug{}_RotR{}A{}S{}_Scale{}_TransR{}A{}S{}'.format(
                                i_subject, i_aug, cur_angles[0], cur_angles[1], cur_angles[2],
                                round(scale_r, 3), round(LR_trans, 1), round(AP_trans, 1), round(SI_trans, 1))
                        aug_feat_name = aug_feat_name.replace('.', '`') + '.vtk'
                        wma.io.write_polydata(aug_feat_pd, os.path.join(aug_data_save_path, aug_feat_name))

                brain_features[i_subject * self.aug_times:(i_subject + 1) * self.aug_times, :, :, :] = aug_features
            else:
                try:
                    brain_features[i_subject, :, :, :] = cur_features
                except Exception:
                    print(f"cur_features has wrong shape!: {cur_features.shape}")

            if self.use_tracts_training:
                ordered_tract_cluster_mapping_dict = obtain_TractClusterMapping()
                cur_labels = cluster2tract_label(cur_labels, ordered_tract_cluster_mapping_dict, output_lst=False)
            if self.aug_times > 0:
                brain_labels[i_subject * self.aug_times:(i_subject + 1) * self.aug_times, ...] = cur_labels[None, ...].repeat(self.aug_times, axis=0)
            else:
                try:
                    brain_labels[i_subject, ...] = cur_labels
                except Exception:
                    pass

        if self.aug_times > 0:
            np.save(os.path.join(self.out_path, '{}_aug_matrices.npy'.format(self.split)), aug_matrices)
            if self.include_org_data:
                assert self.num_fiber == 10000
                org_features = self.features.reshape(num_subject, self.num_fiber, self.num_point, num_feat_per_point)
                org_labels = self.labels.reshape(num_subject, self.num_fiber, 1)
                brain_features = np.concatenate((brain_features, org_features), axis=0)
                brain_labels = np.concatenate((brain_labels, org_labels), axis=0)
                self.logger.info('Include {} original data in the {} data.'.format(org_features.shape[0], self.split))

        return brain_features, brain_labels

    def _cal_info_feat(self):
        num_subjects = self.brain_features.shape[0]
        num_feat_per_point = self.brain_features.shape[-1]

        use_memap = True
        if use_memap:
            mem_path = os.path.join(self.out_path, 'TempMemory', self.split)
            makepath(mem_path)
            local_feat_path = os.path.join(mem_path, 'local_feat.dat')
            global_feat_path = os.path.join(mem_path, 'global_feat.dat')
            new_subidx_path = os.path.join(mem_path, 'new_subidx.dat')
            if self.k > 0:
                local_feat = np.memmap(local_feat_path, dtype=np.float32, mode='w+',
                                       shape=(*self.brain_features.shape, self.k))
            else:
                local_feat = np.memmap(local_feat_path, dtype=np.float32, mode='w+',
                                       shape=(*self.brain_features.shape, 1))
                local_feat = local_feat.reshape(-1, self.num_point, num_feat_per_point, 1)
            if self.k_global > 0:
                global_feat = np.memmap(global_feat_path, dtype=np.float32, mode='w+',
                                        shape=(num_subjects, self.num_point, num_feat_per_point, self.k_global))
            else:
                global_feat = np.memmap(global_feat_path, dtype=np.float32, mode='w+',
                                        shape=(num_subjects, self.num_point, num_feat_per_point, 1))
            new_subidx = np.memmap(new_subidx_path, dtype=np.int64, mode='w+',
                                   shape=(num_subjects, self.num_fiber))
        else:
            if self.k > 0:
                local_feat = np.zeros((*self.brain_features.shape, self.k), dtype=np.float32)
            else:
                local_feat = np.zeros((*self.brain_features.shape, 1), dtype=np.float32)
                local_feat = local_feat.reshape(-1, self.num_point, num_feat_per_point, 1)
            if self.k_global > 0:
                global_feat = np.zeros((num_subjects, self.num_point, num_feat_per_point, self.k_global), dtype=np.float32)
            else:
                global_feat = np.zeros((num_subjects, self.num_point, num_feat_per_point, 1), dtype=np.float32)
            new_subidx = np.zeros((num_subjects, self.num_fiber), dtype=np.int64)

        for cur_idx in range(num_subjects):
            time_start = time.time()
            cur_feat = self.brain_features[cur_idx, ...]
            cur_feat = np.transpose(cur_feat, (0, 2, 1))
            if self.k > 0:
                cur_local_feat = cal_local_feat(cur_feat, self.k_ds_rate, self.k, self.use_endpoints_dist, self.cal_equiv_dist)
                cur_local_feat = cur_local_feat.reshape(self.num_fiber, self.k, num_feat_per_point, self.num_point)
                cur_local_feat = np.transpose(cur_local_feat, (0, 3, 2, 1))
                local_feat[cur_idx, ...] = cur_local_feat
                del cur_local_feat
            if self.k_global > 0:
                random_idx = np.random.randint(0, cur_feat.shape[0], self.k_global)
                cur_global_feat = cur_feat[random_idx, ...]
                cur_global_feat = cur_global_feat.transpose(2, 1, 0)
                global_feat[cur_idx, ...] = cur_global_feat
                del cur_global_feat
            cur_subidx = np.ones((cur_feat.shape[0]), dtype=np.int64) * cur_idx
            new_subidx[cur_idx, ...] = cur_subidx
            time_end = time.time()

            if self.aug_times > 0:
                self.logger.info('Subject {} Aug {} with {} fibers feature calculation time: {:.2f} s'.format(
                    cur_idx // self.aug_times, cur_idx % self.aug_times, self.num_fiber, time_end - time_start))
            else:
                self.logger.info('Subject {} (No Aug) with {} fibers feature calculation time: {:.2f} s'.format(
                    cur_idx, self.num_fiber, time_end - time_start))

            del cur_feat, cur_subidx
            gc.collect()

        if self.k > 0:
            local_feat = local_feat.reshape(-1, self.num_point, num_feat_per_point, self.k)
        new_subidx = new_subidx.reshape(-1, 1)

        fiber_feat = self.brain_features.reshape(-1, self.num_point, num_feat_per_point)
        fiber_label = self.brain_labels.reshape(-1, len(self.atlas))

        return fiber_feat, fiber_label, local_feat, global_feat, new_subidx

    def _compute_samples_per_class(self):
        if len(self.atlas) > 1:
            samples_per_class = [torch.bincount(torch.tensor(self.labels[i]), minlength=self.num_classes[i])
                                 for i in range(len(self.atlas))]
        else:
            samples_per_class = torch.bincount(torch.tensor(self.labels), minlength=self.num_classes[0])
        return samples_per_class


def cal_local_feat(cur_feat, k_ds_rate, k, use_endpoints_dist, cal_equiv_dist):
    near_idx, near_flip_mask, ds_cur_feat, ds_cur_feat_equiv = dist_mat_knn(
        torch.from_numpy(cur_feat), k_ds_rate, k, use_endpoints_dist, cal_equiv_dist)
    cur_local_feat_org = ds_cur_feat[near_idx.reshape(-1), ...]
    cur_local_feat_equiv = ds_cur_feat_equiv[near_idx.reshape(-1), ...]
    near_flip_mask = near_flip_mask.reshape(-1)[:, None, None]
    near_nonflip_mask = 1 - near_flip_mask
    cur_local_feat = cur_local_feat_org * near_nonflip_mask + cur_local_feat_equiv * near_flip_mask
    return cur_local_feat


def dist_mat_knn(brain_feat, k_ds_rate, k, use_endpoints_dist, cal_equiv_dist):
    if 0 < k_ds_rate < 1:
        num_ds_feat = int(brain_feat.shape[0] * k_ds_rate)
        ds_indices = np.random.choice(brain_feat.shape[0], size=num_ds_feat, replace=False)
        downsample_feat = brain_feat[ds_indices, :, :]
    else:
        downsample_feat = brain_feat

    if use_endpoints_dist:
        dist_mat, flip_mask, ds_brain_feat, ds_brain_feat_equiv = MDF_distance_calculation_endpoints(
            brain_feat, downsample_feat, cal_equiv=cal_equiv_dist)
    else:
        dist_mat, flip_mask, ds_brain_feat, ds_brain_feat_equiv = MDF_distance_calculation(
            brain_feat, downsample_feat, cal_equiv=cal_equiv_dist)

    topk_idx = dist_mat.topk(k=k, largest=False, dim=-1)[1]
    near_idx = topk_idx[:, :]
    near_flip_mask = torch.gather(flip_mask, dim=1, index=near_idx)

    return near_idx.numpy(), near_flip_mask.numpy(), ds_brain_feat.numpy(), ds_brain_feat_equiv.numpy()


def center_tractography(input_path, feat_RAS, out_path=None, logger=None, tractography_name=None, save_data=False):
    """Recenter the tractography to atlas center."""
    HCP_center = np.load(os.path.join(input_path, 'HCP_mass_center.npy'))
    test_subject_center = np.mean(feat_RAS, axis=0)
    displacement = HCP_center - test_subject_center
    c_feat_RAS = feat_RAS + displacement
    if save_data:
        recenter_path = os.path.join(out_path, 'recentered_tractography')
        makepath(recenter_path)
        feat_RAS_pd = array2vtkPolyData(c_feat_RAS)
        wma.io.write_polydata(feat_RAS_pd, os.path.join(recenter_path, 'recentered_{}'.format(tractography_name)))
        if logger is not None:
            logger.info('Saved recentered tractography to {}'.format(recenter_path))
    return c_feat_RAS
