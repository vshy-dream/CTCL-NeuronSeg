from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import os
import sys
import cv2
import h5py
import math
import time
import torch
import random
import numpy as np
from PIL import Image
import multiprocessing
from joblib import delayed
from joblib import Parallel
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import glob
import glob

from augmentation import Flip
from augmentation import Elastic
from augmentation import Grayscale
from augmentation import Rotate
from augmentation import Rescale
from utils.seg_util import mknhood3d, genSegMalis
from utils.aff_util import seg_to_affgraph
from utils.utils import center_crop
from data.data_segmentation import seg_widen_border, weight_binary_ratio
from data.data_affinity import seg_to_aff
# from utils.affinity_official import seg2affs
from utils.affinity_ours import gen_affs_mutex_3d
from utils.gen_skele import gen_skele_3d, gen_distanceTransform


class Train(Dataset):
    def __init__(self, cfg):
        super(Train, self).__init__()
        # multiprocess settings
        num_cores = multiprocessing.cpu_count()
        self.parallel = Parallel(n_jobs=num_cores, backend='threading')
        self.cfg = cfg
        self.model_type = cfg.MODEL.model_type
        self.if_dilate = cfg.DATA.if_dilate
        # self.shift_channels = cfg.shift
        self.shift_channels = None
        self.if_skele = cfg.MODEL.if_skele
        self.if_dt = cfg.MODEL.if_dt
        self.skeleton = None
        self.skeletons = None
        try:
            self.if_cascaded = cfg.MODEL.if_cascaded
        except AttributeError:
            self.if_cascaded = None

        # basic settings
        # the input size of network
        if cfg.MODEL.model_type == 'superhuman':
            self.crop_size = [18, 160, 160]
            self.net_padding = [0, 0, 0]
            if cfg.DATA.dataset_name == 'zebrafinch':
                if hasattr(cfg.DATA, 'raw_3d') and cfg.DATA.raw_3d is True:
                    self.crop_size = [18, 128, 128]
                    self.net_padding = [0, 0, 0]
                else:
                    self.crop_size = [64, 64, 64]
                    self.net_padding = [16, 16, 16]
        elif cfg.MODEL.model_type == 'unet_lsd':
            # self.crop_size = [16, 128, 128]
            # self.net_padding = [0, 0, 0]
            self.crop_size = [18, 160, 160]
            self.net_padding = [0, 0, 0]
        elif cfg.MODEL.model_type == 'mala':
            self.crop_size = [53, 268, 268]
            self.net_padding = [14, 106, 106]  # the edge size of patch reduced by network
        elif cfg.MODEL.model_type == 'segmamba':
            self.crop_size = [16, 160, 160]
            self.net_padding = [0, 0, 0]
            if cfg.DATA.dataset_name == 'zebrafinch':
                if hasattr(cfg.DATA, 'raw_3d') and cfg.DATA.raw_3d is True:
                    self.crop_size = [16, 128, 128]
                    self.net_padding = [0, 0, 0]
                else:
                    self.crop_size = [64, 64, 64]
                    self.net_padding = [16, 16, 16]
        elif cfg.MODEL.model_type == 'unetr':
            self.crop_size = [32, 160, 160]
            self.net_padding = [0, 0, 0]
            if cfg.DATA.dataset_name == 'zebrafinch':
                if hasattr(cfg.DATA, 'raw_3d') and cfg.DATA.raw_3d is True:
                    self.crop_size = [16, 128, 128]
                    self.net_padding = [0, 0, 0]
                else:
                    self.crop_size = [64, 64, 64]
                    self.net_padding = [16, 16, 16]
        else:
            raise AttributeError('No this model type!')

        # the output size of network
        # for mala: [25, 56, 56]
        # for superhuman: [18, 160, 160]
        self.out_size = [self.crop_size[k] - 2 * self.net_padding[k] for k in range(len(self.crop_size))]

        # training dataset files (h5), may contain many datasets
        # if cfg.DATA.dataset_name == 'cremi-A' or cfg.DATA.dataset_name == 'cremi':
        #     self.sub_path = 'cremi'
        #     self.train_datasets = ['cremiA_inputs_interp.h5']
        #     self.train_labels = ['cremiA_labels.h5']
        # elif cfg.DATA.dataset_name == 'cremi-B':
        #     self.sub_path = 'cremi'
        #     self.train_datasets = ['cremiB_inputs_interp.h5']
        #     self.train_labels = ['cremiB_labels.h5']
        # elif cfg.DATA.dataset_name == 'cremi-C':
        #     self.sub_path = 'cremi'
        #     self.train_datasets = ['cremiC_inputs_interp.h5']
        #     self.train_labels = ['cremiC_labels.h5']
        # elif cfg.DATA.dataset_name == 'cremi-all':
        #     self.sub_path = 'cremi'
        #     self.train_datasets = ['cremiA_inputs_interp.h5', 'cremiB_inputs_interp.h5', 'cremiC_inputs_interp.h5']
        #     self.train_labels = ['cremiA_labels.h5', 'cremiB_labels.h5', 'cremiC_labels.h5']
        if cfg.DATA.dataset_name == 'cremi-A' or cfg.DATA.dataset_name == 'cremi':
            self.sub_path = 'cremi'
            self.train_datasets = ['cremiA_inputs.h5']
            self.train_labels = ['cremiA_labels.h5']
        elif cfg.DATA.dataset_name == 'cremi-B':
            self.sub_path = 'cremi'
            self.train_datasets = ['cremiB_inputs.h5']
            self.train_labels = ['cremiB_labels.h5']
        elif cfg.DATA.dataset_name == 'cremi-C':
            self.sub_path = 'cremi'
            self.train_datasets = ['cremiC_inputs.h5']
            self.train_labels = ['cremiC_labels.h5']
        elif cfg.DATA.dataset_name == 'cremi-all':
            self.sub_path = 'cremi'
            self.train_datasets = ['cremiA_inputs.h5', 'cremiB_inputs.h5', 'cremiC_inputs.h5']
            self.train_labels = ['cremiA_labels.h5', 'cremiB_labels.h5', 'cremiC_labels.h5']
        elif cfg.DATA.dataset_name == 'isbi':
            self.sub_path = 'snemi3d'
            self.train_datasets = ['isbi_inputs.h5']
            self.train_labels = ['isbi_labels.h5']
        elif cfg.DATA.dataset_name == 'ac3':
            self.sub_path = 'ac3_ac4'
            self.train_datasets = ['AC3_inputs.h5']
            self.train_labels = ['AC3_labels.h5']
        elif cfg.DATA.dataset_name == 'ac4':
            self.sub_path = 'ac3_ac4'
            self.train_datasets = ['AC4_inputs.h5']
            self.train_labels = ['AC4_labels.h5']
        elif cfg.DATA.dataset_name == 'fib':
            self.sub_path = 'fib'
            self.train_datasets = ['fib_inputs.h5']
            self.train_labels = ['fib_labels.h5']
        elif cfg.DATA.dataset_name == 'wafer':
            self.sub_path = 'wafer'
            self.train_datasets = ['wafer25_inputs.h5', 'wafer26_inputs.h5', 'wafer26_2_inputs.h5', 'wafer36_inputs.h5']
            self.train_labels = ['wafer25_labels.h5', 'wafer26_labels.h5', 'wafer26_2_labels.h5', 'wafer36_labels.h5']
        elif cfg.DATA.dataset_name == 'wafer4':
            self.sub_path = 'wafer'
            self.train_datasets = ['wafer4_inputs.h5']
            self.train_labels = ['wafer4_labels.h5']
        elif cfg.DATA.dataset_name == 'zebrafinch':
            self.sub_path = 'zebrafinch/train'
            self.folder_name = os.path.join(cfg.DATA.data_folder, self.sub_path)
            pattern = f'{self.folder_name}/image*.h5'
            file_paths = sorted(glob.glob(pattern))
            self.train_datasets = [os.path.basename(file_path) for file_path in file_paths]
            pattern = f'{self.folder_name}/gt*.h5'
            file_paths = sorted(glob.glob(pattern))
            self.train_labels = [os.path.basename(file_path) for file_path in file_paths]
        else:
            raise AttributeError('No this dataset type!')

        # the path of datasets, need first-level and second-level directory, such as: os.path.join('../data', 'cremi')
        self.folder_name = os.path.join(cfg.DATA.data_folder, self.sub_path)
        assert len(self.train_datasets) == len(self.train_labels)

        # split training data
        self.train_split = cfg.DATA.train_split

        # augmentation
        self.if_scale_aug = cfg.DATA.if_scale_aug
        self.if_filp_aug = cfg.DATA.if_filp_aug
        self.if_elastic_aug = cfg.DATA.if_elastic_aug
        self.if_intensity_aug = cfg.DATA.if_intensity_aug
        self.if_rotation_aug = cfg.DATA.if_rotation_aug

        # load dataset
        self.dataset = []
        self.labels = []
        if cfg.DATA.dataset_name == 'wafer':
            for k in range(len(self.train_datasets)):
                print('load ' + self.train_datasets[k] + ' ...')
                # load raw data
                f_raw = h5py.File(os.path.join(self.folder_name, self.train_datasets[k]), 'r')
                data = f_raw['main'][:]
                f_raw.close()
                # data = data[:self.train_split]
                self.dataset.append(data)

                # load labels
                f_label = h5py.File(os.path.join(self.folder_name, self.train_labels[k]), 'r')
                label = f_label['main'][:]
                f_label.close()
                # label = label[:self.train_split]
                if self.if_dilate:
                    # label = genSegMalis(label, 1)
                    label = seg_widen_border(label, tsz_h=1)
                self.labels.append(label)
        elif cfg.DATA.dataset_name == 'zebrafinch':
            for k in range(len(self.train_datasets)):
                print('load ' + self.train_datasets[k] + ' ...')
                # load raw data
                f_raw = h5py.File(os.path.join(self.folder_name, self.train_datasets[k]), 'r')
                data = f_raw['main'][:]
                f_raw.close()
                # data = data[:self.train_split]

                # 如果模型类型是mala，调整数据尺寸为[53, 268, 268]
                # if cfg.MODEL.model_type == 'mala':
                #     # 使用resize或插值方法将数据调整为指定大小
                #     orig_shape = data.shape
                #     zoom_factors = [300/orig_shape[0], 300/orig_shape[1], 300/orig_shape[2]]
                #     data = zoom(data, zoom_factors, order=1)  # order=1表示线性插值
                #     print(f'Resized zebrafinch data from {orig_shape} to {data.shape} for mala model')

                self.dataset.append(data)

                # load labels
                f_label = h5py.File(os.path.join(self.folder_name, self.train_labels[k]), 'r')
                label = f_label['main'][:]
                f_label.close()
                # label = label[:self.train_split]
                # if cfg.MODEL.model_type == 'mala':
                #     # 使用resize或插值方法将数据调整为指定大小
                #     orig_shape = label.shape
                #     zoom_factors = [300/orig_shape[0], 300/orig_shape[1], 300/orig_shape[2]]
                #     label = zoom(label, zoom_factors, order=0)  # order=0表示最近邻插值
                #     print(f'Resized zebrafinch label from {orig_shape} to {label.shape} for mala model')

                if self.if_dilate:
                    # label = genSegMalis(label, 1)
                    label = seg_widen_border(label, tsz_h=1)
                self.labels.append(label)
        else:
            for k in range(len(self.train_datasets)):
                print('load ' + self.train_datasets[k] + ' ...')
                # load raw data
                f_raw = h5py.File(os.path.join(self.folder_name, self.train_datasets[k]), 'r')
                data = f_raw['main'][:]
                f_raw.close()
                data = data[:self.train_split]
                self.dataset.append(data)

                # load labels
                f_label = h5py.File(os.path.join(self.folder_name, self.train_labels[k]), 'r')
                label = f_label['main'][:]
                f_label.close()
                label = label[:self.train_split]
                if self.if_dilate:
                    # label = genSegMalis(label, 1)
                    label = seg_widen_border(label, tsz_h=1)
                self.labels.append(label)


        # for k in range(len(self.train_datasets)):
        #     print('load ' + self.train_datasets[k] + ' ...')
        #     # load raw data
        #     f_raw = h5py.File(os.path.join(self.folder_name, self.train_datasets[k]), 'r')
        #     data = f_raw['main'][:]
        #     f_raw.close()
        #     data = data[:self.train_split]
        #     self.dataset.append(data)

        #     # load labels
        #     f_label = h5py.File(os.path.join(self.folder_name, self.train_labels[k]), 'r')
        #     label = f_label['main'][:]
        #     f_label.close()
        #     label = label[:self.train_split]
        #     if self.if_dilate:
        #         # label = genSegMalis(label, 1)
        #         label = seg_widen_border(label, tsz_h=1)
        #     self.labels.append(label)
        if self.if_skele is True:
            self.skeletons = []
            if cfg.DATA.dataset_name == 'ac4':
                print('load ' + 'AC4_skeleton.h5' + ' ...')
                f_skeleton = h5py.File(os.path.join(self.folder_name, 'AC4_skeleton.h5'), 'r')
                skeleton = f_skeleton['main'][:]
                f_skeleton.close()
                self.skeleton = skeleton[:self.train_split]
                self.skeletons.append(self.skeleton)
            elif cfg.DATA.dataset_name == 'cremi-A':
                print('load ' + 'cremiA_skeleton.h5' + ' ...')
                f_skeleton = h5py.File(os.path.join(self.folder_name, 'cremiA_skeleton.h5'), 'r')
                skeleton = f_skeleton['main'][:]
                f_skeleton.close()
                self.skeleton = skeleton[:self.train_split]
                self.skeletons.append(self.skeleton)
            elif cfg.DATA.dataset_name == 'cremi-B':
                print('load ' + 'cremiB_skeleton.h5' + ' ...')
                f_skeleton = h5py.File(os.path.join(self.folder_name, 'cremiB_skeleton.h5'), 'r')
                skeleton = f_skeleton['main'][:]
                f_skeleton.close()
                self.skeleton = skeleton[:self.train_split]
                self.skeletons.append(self.skeleton)
            elif cfg.DATA.dataset_name == 'cremi-C':
                print('load ' + 'cremiC_skeleton.h5' + ' ...')
                f_skeleton = h5py.File(os.path.join(self.folder_name, 'cremiC_skeleton.h5'), 'r')
                skeleton = f_skeleton['main'][:]
                f_skeleton.close()
                self.skeleton = skeleton[:self.train_split]
                self.skeletons.append(self.skeleton)
            elif cfg.DATA.dataset_name == 'wafer':
                self.train_skeletons = ['wafer25_skeleton.h5', 'wafer26_skeleton.h5', 'wafer26_2_skeleton.h5', 'wafer36_skeleton.h5']
                for k in range(len(self.train_skeletons)):
                    print('load ' + self.train_skeletons[k] + ' ...')
                    # load raw data
                    f_raw = h5py.File(os.path.join(self.folder_name, self.train_skeletons[k]), 'r')
                    data = f_raw['main'][:]
                    f_raw.close()
                    # data = data[:self.train_split]
                    self.skeletons.append(data)
            elif cfg.DATA.dataset_name == 'wafer4':
                self.train_skeletons = ['wafer4_skeleton.h5']
                for k in range(len(self.train_skeletons)):
                    print('load ' + self.train_skeletons[k] + ' ...')
                    # load raw data
                    f_raw = h5py.File(os.path.join(self.folder_name, self.train_skeletons[k]), 'r')
                    data = f_raw['main'][:]
                    f_raw.close()
                    data = data[:self.train_split]
                    self.skeletons.append(data)
            elif cfg.DATA.dataset_name == 'zebrafinch':
                # 对于zebrafinch，使用与gt文件对应的skeleton文件
                skeleton_pattern = 'zebrafinch_skeleton_*.h5'
                skeleton_files = sorted(glob.glob(os.path.join(self.folder_name, skeleton_pattern)))
                print(f'Found {len(skeleton_files)} zebrafinch skeleton files')
                for skeleton_file in skeleton_files:
                    print('load ' + os.path.basename(skeleton_file) + ' ...')
                    f_skeleton = h5py.File(skeleton_file, 'r')
                    skeleton = f_skeleton['main'][:]
                    f_skeleton.close()
                    # skeleton = skeleton[:self.train_split]  # zebrafinch不使用train_split
                    self.skeletons.append(skeleton)

        # if self.if_dt is True:
        #     self.dts = []
        #     if cfg.DATA.dataset_name == 'ac4':
        #         print('load ' + 'AC4_dt.h5' + ' ...')
        #         f_dt = h5py.File(os.path.join(self.folder_name, 'AC4_dt.h5'), 'r')
        #         dt = f_dt['main'][:]
        #         f_dt.close()
        #         dt = dt[:self.train_split]
        #         self.dts.append(dt)
        #     elif cfg.DATA.dataset_name == 'cremi-A':
        #         print('load ' + 'cremiA_dt.h5' + ' ...')
        #         f_dt = h5py.File(os.path.join(self.folder_name, 'cremiA_dt.h5'), 'r')
        #         dt = f_dt['main'][:]
        #         f_dt.close()
        #         dt = dt[:self.train_split]
        #         self.dts.append(dt)
        #     elif cfg.DATA.dataset_name == 'cremi-B':
        #         print('load ' + 'cremiB_dt.h5' + ' ...')
        #         f_dt = h5py.File(os.path.join(self.folder_name, 'cremiB_dt.h5'), 'r')
        #         dt = f_dt['main'][:]
        #         f_dt.close()
        #         dt = dt[:self.train_split]
        #         self.dts.append(dt)
        #     elif cfg.DATA.dataset_name == 'cremi-C':
        #         print('load ' + 'cremiC_dt.h5' + ' ...')
        #         f_dt = h5py.File(os.path.join(self.folder_name, 'cremiC_dt.h5'), 'r')
        #         dt = f_dt['main'][:]
        #         f_dt.close()
        #         dt = dt[:self.train_split]
        #         self.dts.append(dt)
        #     elif cfg.DATA.dataset_name == 'zebrafinch':
        #         # 对于zebrafinch，使用与gt文件对应的dt文件
        #         dt_pattern = 'zebrafinch_dt_*.h5'
        #         dt_files = sorted(glob.glob(os.path.join(self.folder_name, dt_pattern)))
        #         print(f'Found {len(dt_files)} zebrafinch dt files')
        #         for dt_file in dt_files:
        #             print('load ' + os.path.basename(dt_file) + ' ...')
        #             f_dt = h5py.File(dt_file, 'r')
        #             dt = f_dt['main'][:]
        #             f_dt.close()
        #             # dt = dt[:self.train_split]  # zebrafinch不使用train_split
        #             self.dts.append(dt)

        # padding when the shape(z) of raw data is smaller than the input of network
        numz_dataset = self.dataset[0].shape[0]
        if numz_dataset < self.crop_size[0]:
            padding_size_z_left = (self.crop_size[0] - numz_dataset) // 2
            if numz_dataset % 2 == 0:
                padding_size_z_right = padding_size_z_left
            else:
                padding_size_z_right = padding_size_z_left + 1
            for k in range(len(self.dataset)):
                self.dataset[k] = np.pad(self.dataset[k], ((padding_size_z_left, padding_size_z_right), \
                                                           (0, 0), \
                                                           (0, 0)), mode='reflect')
                self.labels[k] = np.pad(self.labels[k], ((padding_size_z_left, padding_size_z_right), \
                                                         (0, 0), \
                                                         (0, 0)), mode='reflect')

        # padding by 'reflect' mode for mala network
        if cfg.MODEL.model_type == 'mala':
            for k in range(len(self.dataset)):
                self.dataset[k] = np.pad(self.dataset[k], ((self.net_padding[0], self.net_padding[0]), \
                                                           (self.net_padding[1], self.net_padding[1]), \
                                                           (self.net_padding[2], self.net_padding[2])), mode='reflect')
                self.labels[k] = np.pad(self.labels[k], ((self.net_padding[0], self.net_padding[0]), \
                                                         (self.net_padding[1], self.net_padding[1]), \
                                                         (self.net_padding[2], self.net_padding[2])), mode='reflect')
                if self.skeletons is not None:
                    self.skeletons[k] = np.pad(self.skeletons[k], ((self.net_padding[0], self.net_padding[0]), \
                                                            (self.net_padding[1], self.net_padding[1]), \
                                                            (self.net_padding[2], self.net_padding[2])), mode='reflect')

        # the training dataset size
        self.raw_data_shape = list(self.dataset[0].shape)
        print('raw data shape: ', self.raw_data_shape)

        # padding for random rotation
        self.crop_from_origin = [0, 0, 0]
        self.padding = cfg.DATA.padding
        self.crop_from_origin[0] = self.crop_size[0]
        self.crop_from_origin[1] = self.crop_size[1] + 2 * self.padding
        self.crop_from_origin[2] = self.crop_size[2] + 2 * self.padding

        # augmentation initoalization
        self.augs_init()

    def __getitem__(self, index):
        # random select one dataset if contain many datasets
        k = random.randint(0, len(self.train_datasets) - 1)
        used_data = self.dataset[k]
        used_label = self.labels[k]

        if self.cfg.DATA.dataset_name == 'zebrafinch':
            self.raw_data_shape = list(used_data.shape)
            # 这里如果数据是150的 首先crop_from_origin是[18,200,200] raw_data_shape是150的话一减就变成负的了，所以需要调整crop_size进而影响crop_from_orgin,是的random_y不会为负，crop_size应减小到+2*padding仍小于raw_shape.

        random_z = random.randint(0, self.raw_data_shape[0] - self.crop_from_origin[0])
        random_y = random.randint(0, self.raw_data_shape[1] - self.crop_from_origin[1])
        random_x = random.randint(0, self.raw_data_shape[2] - self.crop_from_origin[2])
        imgs = used_data[random_z:random_z + self.crop_from_origin[0], \
               random_y:random_y + self.crop_from_origin[1], \
               random_x:random_x + self.crop_from_origin[2]].copy()
        lb = used_label[random_z:random_z + self.crop_from_origin[0], \
             random_y:random_y + self.crop_from_origin[1], \
             random_x:random_x + self.crop_from_origin[2]].copy()
        
        # if self.skeleton is not None:
        #     used_skeleton = self.skeleton
        if self.skeletons is not None:
            used_skeleton = self.skeletons[k]
            skele = used_skeleton[random_z:random_z + self.crop_from_origin[0], \
                    random_y:random_y + self.crop_from_origin[1], \
                    random_x:random_x + self.crop_from_origin[2]].copy()
            skele = center_crop(skele, det_shape=self.crop_size)

        # if self.dts is not None:
        #     used_dt = self.dts[k]
        #     dt = used_dt[random_z:random_z + self.crop_from_origin[0], \
        #             random_y:random_y + self.crop_from_origin[1], \
        #             random_x:random_x + self.crop_from_origin[2]].copy()
        #     dt = center_crop(dt, det_shape=self.crop_size)
            

        imgs = imgs.astype(np.float32) / 255.0
        data = {'image': imgs, 'label': lb}
        # p=0.5 for augmentation
        if np.random.rand() < 0.5:
            data = self.augs_mix(data)
        imgs = data['image']
        lb = data['label']
        imgs = center_crop(imgs, det_shape=self.crop_size)
        lb = center_crop(lb, det_shape=self.crop_size)

        # convert label to affinity
        if self.model_type == 'mala':
            lb = lb[self.net_padding[0]:-self.net_padding[0], \
                 self.net_padding[1]:-self.net_padding[1], \
                 self.net_padding[2]:-self.net_padding[2]]
            if self.skeletons is not None:
                skele = skele[self.net_padding[0]:-self.net_padding[0], \
                    self.net_padding[1]:-self.net_padding[1], \
                    self.net_padding[2]:-self.net_padding[2]]
        # lb = genSegMalis(lb, 1)
        # lb_affs = seg_to_affgraph(lb, mknhood3d(1), pad='replicate').astype(np.float32)
        if self.shift_channels is None:
            lb_affs = seg_to_aff(lb).astype(np.float32)
        else:
            lb_affs = gen_affs_mutex_3d(lb, shift=self.shift_channels,
                                        padding=True, background=True)
        # lb = lb.astype(np.uint64)
        # lb_affs = seg2affs(lb, offsets=[[-1, 0, 0], [0, -1, 0], [0, 0, -1]],
        # 				retain_mask=False,
        # 				ignore_label=0,
        # 				retain_segmentation=False,
        # 				segmentation_to_binary=False,
        # 				map_to_foreground=True,
        # 				learn_ignore_transitions=False)

        # generate weights map for affinity
        # weight_factor = np.sum(lb_affs) / np.size(lb_affs)
        # weight_factor = np.clip(weight_factor, 1e-3, 1)
        # weightmap = lb_affs * (1 - weight_factor) / weight_factor + (1 - lb_affs)
        weightmap = weight_binary_ratio(lb_affs)


        if self.if_cascaded is not None:
            if self.if_cascaded is True:
                imgs = imgs[np.newaxis, ...]
                imgs = np.ascontiguousarray(imgs, dtype=np.float32)
                lb_affs = np.ascontiguousarray(lb_affs, dtype=np.float32)
                weightmap = np.ascontiguousarray(weightmap, dtype=np.float32)

                return imgs, lb_affs, weightmap

        #===============================
        if self.if_skele is True and self.if_dt is False:
            # lb_skele = gen_skele_3d(lb)
            # lb_skele = lb_skele[np.newaxis, ...]
            lb_skele = skele[np.newaxis, ...]
            lb_skele = np.ascontiguousarray(lb_skele, dtype=np.float32)
            imgs = imgs[np.newaxis, ...]
            imgs = np.ascontiguousarray(imgs, dtype=np.float32)
            lb_affs = np.ascontiguousarray(lb_affs, dtype=np.float32)
            weightmap = np.ascontiguousarray(weightmap, dtype=np.float32)

            return imgs, lb_affs, weightmap, lb_skele
            
        elif self.if_skele is False and self.if_dt is True:
            lb_dt = gen_distanceTransform(lb)
            lb_dt = lb_dt[np.newaxis, ...]
            lb_dt = np.ascontiguousarray(lb_dt, dtype=np.float32)
            imgs = imgs[np.newaxis, ...]
            imgs = np.ascontiguousarray(imgs, dtype=np.float32)
            lb_affs = np.ascontiguousarray(lb_affs, dtype=np.float32)
            weightmap = np.ascontiguousarray(weightmap, dtype=np.float32)

            return imgs, lb_affs, weightmap, lb_dt

        elif self.if_skele is True and self.if_dt is True:
            # lb_skele = gen_skele_3d(lb)
            lb_dt = gen_distanceTransform(lb)
            # lb_skele = lb_skele[np.newaxis, ...]
            lb_skele = skele[np.newaxis, ...]
            lb_dt = lb_dt[np.newaxis, ...]
            imgs = imgs[np.newaxis, ...]
            lb_skele = np.ascontiguousarray(lb_skele, dtype=np.float32)
            lb_dt = np.ascontiguousarray(lb_dt, dtype=np.float32)
            imgs = np.ascontiguousarray(imgs, dtype=np.float32)
            lb_affs = np.ascontiguousarray(lb_affs, dtype=np.float32)
            weightmap = np.ascontiguousarray(weightmap, dtype=np.float32)

            return imgs, lb_affs, weightmap, lb_skele, lb_dt
        
        else:
            imgs = imgs[np.newaxis, ...]
            imgs = np.ascontiguousarray(imgs, dtype=np.float32)
            lb_affs = np.ascontiguousarray(lb_affs, dtype=np.float32)
            weightmap = np.ascontiguousarray(weightmap, dtype=np.float32)

            return imgs, lb_affs, weightmap
        #==================================================

        # Norm images
        # if self.if_norm_images:
        # 	imgs = (imgs - 0.5) / 0.5
        # extend dimension
        imgs = imgs[np.newaxis, ...]
        imgs = np.ascontiguousarray(imgs, dtype=np.float32)
        lb_affs = np.ascontiguousarray(lb_affs, dtype=np.float32)
        weightmap = np.ascontiguousarray(weightmap, dtype=np.float32)
        return imgs, lb_affs, weightmap

    def __len__(self):
        return int(sys.maxsize)

    def augs_init(self):
        # https://zudi-lin.github.io/pytorch_connectomics/build/html/notes/dataloading.html#data-augmentation
        self.aug_rotation = Rotate(p=0.5)
        self.aug_rescale = Rescale(p=0.5)
        self.aug_flip = Flip(p=1.0, do_ztrans=0)
        self.aug_elastic = Elastic(p=0.75, alpha=16, sigma=4.0)
        self.aug_grayscale = Grayscale(p=0.75)

    # TO DO
    def augs_single(self, data):
        random_id = np.random.randint(1, 5 + 1)
        if random_id == 1:
            data = self.aug_rotation(data)
        elif random_id == 2:
            data = self.aug_rescale(data)
        elif random_id == 3:
            data = self.aug_flip(data)
        elif random_id == 4:
            data = self.aug_elastic(data)
        elif random_id == 5:
            data = self.aug_grayscale(data)
        else:
            raise NotImplementedError
        return data

    def augs_mix(self, data):
        if self.if_filp_aug and random.random() > 0.5:
            data = self.aug_flip(data)
        if self.if_rotation_aug and random.random() > 0.5:
            data = self.aug_rotation(data)
        if self.if_scale_aug and random.random() > 0.5:
            data = self.aug_rescale(data)
        if self.if_elastic_aug and random.random() > 0.5:
            data = self.aug_elastic(data)
        if self.if_intensity_aug and random.random() > 0.5:
            data = self.aug_grayscale(data)
        return data


def collate_fn(batchs):
    out_input = []
    for batch in batchs:
        out_input.append(torch.from_numpy(batch['image']))

    out_input = torch.stack(out_input, 0)
    return {'image': out_input}


class Provider(object):
    def __init__(self, stage, cfg):
        # patch_size, batch_size, num_workers, is_cuda=True):
        self.stage = stage
        if self.stage == 'train':
            self.data = Train(cfg)
            self.batch_size = cfg.TRAIN.batch_size
            self.num_workers = cfg.TRAIN.num_workers
        elif self.stage == 'valid':
            # return valid(folder_name, kwargs['data_list'])
            pass
        else:
            raise AttributeError('Stage must be train/valid')
        self.is_cuda = cfg.TRAIN.if_cuda
        self.data_iter = None
        self.iteration = 0
        self.epoch = 1

    def __len__(self):
        return self.data.num_per_epoch

    def build(self):
        if self.stage == 'train':
            self.data_iter = iter(
                DataLoader(dataset=self.data, batch_size=self.batch_size, num_workers=self.num_workers,
                           shuffle=False, drop_last=False, pin_memory=True))
        else:
            self.data_iter = iter(DataLoader(dataset=self.data, batch_size=1, num_workers=0,
                                             shuffle=False, drop_last=False, pin_memory=True))

    def next(self):
        if self.data_iter is None:
            self.build()
        try:
            # batch = next(self.data_iter)
            batch = next(self.data_iter)
            self.iteration += 1
            if self.is_cuda:
                batch[0] = batch[0].cuda()
                batch[1] = batch[1].cuda()
                batch[2] = batch[2].cuda()
            return batch
        except StopIteration:
            self.epoch += 1
            self.build()
            self.iteration += 1
            batch = next(self.data_iter)
            if self.is_cuda:
                batch[0] = batch[0].cuda()
                batch[1] = batch[1].cuda()
                batch[2] = batch[2].cuda()
            return batch


def show(img3d):
    # only used for image with shape [18, 160, 160]
    num = img3d.shape[0]
    column = 5
    row = math.ceil(num / float(column))
    size = img3d.shape[1]
    img_all = np.zeros((size * row, size * column), dtype=np.uint8)
    for i in range(row):
        for j in range(column):
            index = i * column + j
            if index >= num:
                img = np.zeros_like(img3d[0], dtype=np.uint8)
            else:
                img = (img3d[index] * 255).astype(np.uint8)
            img_all[i * size:(i + 1) * size, j * size:(j + 1) * size] = img
    return img_all


if __name__ == '__main__':
    import yaml
    from attrdict import AttrDict
    from utils.show import show_one
    from utils.shift_channels import shift_func

    """"""
    seed = 555
    np.random.seed(seed)
    random.seed(seed)
    cfg_file = 'seg_3d_ac4_data80.yaml'
    with open('./config/' + cfg_file, 'r') as f:
        cfg = AttrDict(yaml.load(f))

    if cfg.DATA.shift_channels is not None:
        cfg.shift = shift_func(cfg.DATA.shift_channels)
    else:
        cfg.shift = None
    out_path = os.path.join('./', 'data_temp')
    if not os.path.exists(out_path):
        os.mkdir(out_path)
    data = Train(cfg)
    t = time.time()
    for i in range(0, 50):
        t1 = time.time()
        tmp_data, affs, weightmap = iter(data).__next__()
        print('single cost time: ', time.time() - t1)
        tmp_data = np.squeeze(tmp_data)
        if cfg.MODEL.model_type == 'mala':
            tmp_data = tmp_data[14:-14, 106:-106, 106:-106]
        affs_xy = affs[-1]
        weightmap_xy = weightmap[-1]

        img_data = show_one(tmp_data)
        img_affs = show_one(affs_xy)
        img_weight = show_one(weightmap_xy)
        im_cat = np.concatenate([img_data, img_affs, img_weight], axis=1)
        Image.fromarray(im_cat).save(os.path.join(out_path, str(i).zfill(4) + '.png'))
    print(time.time() - t)

# from lsd_extractor import LsdExtractor

# # 在data_provider_labeled.py中添加新类
# class TrainLSD(Train):
#     """专门用于LSD+亲和力多任务学习的数据提供器"""
    
#     def __init__(self, cfg):
#         super(TrainLSD, self).__init__(cfg)
        
#         # LSD相关配置
#         self.lsd_sigma = cfg.DATA.get('lsd_sigma', (80, 80, 80))
#         self.lsd_downsample = cfg.DATA.get('lsd_downsample', 2)
#         self.voxel_size = cfg.DATA.get('voxel_size', (8, 8, 8))
#         self.lsd_extractor = LsdExtractor(
#             self.lsd_sigma, 
#             mode="gaussian", 
#             downsample=self.lsd_downsample
#         )
        
#         # 强制禁用skeleton和dt
#         self.if_skele = False
#         self.if_dt = False
        
#         print("TrainLSD initialized with LSD parameters:")
#         print(f"  sigma: {self.lsd_sigma}")
#         print(f"  downsample: {self.lsd_downsample}")
#         print(f"  voxel_size: {self.voxel_size}")
    
#     def __getitem__(self, index):
#         # 获取基础数据（调用父类方法到计算lb_affs和weightmap）
#         k = random.randint(0, len(self.train_datasets) - 1)
#         used_data = self.dataset[k]
#         used_label = self.labels[k]
        
#         random_z = random.randint(0, self.raw_data_shape[0] - self.crop_from_origin[0])
#         random_y = random.randint(0, self.raw_data_shape[1] - self.crop_from_origin[1])
#         random_x = random.randint(0, self.raw_data_shape[2] - self.crop_from_origin[2])
#         imgs = used_data[random_z:random_z + self.crop_from_origin[0], 
#                random_y:random_y + self.crop_from_origin[1], 
#                random_x:random_x + self.crop_from_origin[2]].copy()
#         lb = used_label[random_z:random_z + self.crop_from_origin[0], 
#              random_y:random_y + self.crop_from_origin[1], 
#              random_x:random_x + self.crop_from_origin[2]].copy()
        
#         imgs = imgs.astype(np.float32) / 255.0
#         data = {'image': imgs, 'label': lb}
        
#         # 数据增强
#         if np.random.rand() < 0.5:
#             data = self.augs_mix(data)
#         imgs = data['image']
#         lb = data['label']
#         imgs = center_crop(imgs, det_shape=self.crop_size)
#         lb = center_crop(lb, det_shape=self.crop_size)
        
#         # 根据模型类型调整
#         if self.model_type == 'mala':
#             lb = lb[self.net_padding[0]:-self.net_padding[0], 
#                  self.net_padding[1]:-self.net_padding[1], 
#                  self.net_padding[2]:-self.net_padding[2]]
        
#         # 保存原始标签用于LSD计算
#         lb_for_lsd = lb.copy()
        
#         # 计算亲和力
#         if self.shift_channels is None:
#             lb_affs = seg_to_aff(lb).astype(np.float32)
#         else:
#             lb_affs = gen_affs_mutex_3d(lb, shift=self.shift_channels,
#                                         padding=True, background=True)
        
#         # 计算权重
#         weightmap = weight_binary_ratio(lb_affs)
        
#         # 计算LSD
#         try:
#             lb_lsds = self.lsd_extractor.get_descriptors(
#                 segmentation=lb_for_lsd,
#                 voxel_size=self.voxel_size
#             )
#             lb_lsds = np.ascontiguousarray(lb_lsds, dtype=np.float32)
#         except Exception as e:
#             print(f"Warning: Failed to compute LSD: {e}")
#             # 使用零填充作为备用方案
#             lb_lsds = np.zeros((10,) + lb_for_lsd.shape, dtype=np.float32)
        
#         # 准备返回数据
#         imgs = imgs[np.newaxis, ...]
#         imgs = np.ascontiguousarray(imgs, dtype=np.float32)
#         lb_affs = np.ascontiguousarray(lb_affs, dtype=np.float32)
#         weightmap = np.ascontiguousarray(weightmap, dtype=np.float32)
        
#         return imgs, lb_affs, weightmap, lb_lsds


# class ProviderLSD(Provider):
#     """专门用于LSD+亲和力多任务学习的Provider"""
    
#     def __init__(self, stage, cfg):
#         self.stage = stage
#         if self.stage == 'train':
#             self.data = TrainLSD(cfg)  # 使用TrainLSD而不是Train
#             self.batch_size = cfg.TRAIN.batch_size
#             self.num_workers = cfg.TRAIN.num_workers
#         elif self.stage == 'valid':
#             pass
#         else:
#             raise AttributeError('Stage must be train/valid')
#         self.is_cuda = cfg.TRAIN.if_cuda
#         self.data_iter = None
#         self.iteration = 0
#         self.epoch = 1
    
#     def next(self):
#         if self.data_iter is None:
#             self.build()
#         try:
#             batch = next(self.data_iter)
#             self.iteration += 1
#             if self.is_cuda:
#                 batch[0] = batch[0].cuda()  # imgs
#                 batch[1] = batch[1].cuda()  # lb_affs
#                 batch[2] = batch[2].cuda()  # weightmap
#                 batch[3] = batch[3].cuda()  # lb_lsds
#             return batch
#         except StopIteration:
#             self.epoch += 1
#             self.build()
#             self.iteration += 1
#             batch = next(self.data_iter)
#             if self.is_cuda:
#                 batch[0] = batch[0].cuda()
#                 batch[1] = batch[1].cuda()
#                 batch[2] = batch[2].cuda()
#                 batch[3] = batch[3].cuda()
#             return batch