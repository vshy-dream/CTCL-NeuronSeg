import os
import cv2
import h5py
import math
import random
import numpy as np
import sys
from PIL import Image
from torch.utils.data import Dataset
from utils.seg_util import mknhood3d, genSegMalis
from utils.aff_util import seg_to_affgraph
from data.data_segmentation import seg_widen_border, weight_binary_ratio
from data.data_affinity import seg_to_aff
# from utils.affinity_official import seg2affs
from utils.affinity_ours import gen_affs_mutex_3d

class Provider_valid(Dataset):
    def __init__(self, cfg, valid_data=None, num_z=18, test=False, test_split=None):
        # basic settings
        self.cfg = cfg
        self.model_type = cfg.MODEL.model_type
        self.if_dilate = cfg.DATA.if_dilate
        self.shift_channels = cfg.DATA.shift_channels
        self.output_nc = cfg.MODEL.output_nc
        # self.output_nc = 3 # valid for LSD-based affinity 
        # self.separate_weight = cfg.TRAIN.separate_weight
        self.test = test
        if valid_data is not None:
            valid_dataset_name = valid_data
        else:
            try:
                valid_dataset_name = cfg.DATA.valid_dataset
                print('valid on valid dataset!')
            except:
                valid_dataset_name = cfg.DATA.dataset_name
                print('valid on train dataset!')

        # basic settings
        # the input size of network
        if cfg.MODEL.model_type == 'superhuman':
            self.crop_size = [18, 160, 160]
            self.net_padding = [0, 0, 0]
            if valid_dataset_name == 'zebrafinch':
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
        # elif cfg.MODEL.model_type == 'UNETR':
        elif cfg.MODEL.model_type == 'unetr':
            self.crop_size = [32, 160, 160]
            self.net_padding = [0, 0, 0]
        elif cfg.MODEL.model_type == 'segmamba':
            self.crop_size = [32, 160, 160]
            self.net_padding = [0, 0, 0]
        elif cfg.MODEL.model_type == 'ResUNet':  # 100M
            self.crop_size = [24, 224, 224]
            self.net_padding = [0,0,0]
        elif cfg.MODEL.model_type == 'ResUNetWithAttention':  # 100M
            self.crop_size = [24, 224, 224]
            self.net_padding = [0,0,0]
        elif cfg.MODEL.model_type == 'SwinUNETR':  # 100M
            self.crop_size = [32, 160, 160]
            self.net_padding = [0,0,0]
        else:
            raise AttributeError('No this model type!')
        # self.num_z = self.crop_size[0] #num_z
        self.num_z = num_z #num_z only for lsd
        # the output size of network
        # for mala: [25, 56, 56]
        # for superhuman: [18, 160, 160]
        self.out_size = [self.crop_size[k] - 2 * self.net_padding[k] for k in range(len(self.crop_size))]

        # training dataset files (h5), may contain many datasets
        # if cfg.DATA.valid_dataset == 'cremi-A' or cfg.DATA.valid_dataset == 'cremi':
        #     self.sub_path = 'CREMI'
        #     self.train_datasets = ['cremiA_inputs_interp.h5']
        #     self.train_labels = ['cremiA_labels.h5']
        # elif cfg.DATA.valid_dataset == 'cremi-B':
        #     self.sub_path = 'CREMI'
        #     self.train_datasets = ['cremiB_inputs_interp.h5']
        #     self.train_labels = ['cremiB_labels.h5']
        # elif cfg.DATA.valid_dataset == 'cremi-C':
        #     self.sub_path = 'CREMI'
        #     self.train_datasets = ['cremiC_inputs_interp.h5']
        #     self.train_labels = ['cremiC_labels.h5']
        # elif cfg.DATA.valid_dataset == 'cremi-all':
        #     self.sub_path = 'CREMI'
        #     self.train_datasets = ['cremiA_inputs_interp.h5', 'cremiB_inputs_interp.h5', 'cremiC_inputs_interp.h5']
        #     self.train_labels = ['cremiA_labels.h5', 'cremiB_labels.h5', 'cremiC_labels.h5']
        # elif cfg.DATA.valid_dataset == 'isbi':
        #     self.sub_path = 'snemi3d'
        #     self.train_datasets = ['isbi_inputs.h5']
        #     self.train_labels = ['isbi_labels.h5']
        # elif cfg.DATA.valid_dataset == 'ac3':
        #     # self.sub_path = 'AC3_AC4/AC3'
        #     self.sub_path = 'ac3_ac4'
        #     self.train_datasets = ['AC3_inputs.h5']
        #     self.train_labels = ['AC3_labels.h5']
        # elif cfg.DATA.valid_dataset == 'ac4':
        #     # self.sub_path = 'AC3_AC4/AC4'
        #     self.sub_path = 'ac3_ac4'
        #     self.train_datasets = ['AC4_inputs.h5']
        #     self.train_labels = ['AC4_labels.h5']
        # elif cfg.DATA.valid_dataset == 'fib':
        #     self.sub_path = 'fib'
        #     self.train_datasets = ['fib1_inputs.h5']
        #     self.train_labels = ['fib1_labels.h5']
        # else:
        #     raise AttributeError('No this dataset type!')
        

        if valid_dataset_name == 'cremi-A':
            self.sub_path = 'cremi'
            self.train_datasets = ['cremiA_inputs.h5']
            self.train_labels = ['cremiA_labels.h5']
        elif valid_dataset_name == 'cremi-B':
            self.sub_path = 'cremi'
            self.train_datasets = ['cremiB_inputs.h5']
            self.train_labels = ['cremiB_labels.h5']
        elif valid_dataset_name == 'cremi-C':
            self.sub_path = 'cremi'
            self.train_datasets = ['cremiC_inputs.h5']
            self.train_labels = ['cremiC_labels.h5']
        elif valid_dataset_name == 'cremi-all':
            self.sub_path = 'cremi'
            self.train_datasets = ['cremiC_inputs_interp.h5']
            self.train_labels = ['cremiC_labels.h5']
            # self.sub_path = 'cremi'
            # self.train_datasets = ['cremiA_inputs_interp.h5', 'cremiB_inputs_interp.h5', 'cremiC_inputs_interp.h5']
            # self.train_labels = ['cremiA_labels.h5', 'cremiB_labels.h5', 'cremiC_labels.h5']
        elif valid_dataset_name == 'isbi':
            self.sub_path = 'snemi3d'
            self.train_datasets = ['isbi_inputs.h5']
            self.train_labels = ['isbi_labels.h5']
        elif valid_dataset_name == 'isbi_test':
            self.sub_path = 'snemi3d'
            self.train_datasets = ['isbi_test_inputs.h5']
            self.train_labels = ['isbi_test_labels.h5']
        elif valid_dataset_name == 'ac3':
            self.sub_path = 'ac3_ac4'
            self.train_datasets = ['AC3_inputs.h5']
            self.train_labels = ['AC3_labels.h5']
        elif valid_dataset_name == 'ac4':
            self.sub_path = 'ac3_ac4'
            self.train_datasets = ['AC4_inputs.h5']
            self.train_labels = ['AC4_labels.h5']
        elif valid_dataset_name == 'fib':
            self.sub_path = 'fib'
            self.train_datasets = ['fib_inputs.h5']
            self.train_labels = ['fib_labels.h5']
        elif valid_dataset_name == 'wafer':
            self.sub_path = 'wafer'
            # self.train_datasets = ['wafer4_raw.h5', 'wafer25_raw.h5']
            # self.train_labels = ['wafer4_seg.h5', 'wafer25_seg.h5']
            self.train_datasets = ['wafer4_inputs.h5']
            self.train_labels = ['wafer4_labels.h5']
            # self.train_datasets = ['wafer25_inputs.h5', 'wafer26_inputs.h5', 'wafer26_2_inputs.h5', 'wafer36_inputs.h5']
            # self.train_labels = ['wafer25_labels.h5', 'wafer26_labels.h5', 'wafer26_2_labels.h5', 'wafer36_labels.h5']
        elif valid_dataset_name == 'wafer36_2':
            self.sub_path = 'wafer'
            self.train_datasets = ['wafer36_2_inputs.h5']
            self.train_labels = ['wafer36_2_labels.h5']
        elif valid_dataset_name == 'wafer4':
            self.sub_path = 'wafer'
            self.train_datasets = ['wafer4_inputs.h5']
            self.train_labels = ['wafer4_labels.h5']
        elif valid_dataset_name == 'zebrafinch':
            self.sub_path = 'zebrafinch/val'
            # zebrafinch验证集有多个文件，使用glob模式匹配
            import glob
            image_pattern = os.path.join(cfg.DATA.data_folder, self.sub_path, 'image_*.h5')
            gt_pattern = os.path.join(cfg.DATA.data_folder, self.sub_path, 'gt_*.h5')
            self.train_datasets = [os.path.basename(f) for f in sorted(glob.glob(image_pattern))]
            self.train_labels = [os.path.basename(f) for f in sorted(glob.glob(gt_pattern))]
        else:
            raise AttributeError('No this dataset type!')


        # the path of datasets, need first-level and second-level directory, such as: os.path.join('../data', 'cremi')
        self.folder_name = os.path.join(cfg.DATA.data_folder, self.sub_path)
        assert len(self.train_datasets) == len(self.train_labels)

        # split validation data
        # if valid_dataset_name == 'snemi3d-ac4':
        #     self.test_split = 100
        # else:
        if test_split is None:
            self.test_split = cfg.DATA.test_split
            # 只有在没有显式传入 test_split 时，才应用默认覆盖规则
            if valid_dataset_name == 'isbi_test' or valid_dataset_name == 'ac3':
                self.test_split = 100
        else:
            # 如果显式传入了 test_split，使用传入的值（不覆盖）
            self.test_split = test_split
        print('the number of valid(test) = %d' % self.test_split)

        # load dataset
        self.dataset = []
        self.labels = []
        self.labels_origin = []
        for k in range(len(self.train_datasets)):
            print('load ' + self.train_datasets[k] + ' ...')
            # load raw data
            f_raw = h5py.File(os.path.join(self.folder_name, self.train_datasets[k]), 'r')
            data = f_raw['main'][:]
            f_raw.close()
            if valid_dataset_name == 'ac3':
                print('top data')
                data = data[:self.test_split]
            elif valid_dataset_name == 'wafer4':  # 20240511
                data = data[-self.test_split:]
                print('bottom 25 wafer4 data')
            elif valid_dataset_name == 'zebrafinch':
                # zebrafinch验证集每个文件都是独立的，不需要切片
                print('zebrafinch validation data - using full file')
                pass  # 使用完整文件
            else:
                data = data[-self.test_split:]
            self.dataset.append(data)

            # load labels
            f_label = h5py.File(os.path.join(self.folder_name, self.train_labels[k]), 'r')
            label = f_label['main'][:]
            f_label.close()
            if valid_dataset_name == 'ac3':
                print('top label')
                label = label[:self.test_split]
            elif valid_dataset_name == 'wafer4':  # 20240511
                label = label[-self.test_split:]
                print('bottom 25 wafer4 label')
            elif valid_dataset_name == 'zebrafinch':
                # zebrafinch验证集每个文件都是独立的，不需要切片
                print('zebrafinch validation label - using full file')
                pass  # 使用完整文件
            else:
                label = label[-self.test_split:]
            self.labels_origin.append(label.copy())
            if self.if_dilate:
                label = seg_widen_border(label, tsz_h=1)

                # if cfg.DATA.widen_way:
                #     label = seg_widen_border(label, tsz_h=1)
                # else:
                #     label = genSegMalis(label, 1)
            self.labels.append(label)
        self.origin_data_shape = list(self.dataset[0].shape)

        # generate gt affinity
        self.gt_affs = []
        for k in range(len(self.labels)):
            temp = self.labels[k]
            # self.gt_affs.append(seg_to_affgraph(temp, mknhood3d(1), pad='replicate').astype(np.float32))
            self.gt_affs.append(seg_to_aff(temp).astype(np.float32))

        # padding by 'reflect' mode for inference
        if cfg.MODEL.model_type == 'mala':
            self.stride = self.out_size           # [25, 56, 56]
            self.valid_padding = self.net_padding # [14, 106, 106]
            assert self.dataset[0].shape[0] % 25 == 0, "the shape of test data must be 25*"
            padding_z = self.dataset[0].shape[0] // 25
            if 'fib' in valid_dataset_name:
                padding_xy = 10
            else:
                # padding_xy = 19
                padding_xy = 23
            self.num_zyx = [padding_z, padding_xy, padding_xy]
        else:
            if 'fib' in valid_dataset_name:
                padding_xy = 20
                num_xy = 6
            elif 'cremi-A' in valid_dataset_name or 'cremi-B' in valid_dataset_name or 'cremi-C' in valid_dataset_name or 'wafer4' in valid_dataset_name or 'wafer' in valid_dataset_name:
            # elif 'wafer4' in valid_dataset_name or 'wafer' in valid_dataset_name:
                padding_xy = 15
                num_xy = 15
            elif valid_dataset_name == 'zebrafinch' or valid_dataset_name == 'zebrafinch_test' or valid_dataset_name == 'zebrafinch_test_128':
                padding_xy = 16
                if hasattr(cfg.DATA, 'raw_3d') and cfg.DATA.raw_3d is True:
                    num_xy = 6 # cropsize原始 18,128,128
                else:
                    num_xy = 8 # 修改的 self.crop_size = [64, 64, 64]
            else:
                padding_xy = 48
                num_xy = 13
            if self.dataset[0].shape[0] == 200:
                self.stride = [10, 80, 80]
                self.valid_padding = [4, padding_xy, padding_xy]
                self.num_zyx = [20, num_xy, num_xy]
            elif self.dataset[0].shape[0] == 125:
                self.stride = [10, 80, 80]
                self.valid_padding = [4, padding_xy, padding_xy]
                self.num_zyx = [13, num_xy, num_xy]
            elif self.dataset[0].shape[0] == 100:
                self.stride = [10, 80, 80]
                self.valid_padding = [4, padding_xy, padding_xy]
                self.num_zyx = [10, num_xy, num_xy]
            elif self.dataset[0].shape[0] == 50:
                self.stride = [10, 80, 80]
                self.valid_padding = [4, padding_xy, padding_xy]
                self.num_zyx = [5, num_xy, num_xy]
            elif self.dataset[0].shape[0] == 25:
                # for rapid inference
                self.stride = [15, 80, 80]
                self.valid_padding = [4, padding_xy, padding_xy]
                self.num_zyx = [2, num_xy, num_xy]
            elif self.dataset[0].shape[0] == 20:
                self.stride = [10, 80, 80]
                self.valid_padding = [4, padding_xy, padding_xy]
                self.num_zyx = [2, num_xy, num_xy]
            elif self.dataset[0].shape[0] == 150 or 128:  # 针对于zebrafinch的要重新计算，20250425
                self.valid_padding = [16, padding_xy, padding_xy]
                if hasattr(cfg.DATA, 'raw_3d') and cfg.DATA.raw_3d is True:
                    self.stride = [16, 32, 32]
                    self.num_zyx = [12, num_xy, num_xy] # cropsize原始 18,128,128  160太大裁不了
                else:
                    self.stride = [32, 32, 32]
                    self.num_zyx = [5, num_xy, num_xy] # 修改的cropsize
            else:
                raise NotImplementedError

        # only for superhuman and the num-z = 10
        if self.num_z < 18:
            raise NotImplementedError

        for k in range(len(self.dataset)):
            self.dataset[k] = np.pad(self.dataset[k], ((self.valid_padding[0], self.valid_padding[0]), \
                                                    (self.valid_padding[1], self.valid_padding[1]), \
                                                    (self.valid_padding[2], self.valid_padding[2])), mode='reflect')
            self.labels[k] = np.pad(self.labels[k], ((self.valid_padding[0], self.valid_padding[0]), \
                                                    (self.valid_padding[1], self.valid_padding[1]), \
                                                    (self.valid_padding[2], self.valid_padding[2])), mode='reflect')

        # the training dataset size
        self.raw_data_shape = list(self.dataset[0].shape)

        self.reset_output()
        self.weight_vol = self.get_weight()
        if self.num_z < 18:
            raise NotImplementedError

        # the number of inference times
        self.num_per_dataset = self.num_zyx[0] * self.num_zyx[1] * self.num_zyx[2]
        self.iters_num = self.num_per_dataset * len(self.dataset)

    def __getitem__(self, index):
        # print(index)
        pos_data = index // self.num_per_dataset
        pre_data = index % self.num_per_dataset
        pos_z = pre_data // (self.num_zyx[1] * self.num_zyx[2])
        pos_xy = pre_data % (self.num_zyx[1] * self.num_zyx[2])
        pos_x = pos_xy // self.num_zyx[2]
        pos_y = pos_xy % self.num_zyx[2]

        # find position
        fromz = pos_z * self.stride[0]
        endz = fromz + self.crop_size[0]
        if endz > self.raw_data_shape[0]:
            endz = self.raw_data_shape[0]
            fromz = endz - self.crop_size[0]
        fromy = pos_y * self.stride[1]
        endy = fromy + self.crop_size[1]
        if endy > self.raw_data_shape[1]:
            endy = self.raw_data_shape[1]
            fromy = endy - self.crop_size[1]
        fromx = pos_x * self.stride[2]
        endx = fromx + self.crop_size[2]
        if endx > self.raw_data_shape[2]:
            endx = self.raw_data_shape[2]
            fromx = endx - self.crop_size[2]
        self.pos = [fromz, fromy, fromx]

        imgs = self.dataset[pos_data][fromz:endz, fromx:endx, fromy:endy].copy()
        lb = self.labels[pos_data][fromz:endz, fromx:endx, fromy:endy].copy()

        if self.num_z < 18:
            raise NotImplementedError

        # convert label to affinity
        if self.model_type == 'mala':
            lb = lb[self.net_padding[0]:-self.net_padding[0], \
                    self.net_padding[1]:-self.net_padding[1], \
                    self.net_padding[2]:-self.net_padding[2]]
        # lb = genSegMalis(lb, 1)
        # lb_affs = seg_to_affgraph(lb, mknhood3d(1), pad='replicate').astype(np.float32)
        if self.shift_channels is None:
            if self.output_nc == 3:
                lb_affs = seg_to_aff(lb).astype(np.float32)
            # elif self.output_nc == 13:
            #     lb_affs = seg_to_aff(lb).astype(np.float32)
            elif self.output_nc == 12:
                nhood233 = np.asarray([-2, 0, 0, 0, -3, 0, 0, 0, -3]).reshape((3, 3))
                nhood399 = np.asarray([-3, 0, 0, 0, -9, 0, 0, 0, -9]).reshape((3, 3))
                nhood427 = np.asarray([-4, 0, 0, 0, -27, 0, 0, 0, -27]).reshape((3, 3))
                label111 = seg_to_aff(lb, pad='').astype(np.float32)
                label233 = seg_to_aff(lb, nhood233, pad='')
                label399 = seg_to_aff(lb, nhood399, pad='')
                label427 = seg_to_aff(lb, nhood427, pad='')
                lb_affs = np.concatenate((label111, label233, label399, label427), axis=0)
            else:
                raise NotImplementedError
        else:
            lb_affs = gen_affs_mutex_3d(lb, shift=self.shift_channels,
                                        padding=True, background=True)

        # generate weights map for affinity
        # weight_factor = np.sum(lb_affs) / np.size(lb_affs)
        # weight_factor = np.clip(weight_factor, 1e-3, 1)
        # weightmap = lb_affs * (1 - weight_factor) / weight_factor + (1 - lb_affs)
        # if self.separate_weight:
        #     weightmap = np.zeros_like(lb_affs)
        #     for i in range(lb_affs.shape[0]):
        #         weightmap[i] = weight_binary_ratio(lb_affs[i])
        # else:
        #     weightmap = weight_binary_ratio(lb_affs)
        weightmap = weight_binary_ratio(lb_affs)

        imgs = imgs.astype(np.float32) / 255.0
        imgs = imgs[np.newaxis, ...]
        imgs = np.ascontiguousarray(imgs, dtype=np.float32)
        lb_affs = np.ascontiguousarray(lb_affs, dtype=np.float32)
        weightmap = np.ascontiguousarray(weightmap, dtype=np.float32)
        return imgs, lb_affs, weightmap

    def __len__(self):
        return self.iters_num

    def reset_output(self, default_c=None):
        if default_c is None:
            if self.shift_channels is None:
                channel = self.output_nc
            else:
                channel = len(self.shift_channels)
        else:
            channel = default_c
        if self.model_type != 'mala':
            self.out_affs = np.zeros(tuple([channel]+self.raw_data_shape), dtype=np.float32)
            self.weight_map = np.zeros(tuple([1]+self.raw_data_shape), dtype=np.float32)
        else:
            self.out_affs = np.zeros(tuple([channel]+self.origin_data_shape), dtype=np.float32)
            self.weight_map = np.zeros(tuple([1]+self.origin_data_shape), dtype=np.float32)

    def get_weight(self, sigma=0.2, mu=0.0):
        if self.num_z < 18:
            zz, yy, xx = np.meshgrid(np.linspace(-1, 1, 18, dtype=np.float32),
                                    np.linspace(-1, 1, self.out_size[1], dtype=np.float32),
                                    np.linspace(-1, 1, self.out_size[2], dtype=np.float32), indexing='ij')
        else:
            zz, yy, xx = np.meshgrid(np.linspace(-1, 1, self.out_size[0], dtype=np.float32),
                                    np.linspace(-1, 1, self.out_size[1], dtype=np.float32),
                                    np.linspace(-1, 1, self.out_size[2], dtype=np.float32), indexing='ij')
        dd = np.sqrt(zz * zz + yy * yy + xx * xx)
        weight = 1e-6 + np.exp(-((dd - mu) ** 2 / (2.0 * sigma ** 2)))
        weight = weight[np.newaxis, ...]
        return weight

    def add_vol(self, affs_vol):
        fromz, fromy, fromx = self.pos
        if self.num_z < 18:
            raise NotImplementedError

        if self.model_type != 'mala':
            self.out_affs[:, fromz:fromz+self.out_size[0], \
                             fromx:fromx+self.out_size[1], \
                             fromy:fromy+self.out_size[2]] += affs_vol * self.weight_vol
            self.weight_map[:, fromz:fromz+self.out_size[0], \
                               fromx:fromx+self.out_size[1], \
                               fromy:fromy+self.out_size[2]] += self.weight_vol
        else:
            self.out_affs[:, fromz:fromz+self.out_size[0], \
                             fromx:fromx+self.out_size[1], \
                             fromy:fromy+self.out_size[2]] = affs_vol

    def get_results(self):
        if self.model_type != 'mala':
            self.out_affs = self.out_affs / self.weight_map
            if self.valid_padding[0] == 0:
                self.out_affs = self.out_affs[:, :, \
                                                self.valid_padding[1]:-self.valid_padding[1], \
                                                self.valid_padding[2]:-self.valid_padding[2]]
            else:
                self.out_affs = self.out_affs[:, self.valid_padding[0]:-self.valid_padding[0], \
                                                self.valid_padding[1]:-self.valid_padding[1], \
                                                self.valid_padding[2]:-self.valid_padding[2]]
        return self.out_affs

    def get_gt_affs(self, num_data=0):
        return self.gt_affs[num_data]

    def get_gt_lb(self, num_data=0):
        # lbs = self.labels_origin[num_data].copy()
        # lbs = lbs[self.valid_padding[0]:-self.valid_padding[0], \
        #           self.valid_padding[1]:-self.valid_padding[1], \
        #           self.valid_padding[2]:-self.valid_padding[2]]
        return self.labels_origin[num_data]

    def get_raw_data(self, num_data=0):
        out = self.dataset[num_data].copy()
        return out[self.valid_padding[0]:-self.valid_padding[0], \
                    self.valid_padding[1]:-self.valid_padding[1], \
                    self.valid_padding[2]:-self.valid_padding[2]]


