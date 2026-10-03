# from __future__ import absolute_import
# from __future__ import print_function
# from __future__ import division

# import os
# import sys
# import yaml
# import time
# import cv2
# import h5py
# import random
# import logging
# import argparse
# import numpy as np
# from PIL import Image
# from attrdict import AttrDict
# from tensorboardX import SummaryWriter
# from collections import OrderedDict
# import multiprocessing as mp
# from sklearn.metrics import f1_score, average_precision_score, roc_auc_score

# import torch
# import torch.nn as nn
# import torch.optim as optim
# import torch.nn.functional as F

# from data_provider_labeled import Provider
# # from provider_valid import Provider_valid
# from provider_valid2 import Provider_valid
# from loss.loss import WeightedMSE, WeightedBCE
# from loss.loss import MSELoss, BCELoss
# from utils.show import show_affs, show_affs_whole
# # from unet3d_mala import UNet3D_MALA
# from unet3d_mala_multitask import UNet3D_MALA
# # from model_superhuman2 import UNet_PNI
# from segmamba import SegMamba
# from model_unetr import UNETR
# from model_superhuman2_multitask import UNet_PNI
# from tqdm import tqdm


import os
import cv2
import h5py
import yaml
import torch
import argparse
import numpy as np
from skimage import morphology
from attrdict import AttrDict
from collections import OrderedDict
import torch.nn as nn
import torch.nn.functional as F
import time
from tqdm import tqdm
from sklearn.metrics import f1_score, average_precision_score, roc_auc_score

from provider_valid import Provider_valid
# from provider_valid2 import Provider_valid
from loss.loss import BCELoss, WeightedBCE, MSELoss, WeightedMSE
# from unet3d_mala import UNet3D_MALA
from unet3d_mala_multitask import UNet3D_MALA
# from model_superhuman import UNet_PNI
# from model_superhuman2 import UNet_PNI_FT2 as UNet_PNI
# from model_superhuman2 import UNet_PNI
from model_superhuman2_multitask import UNet_PNI, UNet_PNI_MOE_Add
# from utils.malis_loss import malis_loss
# from segmamba import SegMamba
# from model_unetr import UNETR

from utils.show import draw_fragments_3d
from utils.fragment import watershed, elf_watershed
from utils.shift_channels import shift_func
# from utils.lmc import mc_baseline
from utils.fragment import watershed, randomlabel, relabel

import waterz
# from utils.lmc import mc_baseline
# import evaluate as ev
from skimage.metrics import adapted_rand_error as adapted_rand_ref
from skimage.metrics import variation_of_information as voi_ref


# from thop import profile
# from thop import clever_format
import warnings
warnings.filterwarnings("ignore")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument('-c', '--cfg', type=str, default='seg_3d_ac4_data80', help='path to config file')
    parser.add_argument('-c', '--cfg', type=str, required=True, help='config filename stem under config/')
    # parser.add_argument('-c', '--cfg', type=str, default='wafer', help='path to config file')
    parser.add_argument('-mn', '--model_name', type=str, required=True)
    parser.add_argument('-id', '--model_id', type=int, required=True)
    parser.add_argument('-m', '--mode', type=str, default=None)
    parser.add_argument('-ts', '--test_split', type=int, default=None)
    parser.add_argument('-sm', '--seg_mode', type=str, default='waterz')
    parser.add_argument('-f', '--fragment', type=str, default='mahotas')  # 'elf', 'mahotas'
    parser.add_argument('-pm', '--pixel_metric', action='store_true', default=False)
    parser.add_argument('-s', '--save', action='store_false', default=True)
    parser.add_argument('-sw', '--show', action='store_true', default=False)
    parser.add_argument('-malis', '--malis_loss', action='store_true', default=False)
    args = parser.parse_args()

    cfg_file = args.cfg + '.yaml'
    print('cfg_file: ' + cfg_file)

    with open(os.path.join(os.path.dirname(__file__), 'config', cfg_file), 'r') as f:
        # cfg = AttrDict(yaml.load(f))
        cfg = AttrDict(yaml.safe_load(f))

    if args.mode is None:
        args.mode = cfg.DATA.valid_dataset

    if cfg.DATA.shift_channels is None:
        assert cfg.MODEL.output_nc == 3, "output_nc must be 3"
        cfg.shift = None
    else:
        assert cfg.MODEL.output_nc == cfg.DATA.shift_channels, "output_nc must be equal to shift_channels"
        cfg.shift = shift_func(cfg.DATA.shift_channels)

    if args.model_name is not None:
        trained_model = args.model_name
    else:
        trained_model = cfg.TEST.model_name

    out_path = os.path.join('outputs', 'inference', trained_model, args.mode)
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    img_folder = 'affs_'+str(args.model_id)
    out_affs = os.path.join(out_path, img_folder)
    if not os.path.exists(out_affs):
        os.makedirs(out_affs)
    print('out_path: ' + out_affs)
    affs_img_path = os.path.join(out_affs, 'affs_img')
    seg_img_path = os.path.join(out_affs, 'seg_img')
    if not os.path.exists(affs_img_path):
        os.makedirs(affs_img_path)
    if not os.path.exists(seg_img_path):
        os.makedirs(seg_img_path)

    device = torch.device('cuda:0')
    # if cfg.MODEL.model_type == 'mala':
    #     print('load mala model!')
    #     # model = UNet3D_MALA(output_nc=cfg.MODEL.output_nc, if_sigmoid=cfg.MODEL.if_sigmoid, init_mode=cfg.MODEL.init_mode_mala).to(device)
    #     model = UNet3D_MALA(in_planes=cfg.MODEL.input_nc,
    #                         output_nc=cfg.MODEL.output_nc, 
    #                         if_sigmoid=cfg.MODEL.if_sigmoid,
    #                         init_mode=cfg.MODEL.init_mode_mala,
    #                         if_skele=cfg.MODEL.if_skele,
    #                         if_dt=cfg.MODEL.if_dt,
    #                         ).to(device)
    # else:
    #     print('load superhuman model!')
    #     model = UNet_PNI(in_planes=cfg.MODEL.input_nc,
    #                      out_planes=cfg.MODEL.output_nc,
    #                      filters=cfg.MODEL.filters,
    #                      upsample_mode=cfg.MODEL.upsample_mode,
    #                      decode_ratio=cfg.MODEL.decode_ratio,
    #                      merge_mode=cfg.MODEL.merge_mode,
    #                      pad_mode=cfg.MODEL.pad_mode,
    #                      bn_mode=cfg.MODEL.bn_mode,
    #                      relu_mode=cfg.MODEL.relu_mode,
    #                      init_mode=cfg.MODEL.init_mode,
    #                      if_skele=cfg.MODEL.if_skele,
    #                      if_dt=cfg.MODEL.if_dt,
    #                      ).to(device)
        
    if cfg.MODEL.model_type == 'mala':
        print('load mala model!')
        model = UNet3D_MALA(in_planes=cfg.MODEL.input_nc,
                            output_nc=cfg.MODEL.output_nc, 
                            if_sigmoid=cfg.MODEL.if_sigmoid,
                            init_mode=cfg.MODEL.init_mode_mala,
                            if_skele=cfg.MODEL.if_skele,
                            if_dt=cfg.MODEL.if_dt,
                            ).to(device)
    # elif cfg.MODEL.model_type == 'segmamba':
    #     print("load segmamba model!")
    #     model = SegMamba(in_chans=1,
    #                      out_chans=3, 
    #                     #  kernel_size=(1,3,3),
    #                     # args=args
    #                      ).to(device)
    #     # args.crop_size = cfg.MODEL.crop_size
    # elif cfg.MODEL.model_type == 'unetr':
    #     print("load UNETR model!")
    #     model = UNETR(
    #             in_channels=cfg.MODEL.input_nc,
    #             out_channels=cfg.MODEL.output_nc,
    #             # img_size=cfg.MODEL.unetr_size,
    #             # patch_size=cfg.MODEL.patch_size,
    #             feature_size=16,
    #             hidden_size=768,
    #             mlp_dim=2048,
    #             num_heads=8,
    #             pos_embed='perceptron',
    #             norm_name='instance',
    #             conv_block=True,
    #             res_block=True,
    #             # kernel_size=cfg.MODEL.kernel_size,
    #             kernel_size=[1,3,3],
    #             skip_connection=False,
    #             show_feature=False,
    #             dropout_rate=0.1).to(device)  #model_unetr.py的UNETR
    else:
        print('load superhuman model!')
        model = UNet_PNI(in_planes=cfg.MODEL.input_nc,
                         out_planes=cfg.MODEL.output_nc,
                         filters=cfg.MODEL.filters,
                         upsample_mode=cfg.MODEL.upsample_mode,
                         decode_ratio=cfg.MODEL.decode_ratio,
                         merge_mode=cfg.MODEL.merge_mode,
                         pad_mode=cfg.MODEL.pad_mode,
                         bn_mode=cfg.MODEL.bn_mode,
                         relu_mode=cfg.MODEL.relu_mode,
                         init_mode=cfg.MODEL.init_mode,
                         if_skele=cfg.MODEL.if_skele,
                         if_dt=cfg.MODEL.if_dt,
                         ).to(device)
        pruned_checkpoint = torch.load(cfg.MODEL.pruned_model_path, map_location='cpu')
        model_second = UNet_PNI_MOE_Add(in_planes=1,
                                    out_planes=cfg.MODEL.output_nc,
                                    filters=pruned_checkpoint['filters'],
                                    upsample_mode=cfg.MODEL.upsample_mode,
                                    decode_ratio=cfg.MODEL.decode_ratio,
                                    merge_mode=cfg.MODEL.merge_mode,
                                    pad_mode=cfg.MODEL.pad_mode,
                                    bn_mode=cfg.MODEL.bn_mode,
                                    relu_mode=cfg.MODEL.relu_mode,
                                    init_mode=cfg.MODEL.init_mode,
                                    #  if_skele=cfg.MODEL.if_skele,
                                    #  if_dt=cfg.MODEL.if_dt,
                                    if_hard=cfg.MODEL.if_hard,
                                    if_residual=cfg.MODEL.if_residual,
                                    ).to(device)

    # ckpt_path = os.path.join('../models', trained_model, 'model-%06d.ckpt' % args.model_id)
    # checkpoint = torch.load(ckpt_path)

    checkpoint = torch.load(os.path.join(cfg.TRAIN.save_path, cfg.MODEL.trained_model_name,
                                         'model-%06d.ckpt' % cfg.MODEL.trained_model_id), map_location=device)


    new_state_dict = OrderedDict()
    state_dict = checkpoint['model_weights']
    for k, v in state_dict.items():
        # name = k[7:] # remove module.
        name = k[7:] if k.startswith('module.') else k
        new_state_dict[name] = v
    
    model.load_state_dict(new_state_dict)
    model = model.to(device)

    
    checkpoint = torch.load(os.path.join(cfg.TRAIN.save_path, trained_model,
                                         'model-%06d.ckpt' % args.model_id), map_location=device)


    new_state_dict = OrderedDict()
    state_dict = checkpoint['model_weights']
    for k, v in state_dict.items():
        # name = k[7:] # remove module.
        name = k[7:] if k.startswith('module.') else k
        new_state_dict[name] = v
    
    model_second.load_state_dict(new_state_dict)
    model_second = model_second.to(device)



    # n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # print('number of params (M): %.2f' % (n_parameters / 1.e6))

    valid_provider = Provider_valid(cfg, valid_data=args.mode, test_split=args.test_split)
    val_loader = torch.utils.data.DataLoader(valid_provider, batch_size=1)

    if cfg.TRAIN.loss_func == 'MSE':
        criterion = MSELoss()
    elif cfg.TRAIN.loss_func == 'WeightedBCELoss':
        criterion = WeightedBCE()
    elif cfg.TRAIN.loss_func == 'BCELoss':
        criterion = BCELoss()
    elif cfg.TRAIN.loss_func == 'WeightedMSELoss':
        criterion = WeightedMSE()
    else:
        raise AttributeError("NO this criterion")

    model.eval()
    model_second.eval()
    loss_all = []
    f_txt = open(os.path.join(out_affs, 'scores.txt'), 'w')
    print('the number of sub-volume:', len(valid_provider))
    losses_valid = []
    t1 = time.time()
    pbar = tqdm(total=len(valid_provider))
    for k, data in enumerate(val_loader, 0):
        inputs, target, weightmap = data
        inputs = inputs.cuda()
        target = target.cuda()
        weightmap = weightmap.cuda()
        with torch.no_grad():
            pred = model(inputs)
            if cfg.MODEL.if_skele is True and cfg.MODEL.if_dt is False:
                pred = pred[:, 1:]
            if cfg.MODEL.if_skele is False and cfg.MODEL.if_dt is True:
                pred = pred[:, 1:]
            if cfg.MODEL.if_skele is True and cfg.MODEL.if_dt is True:
                # pred = pred[:, 2:]
                pred_second = torch.cat((pred[:, :2], inputs), dim=1)
                if cfg.MODEL.if_residual:
                    pred = model_second(pred_second, pred[:, 2:])
                else:
                    pred = model_second(pred_second)
            # macs, params = profile(model, inputs=(inputs,))
            # macs, params = clever_format([macs, params], "%.3f")
            # print(f"thop: MACs: {macs}")
            # print(f"thop: Parameters: {params}") 
        tmp_loss = criterion(pred, target, weightmap)
        losses_valid.append(tmp_loss.item())
        valid_provider.add_vol(np.squeeze(pred.data.cpu().numpy()))
        pbar.update(1)
    pbar.close()
    cost_time = time.time() - t1
    print('Inference time=%.6f' % cost_time)
    f_txt.write('Inference time=%.6f' % cost_time)
    f_txt.write('\n')
    epoch_loss = sum(losses_valid) / len(losses_valid)
    output_affs = valid_provider.get_results()
    gt_affs = valid_provider.get_gt_affs()
    gt_seg = valid_provider.get_gt_lb()
    valid_provider.reset_output()
    gt_seg = gt_seg.astype(np.uint32)


    # save
    if args.save:
        print('save affs...')
        f = h5py.File(os.path.join(out_affs, 'affs.hdf'), 'w')
        f.create_dataset('main', data=output_affs, dtype=np.float32, compression='gzip')
        f.close()

    # segmentation
    print('Segmentation...')
    fragments = watershed(output_affs, 'maxima_distance')
    sf = 'OneMinus<HistogramQuantileAffinity<RegionGraphType, 50, ScoreValue, 256>>'
    # sf = 'OneMinus<EdgeStatisticValue<RegionGraphType, MeanAffinityProvider<RegionGraphType, ScoreValue>>>'
    segmentation = list(waterz.agglomerate(output_affs, [0.50],
                                        fragments=fragments,
                                        scoring_function=sf,
                                        discretize_queue=256))[0]

    segmentation = relabel(segmentation).astype(np.uint64)
    arand = adapted_rand_ref(gt_seg, segmentation, ignore_labels=(0))[0]
    voi_split, voi_merge = voi_ref(gt_seg, segmentation, ignore_labels=(0))
    voi_sum = voi_split + voi_merge
    print('model-%d, VOI-split=%.6f, VOI-merge=%.6f, VOI-sum=%.6f, ARAND=%.6f' %
        (args.model_id, voi_split, voi_merge, voi_sum, arand))
    f_txt.write('model-%d, VOI-split=%.6f, VOI-merge=%.6f, VOI-sum=%.6f, ARAND=%.6f' %
        (args.model_id, voi_split, voi_merge, voi_sum, arand))
    f_txt.write('\n')
    f = h5py.File(os.path.join(out_affs, 'seg.hdf'), 'w')
    f.create_dataset('main', data=segmentation, dtype=segmentation.dtype, compression='gzip')
    f.close()

    # segmentation = mc_baseline(output_affs)
    # segmentation = relabel(segmentation).astype(np.uint64)
    # print('the max id = %d' % np.max(segmentation))
    # f = h5py.File(os.path.join(out_affs, 'seg_lmc.hdf'), 'w')
    # f.create_dataset('main', data=segmentation, dtype=segmentation.dtype, compression='gzip')
    # f.close()

    # arand = adapted_rand_ref(gt_seg, segmentation, ignore_labels=(0))[0]
    # voi_split, voi_merge = voi_ref(gt_seg, segmentation, ignore_labels=(0))
    # voi_sum = voi_split + voi_merge
    # print('LMC: voi_split=%.6f, voi_merge=%.6f, voi_sum=%.6f, arand=%.6f' % \
    #     (voi_split, voi_merge, voi_sum, arand))
    # f_txt.write('LMC: voi_split=%.6f, voi_merge=%.6f, voi_sum=%.6f, arand=%.6f' % \
    #     (voi_split, voi_merge, voi_sum, arand))
    # f_txt.write('\n')




    # compute MSE
    if args.pixel_metric:
        print('MSE...')
        output_affs_prop = output_affs.copy()
        whole_mse = np.sum(np.square(output_affs - gt_affs)) / np.size(gt_affs)
        print('BCE...')
        output_affs = np.clip(output_affs, 0.000001, 0.999999)
        bce = -(gt_affs * np.log(output_affs) + (1 - gt_affs) * np.log(1 - output_affs))
        whole_bce = np.sum(bce) / np.size(gt_affs)
        output_affs[output_affs <= 0.5] = 0
        output_affs[output_affs > 0.5] = 1
        print('F1...')
        whole_arand = 1 - f1_score(gt_affs.astype(np.uint8).flatten(), output_affs.astype(np.uint8).flatten())
        # whole_arand = 0.0
        # new
        print('F1 boundary...')
        whole_arand_bound = f1_score(1 - gt_affs.astype(np.uint8).flatten(), 1 - output_affs.astype(np.uint8).flatten())
        # whole_arand_bound = 0.0
        print('mAP...')
        # whole_map = average_precision_score(1 - gt_affs.astype(np.uint8).flatten(), 1 - output_affs_prop.flatten())
        whole_map = 0.0
        print('AUC...')
        # whole_auc = roc_auc_score(1 - gt_affs.astype(np.uint8).flatten(), 1 - output_affs_prop.flatten())
        whole_auc = 0.0
        ###################################################
        if args.malis_loss:
            # from utils.malis_loss import malis_loss
            # print('Malis...')
            # t1 = time.time()
            # try:
            #     malis = malis_loss(output_affs_prop, gt_affs, gt_seg)
            # except:
            #     malis = 0.0
            # print('COST TIME: ' + str(time.time() - t1))
            malis = 0.0
        else:
            malis = 0.0
        ###################################################
        print('model-%d, valid-loss=%.6f, MSE-loss=%.6f, BCE-loss=%.6f, ARAND-loss=%.6f, F1-bound=%.6f, mAP=%.6f, auc=%.6f, malis-loss=%.6f' % \
            (args.model_id, epoch_loss, whole_mse, whole_bce, whole_arand, whole_arand_bound, whole_map, whole_auc, malis))
        f_txt.write('model-%d, valid-loss=%.6f, MSE-loss=%.6f, BCE-loss=%.6f, ARAND-loss=%.6f, F1-bound=%.6f, mAP=%.6f, auc=%.6f, malis-loss=%.6f' % \
                    (args.model_id, epoch_loss, whole_mse, whole_bce, whole_arand, whole_arand_bound, whole_map, whole_auc, malis))
        f_txt.write('\n')
    else:
        output_affs_prop = output_affs
    f_txt.close()

    # show
    if args.show:
        print('show affs...')
        output_affs_prop = (output_affs_prop * 255).astype(np.uint8)
        gt_affs = (gt_affs * 255).astype(np.uint8)
        for i in range(output_affs_prop.shape[1]):
            cat1 = np.concatenate([output_affs_prop[0,i], output_affs_prop[1,i], output_affs_prop[2,i]], axis=1)
            cat2 = np.concatenate([gt_affs[0,i], gt_affs[1,i], gt_affs[2,i]], axis=1)
            im_cat = np.concatenate([cat1, cat2], axis=0)
            cv2.imwrite(os.path.join(affs_img_path, str(i).zfill(4)+'.png'), im_cat)
        
        print('show seg...')
        # segmentation[gt_seg==0] = 0
        color_seg = draw_fragments_3d(segmentation)
        color_gt = draw_fragments_3d(gt_seg)
        for i in range(color_seg.shape[0]):
            im_cat = np.concatenate([color_seg[i], color_gt[i]], axis=1)
            cv2.imwrite(os.path.join(seg_img_path, str(i).zfill(4)+'.png'), im_cat)
    print('Done')
    print('Done')
