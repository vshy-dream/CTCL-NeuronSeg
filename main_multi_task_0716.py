from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import os
import sys
import yaml
import time
import cv2
import h5py
import random
import logging
import argparse
import numpy as np
from PIL import Image
from attrdict import AttrDict
from tensorboardX import SummaryWriter
from collections import OrderedDict
import multiprocessing as mp
from sklearn.metrics import f1_score, average_precision_score, roc_auc_score

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
# import torch.utils.data

# from data_provider_labeled import Provider
from data_provider_labeled_hcl import Provider
# from provider_valid import Provider_valid
from provider_valid2 import Provider_valid
from loss.loss import WeightedMSE, WeightedBCE, PixelwiseWeightedBCE, UncertaintyWeights, PixelwiseUncertaintyWeights, HierarchicalCalibrationLoss
from loss.loss import MSELoss, BCELoss
from utils.show import show_affs, show_affs_whole
# from unet3d_mala import UNet3D_MALA
# from model_superhuman2 import UNet_PNI
from unet3d_mala_multitask import UNet3D_MALA
from model_superhuman2_multitask import UNet_PNI
from utils.utils import setup_seed, execute
from utils.shift_channels import shift_func

import waterz
from utils.fragment import watershed, randomlabel
# import evaluate as ev
from skimage.metrics import adapted_rand_error as adapted_rand_ref
from skimage.metrics import variation_of_information as voi_ref
import warnings

from typing import Any
# from thop import profile
# from thop import clever_format
warnings.filterwarnings("ignore")

# cfg: Any  # Treat cfg as dynamically-typed to avoid attribute errors in static analysis.

def init_project(cfg):
    def init_logging(path):
        logging.basicConfig(
            level=logging.INFO,
            format='%(message)s',
            datefmt='%m-%d %H:%M',
            filename=path,
            filemode='w')

        # define a Handler which writes INFO messages or higher to the sys.stderr
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)

        # set a format which is simpler for console use
        formatter = logging.Formatter('%(message)s')
        # tell the handler to use this format
        console.setFormatter(formatter)
        logging.getLogger('').addHandler(console)

    # seeds
    setup_seed(cfg.TRAIN.random_seed)
    if cfg.TRAIN.if_cuda:
        if torch.cuda.is_available() is False:
            raise AttributeError('No GPU available')

    prefix = cfg.time
    if cfg.TRAIN.resume:
        model_name = cfg.TRAIN.model_name
    else:
        model_name = prefix + '_' + cfg.NAME
    cfg.cache_path = os.path.join(cfg.TRAIN.cache_path, model_name)
    cfg.save_path = os.path.join(cfg.TRAIN.save_path, model_name)
    # cfg.record_path = os.path.join(cfg.TRAIN.record_path, 'log')
    cfg.record_path = os.path.join(cfg.save_path, model_name)
    cfg.valid_path = os.path.join(cfg.save_path, 'valid')
    if cfg.TRAIN.resume is False:
        if not os.path.exists(cfg.cache_path):
            os.makedirs(cfg.cache_path)
        if not os.path.exists(cfg.save_path):
            os.makedirs(cfg.save_path)
        if not os.path.exists(cfg.record_path):
            os.makedirs(cfg.record_path)
        if not os.path.exists(cfg.valid_path):
            os.makedirs(cfg.valid_path)
    init_logging(os.path.join(cfg.record_path, prefix + '.log'))
    logging.info(cfg)
    writer = SummaryWriter(cfg.record_path)
    writer.add_text('cfg', str(cfg))
    return writer


def load_dataset(cfg):
    print('Caching datasets ... ', end='', flush=True)
    t1 = time.time()
    train_provider = Provider('train', cfg)
    valid_provider = Provider_valid(cfg)
    print('Done (time: %.2fs)' % (time.time() - t1))
    return train_provider, valid_provider


def build_model(cfg, writer):
    print('Building model on ', end='', flush=True)
    t1 = time.time()
    device = torch.device('cuda:0')
    if cfg.MODEL.model_type == 'mala':
        print('load mala model!')
        # model = UNet3D_MALA(output_nc=cfg.MODEL.output_nc, if_sigmoid=cfg.MODEL.if_sigmoid,
        #                     init_mode=cfg.MODEL.init_mode_mala).to(device)
        model = UNet3D_MALA(in_planes=cfg.MODEL.input_nc,
                            output_nc=cfg.MODEL.output_nc, 
                            if_sigmoid=cfg.MODEL.if_sigmoid,
                            init_mode=cfg.MODEL.init_mode_mala,
                            if_skele=cfg.MODEL.if_skele,
                            if_dt=cfg.MODEL.if_dt,
                            ).to(device)
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
                         if_uncertainty=cfg.MODEL.get('if_uncertainty', False),
                         ).to(device)

    if cfg.MODEL.pre_train:
        print('Load pre-trained model ...')
        ckpt_path = os.path.join(cfg.TRAIN.save_path, \
                                 cfg.MODEL.trained_model_name, \
                                 'model-%06d.ckpt' % cfg.MODEL.trained_model_id)
        checkpoint = torch.load(ckpt_path)
        pretrained_dict = checkpoint['model_weights']
        if cfg.MODEL.trained_gpus > 1:
            pretained_model_dict = OrderedDict()
            for k, v in pretrained_dict.items():
                name = k[7:]  # remove module.
                # name = k
                pretained_model_dict[name] = v
        else:
            pretained_model_dict = pretrained_dict

        from utils.encoder_dict import ENCODER_DICT2, ENCODER_DECODER_DICT2
        model_dict = model.state_dict()
        encoder_dict = OrderedDict()
        if cfg.MODEL.if_skip == 'True':
            print('Load the parameters of encoder and decoder!')
            encoder_dict = {k: v for k, v in pretained_model_dict.items() if k.split('.')[0] in ENCODER_DECODER_DICT2}
        else:
            print('Load the parameters of encoder!')
            encoder_dict = {k: v for k, v in pretained_model_dict.items() if k.split('.')[0] in ENCODER_DICT2}
        model_dict.update(encoder_dict)
        model.load_state_dict(model_dict)

    cuda_count = torch.cuda.device_count()
    # cuda_count = 2
    if cuda_count > 1:
        if cfg.TRAIN.batch_size % cuda_count == 0:
            print('%d GPUs ... ' % cuda_count, end='', flush=True)
            model = nn.DataParallel(model)
        else:
            raise AttributeError(
                'Batch size (%d) cannot be equally divided by GPU number (%d)' % (cfg.TRAIN.batch_size, cuda_count))
    else:
        print('a single GPU ... ', end='', flush=True)
    print('Done (time: %.2fs)' % (time.time() - t1))
    return model

def build_lossWeight_model(cfg, writer):
    print('Building model on ', end='', flush=True)
    t1 = time.time()
    device = torch.device('cuda:0')
    # model = UncertaintyWeights(if_skele=cfg.MODEL.if_skele, if_dt=cfg.MODEL.if_dt).to(device)
    # print('load lossweight model!')

    # model = PixelwiseUncertaintyWeights(if_skele=cfg.MODEL.if_skele, if_dt=cfg.MODEL.if_dt).to(device)
    # print('load pixelwise uncertrainty lossweight model!')

    model = HierarchicalCalibrationLoss(if_skele=cfg.MODEL.if_skele, if_dt=cfg.MODEL.if_dt).to(device)
    print('load Hierarchical Calibration Loss model!')

    cuda_count = torch.cuda.device_count()
    # cuda_count = 2
    if cuda_count > 1:
        if cfg.TRAIN.batch_size % cuda_count == 0:
            print('%d GPUs ... ' % cuda_count, end='', flush=True)
            model = nn.DataParallel(model)
        else:
            raise AttributeError(
                'Batch size (%d) cannot be equally divided by GPU number (%d)' % (cfg.TRAIN.batch_size, cuda_count))
    else:
        print('a single GPU ... ', end='', flush=True)
    print('Done (time: %.2fs)' % (time.time() - t1))
    return model


def resume_params(cfg, model, optimizer, resume):
    if resume:
        t1 = time.time()
        model_path = os.path.join(cfg.save_path, 'model-%06d.ckpt' % cfg.TRAIN.model_id)

        print('Resuming weights from %s ... ' % model_path, end='', flush=True)
        if os.path.isfile(model_path):
            checkpoint = torch.load(model_path)
            model.load_state_dict(checkpoint['model_weights'])
            # optimizer.load_state_dict(checkpoint['optimizer_weights'])
        else:
            raise AttributeError('No checkpoint found at %s' % model_path)
        print('Done (time: %.2fs)' % (time.time() - t1))
        print('valid %d' % checkpoint['current_iter'])
        return model, optimizer, checkpoint['current_iter']
    else:
        return model, optimizer, 0


def calculate_lr(iters):
    if iters < cfg.TRAIN.warmup_iters:
        current_lr = (cfg.TRAIN.base_lr - cfg.TRAIN.end_lr) * pow(float(iters) / cfg.TRAIN.warmup_iters,
                                                                  cfg.TRAIN.power) + cfg.TRAIN.end_lr
    else:
        if iters < cfg.TRAIN.decay_iters:
            current_lr = (cfg.TRAIN.base_lr - cfg.TRAIN.end_lr) * pow(
                1 - float(iters - cfg.TRAIN.warmup_iters) / cfg.TRAIN.decay_iters, cfg.TRAIN.power) + cfg.TRAIN.end_lr
        else:
            current_lr = cfg.TRAIN.end_lr
    return current_lr


def loop(cfg, train_provider, valid_provider, model, loss_weight_model, criterion, optimizer, iters, writer):
    f_loss_txt = open(os.path.join(cfg.record_path, 'loss.txt'), 'a')
    f_valid_txt = open(os.path.join(cfg.record_path, 'valid.txt'), 'a')
    rcd_time = []
    sum_time = 0
    sum_loss = 0
    device = torch.device('cuda:0')

    if cfg.TRAIN.loss_func == 'MSELoss':
        criterion = MSELoss()
    elif cfg.TRAIN.loss_func == 'BCELoss':
        criterion = BCELoss()
    elif cfg.TRAIN.loss_func == 'WeightedBCELoss':
        # criterion = WeightedBCE()
        # criterion = PixelwiseWeightedBCE()
        if cfg.MODEL.get('if_uncertainty', False):
            criterion = PixelwiseWeightedBCE()
        else:
            criterion = WeightedBCE()
    elif cfg.TRAIN.loss_func == 'WeightedMSELoss':
        criterion = WeightedMSE()
    else:
        raise AttributeError("NO this criterion")

    if cfg.MODEL.if_dt == True:
        if cfg.TRAIN.dt_loss_func == 'WeightedMSELoss':
            criterion_dt = WeightedMSE()
        elif cfg.TRAIN.dt_loss_func == 'WeightedBCELoss':
            criterion_dt = WeightedBCE()
        else:
            raise AttributeError("NO this criterion")
        
    if cfg.MODEL.if_skele == True:
        if cfg.TRAIN.skele_loss_func == 'WeightedMSELoss':
            criterion_skele = WeightedMSE()
        elif cfg.TRAIN.skele_loss_func == 'WeightedBCELoss':
            criterion_skele = WeightedBCE()
        else:
            raise AttributeError("NO this criterion")

    while iters < cfg.TRAIN.total_iters:
        # train
        model.train()
        iters += 1
        t1 = time.time()
        # inputs, target, weightmap = train_provider.next()


        #=============================
        if cfg.MODEL.if_skele is True and cfg.MODEL.if_dt is False:
            inputs, target, weightmap, skeletons = train_provider.next()
            inputs = inputs.to(device, non_blocking=True, dtype=torch.float32)
            target = target.to(device, non_blocking=True, dtype=torch.float32)
            weightmap = weightmap.to(device, non_blocking=True, dtype=torch.float32)
            skeletons = skeletons.to(device, non_blocking=True, dtype=torch.float32)
        elif cfg.MODEL.if_skele is False and cfg.MODEL.if_dt is True:
            inputs, target, weightmap, distanceTransform = train_provider.next()
            inputs = inputs.to(device, non_blocking=True, dtype=torch.float32)
            target = target.to(device, non_blocking=True, dtype=torch.float32)
            weightmap = weightmap.to(device, non_blocking=True, dtype=torch.float32)
            distanceTransform = distanceTransform.to(device, non_blocking=True, dtype=torch.float32)
        elif cfg.MODEL.if_skele is True and cfg.MODEL.if_dt is True:
            inputs, target, weightmap, skeletons, distanceTransform = train_provider.next()
            inputs = inputs.to(device, non_blocking=True, dtype=torch.float32)
            target = target.to(device, non_blocking=True, dtype=torch.float32)
            weightmap = weightmap.to(device, non_blocking=True, dtype=torch.float32)
            skeletons = skeletons.to(device, non_blocking=True, dtype=torch.float32)
            distanceTransform = distanceTransform.to(device, non_blocking=True, dtype=torch.float32)
        else:
            inputs, target, weightmap = train_provider.next()
            inputs = inputs.to(device, non_blocking=True, dtype=torch.float32)
            target = target.to(device, non_blocking=True, dtype=torch.float32)
            weightmap = weightmap.to(device, non_blocking=True, dtype=torch.float32)

        #=============================

        # decay learning rate
        if cfg.TRAIN.end_lr == cfg.TRAIN.base_lr:
            current_lr = cfg.TRAIN.base_lr
        else:
            current_lr = calculate_lr(iters)
            for param_group in optimizer.param_groups:
                param_group['lr'] = current_lr

        optimizer.zero_grad()

        # 模型前向传播
        if cfg.MODEL.get('if_uncertainty', False):
            pred, uncertainties = model(inputs)
        else:
            pred = model(inputs)
            uncertainties = None

        if cfg.MODEL.if_skele is True and cfg.MODEL.if_dt is False:
            out_skele = pred[:, :1]
            pred = pred[:, 1:]
            loss_skele = criterion_skele(out_skele, skeletons)
            loss_aff = criterion(pred, target, weightmap)
            loss = loss_skele + loss_aff
        if cfg.MODEL.if_skele is False and cfg.MODEL.if_dt is True:
            out_dt = pred[:, :1]
            pred = pred[:, 1:]
            loss_dt = criterion_dt(out_dt, distanceTransform)
            loss_aff = criterion(pred, target, weightmap)
            loss = loss_dt + loss_aff
        if cfg.MODEL.if_skele is True and cfg.MODEL.if_dt is True:
            ##### uncertaintyloss    UncertaintyWeights
            # out_skele = pred[:, :1]
            # out_dt = pred[:, 1:2]
            # pred = pred[:, 2:]
            # loss_skele = criterion_skele(out_skele, skeletons)
            # loss_dt = criterion_dt(out_dt, distanceTransform)
            # loss_aff = criterion(pred, target, weightmap)
            # # loss = loss_dt + loss_skele + loss_aff
            # losses = [loss_aff, loss_skele, loss_dt]
            # loss = loss_weight_model(losses)


            ##### Pixelwise UncertaintyLoss  PixelwiseUncertaintyWeights  
            # pred_aff = pred[:, 2:]
            # # 计算pixel-wise损失（不进行reduction）
            # criterion_no_reduction = nn.BCELoss(reduction='none')  # 或其他适当的损失函数
            # loss_skele = criterion_no_reduction(out_skele, skeletons)
            # loss_dt = criterion_no_reduction(out_dt, distanceTransform)
            # # loss_aff = criterion_no_reduction(pred_aff, target)
            # loss_aff = criterion(pred_aff, target, weightmap)
            
            # if cfg.MODEL.get('if_uncertainty', False) and uncertainties is not None:
            #     # 使用pixel-wise不确定性加权
            #     losses = [loss_aff, loss_skele, loss_dt]
            #     loss = loss_weight_model(losses, uncertainties)
            # else:
            #     # 原有的全局权重方法
            #     losses = [torch.mean(loss_aff), torch.mean(loss_skele), torch.mean(loss_dt)]
            #     loss = loss_weight_model(losses)


            ##### model = HierarchicalCalibrationLoss(if_skele=cfg.MODEL.if_skele, if_dt=cfg.MODEL.if_dt).to(device)
            out_skele = pred[:, :1]
            out_dt = pred[:, 1:2]
            pred_aff = pred[:, 2:]
            # 计算各个任务的基础损失
            loss_skele = criterion_skele(out_skele, skeletons)
            loss_dt = criterion_dt(out_dt, distanceTransform)
            loss_aff = criterion(pred_aff, target, weightmap)
            
            # 使用 HCL 损失函数
            loss = loss_weight_model(loss_aff, loss_skele, loss_dt)


        if cfg.MODEL.if_skele is False and cfg.MODEL.if_dt is False:
            loss = criterion(pred, target, weightmap)
        
        # ##############################
        # # LOSS
        loss.backward()
        # ##############################

        # 应用weight decay
        if cfg.TRAIN.weight_decay is not None:
            for group in optimizer.param_groups:
                for param in group['params']:
                    param.data = param.data.add(-cfg.TRAIN.weight_decay * group['lr'], param.data)
        optimizer.step()

        # ##############################

        sum_loss += loss.item()
        sum_time += time.time() - t1

        # log train
        if iters % cfg.TRAIN.display_freq == 0 or iters == 1:
            rcd_time.append(sum_time)
            if iters == 1:
                logging.info('step %d, loss = %.6f (wt: *1, lr: %.8f, et: %.2f sec, rd: %.2f min)'
                             % (iters, sum_loss * 1, current_lr, sum_time,
                                (cfg.TRAIN.total_iters - iters) / cfg.TRAIN.display_freq * np.mean(
                                    np.asarray(rcd_time)) / 60))
                writer.add_scalar('loss', sum_loss * 1, iters)
            else:
                logging.info('step %d, loss = %.6f (wt: *1, lr: %.8f, et: %.2f sec, rd: %.2f min)'
                             % (iters, sum_loss / cfg.TRAIN.display_freq * 1, current_lr, sum_time,
                                (cfg.TRAIN.total_iters - iters) / cfg.TRAIN.display_freq * np.mean(
                                    np.asarray(rcd_time)) / 60))
                writer.add_scalar('loss', sum_loss / cfg.TRAIN.display_freq * 1, iters)



            if cfg.MODEL.if_skele or cfg.MODEL.if_dt:
                if hasattr(loss_weight_model, 'module'):
                    hcl_info = loss_weight_model.module.get_parameters_info()
                else:
                    hcl_info = loss_weight_model.get_parameters_info()
                
                # 记录到 tensorboard
                writer.add_scalar('hcl/sigma', hcl_info['sigma'], iters)
                writer.add_scalar('hcl/sigma_squared', hcl_info['sigma_squared'], iters)
                
                if 'lambda_skel' in hcl_info:
                    writer.add_scalar('hcl/lambda_skel', hcl_info['lambda_skel'], iters)
                if 'lambda_dt' in hcl_info:
                    writer.add_scalar('hcl/lambda_dt', hcl_info['lambda_dt'], iters)
                
                # 添加到日志输出
                hcl_params_str = f", σ={hcl_info['sigma']:.4f}"
                if 'lambda_skel' in hcl_info:
                    hcl_params_str += f", λ_skel={hcl_info['lambda_skel']:.4f}"
                if 'lambda_dt' in hcl_info:
                    hcl_params_str += f", λ_dt={hcl_info['lambda_dt']:.4f}"
                
                logging.info('step %d, loss = %.6f (wt: *1, lr: %.8f, et: %.2f sec, rd: %.2f min%s)'
                            % (iters, sum_loss / cfg.TRAIN.display_freq * 1, current_lr, sum_time,
                                (cfg.TRAIN.total_iters - iters) / cfg.TRAIN.display_freq * np.mean(
                                    np.asarray(rcd_time)) / 60, hcl_params_str))
                


            f_loss_txt.write('step = ' + str(iters) + ', loss = ' + str(sum_loss / cfg.TRAIN.display_freq * 1))
            f_loss_txt.write('\n')
            f_loss_txt.flush()
            sys.stdout.flush()
            sum_time = 0
            sum_loss = 0



        # display
        if iters % cfg.TRAIN.valid_freq == 0 or iters == 1:
            show_affs(iters, inputs, pred[:, :3], target[:, :3], cfg.cache_path, model_type=cfg.MODEL.model_type)

        # valid
        if cfg.TRAIN.if_valid:
            # if iters % cfg.TRAIN.save_freq == 0 or iters == 1:
            if iters % cfg.TRAIN.save_freq == 0 and iters >= 999:
                device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
                model.eval()
                dataloader = torch.utils.data.DataLoader(valid_provider, batch_size=1, num_workers=0,
                                                         shuffle=False, drop_last=False, pin_memory=True)
                losses_valid = []

                # 计算volume的数量和每个volume的大小
                num_volumes = len(valid_provider.dataset)
                volume_size = valid_provider.num_per_dataset

                all_volume_preds = [[] for _ in range(num_volumes)]
                all_volume_losses = [[] for _ in range(num_volumes)]

                # 添加进度条
                from tqdm import tqdm
                pbar = tqdm(total=len(valid_provider), desc="Validating")

                for k, batch in enumerate(dataloader, 0):
                    inputs, target, weightmap = batch
                    inputs = inputs.to(device, non_blocking=True, dtype=torch.float32)
                    target = target.to(device, non_blocking=True, dtype=torch.float32)
                    weightmap = weightmap.to(device, non_blocking=True, dtype=torch.float32)

                    volume_idx = k // volume_size
                    slice_idx = k % volume_size

                    with torch.no_grad():
                        if cfg.MODEL.get('if_uncertainty', False):
                            pred, uncertainties = model(inputs)
                        else:
                            pred = model(inputs)
                            uncertainties = None
                        # pred = model(inputs)
                        if cfg.MODEL.if_skele is True and cfg.MODEL.if_dt is False:
                            pred = pred[:, 1:]
                        if cfg.MODEL.if_skele is False and cfg.MODEL.if_dt is True:
                            pred = pred[:, 1:]
                        if cfg.MODEL.if_skele is True and cfg.MODEL.if_dt is True:
                            pred = pred[:, 2:]
                        # macs, params = profile(model, inputs=(inputs,))
                        # macs, params = clever_format([macs, params], "%.3f")
                        # print(f"model - MACs: {macs}")
                        # print(f"model - Parameters: {params}")
                    tmp_loss = criterion(pred, target, weightmap)
                    losses_valid.append(tmp_loss.mean().item())

                    pred_np = np.squeeze(pred.data.cpu().numpy())
                    all_volume_preds[volume_idx].append(pred_np)
                    all_volume_losses[volume_idx].append(tmp_loss.item())

                    # 将预测结果添加到volume中进行重新组合（所有数据集都需要）
                    valid_provider.add_vol(pred_np)

                    pbar.update(1)

                    # 检查是否完成了一个volume的处理
                    if (k + 1) % volume_size == 0 or k == len(dataloader) - 1:
                        pbar.set_description(f"Processed volume {volume_idx}")

                pbar.close()

                # 为每个volume计算指标
                all_metrics = {
                    'loss': [],
                    'mse': [],
                    'bce': [],
                    'f1': [],
                    'voi_split': [],
                    'voi_merge': [],
                    'voi_sum': [],
                    'arand': [],
                }

                # 为详细的volume级结果创建日志文件
                f_valid_volumes_txt = open(os.path.join(cfg.record_path, f'valid_volumes_{iters}.txt'), 'w')

                for vol_idx in range(len(all_volume_preds)):
                    if len(all_volume_preds[vol_idx]) == 0:
                        print(f"Warning: Volume {vol_idx} has no predictions.")
                        continue

                    # 计算当前volume的平均损失
                    avg_volume_loss = sum(all_volume_losses[vol_idx]) / len(all_volume_losses[vol_idx])
                    all_metrics['loss'].append(avg_volume_loss)

                    # 获取重新组合后的完整预测结果（所有数据集都需要）
                    full_pred = valid_provider.get_results()

                    # 获取当前volume的ground truth来确定正确的切片范围
                    gt_affs_temp = valid_provider.get_gt_affs(vol_idx)
                    gt_shape = gt_affs_temp.shape

                    # 计算当前volume在完整结果中的起始位置
                    # 假设volumes是沿着Z维度连续排列的
                    vol_depth = gt_shape[1]  # Z维度大小
                    vol_start_z = vol_idx * vol_depth
                    vol_end_z = (vol_idx + 1) * vol_depth

                    # 切片提取当前volume的预测结果
                    volume_pred = full_pred[:, vol_start_z:vol_end_z, :, :].copy()
                    volume_pred = volume_pred.astype(np.float32)

                    # 获取真实标签
                    gt_seg = valid_provider.get_gt_lb(vol_idx)
                    gt_affs = valid_provider.get_gt_affs(vol_idx).copy()

                    # 计算MSE
                    mse = np.sum(np.square(volume_pred - gt_affs)) / np.size(gt_affs)
                    all_metrics['mse'].append(mse)

                    # 计算BCE
                    volume_pred_clipped = np.clip(volume_pred, 0.000001, 0.999999)
                    bce = -(gt_affs * np.log(volume_pred_clipped) + (1 - gt_affs) * np.log(1 - volume_pred_clipped))
                    bce_mean = np.sum(bce) / np.size(gt_affs)
                    all_metrics['bce'].append(bce_mean)

                    # 计算F1分数
                    volume_pred_binary = volume_pred.copy()
                    volume_pred_binary[volume_pred_binary <= 0.5] = 0
                    volume_pred_binary[volume_pred_binary > 0.5] = 1
                    f1 = f1_score(1 - gt_affs.astype(np.uint8).flatten(), 1 - volume_pred_binary.astype(np.uint8).flatten())
                    all_metrics['f1'].append(f1)

                    # 保存可视化结果
                    save_path_vol = os.path.join(cfg.valid_path, f'volume_{vol_idx}')
                    os.makedirs(save_path_vol, exist_ok=True)
                    show_affs_whole(iters, volume_pred, gt_affs, save_path_vol)

                    # 分割评估
                    if cfg.TRAIN.if_seg:
                        ##############
                        # segmentation
                        if iters > 1:
                            fragments = watershed(volume_pred, 'maxima_distance')
                            sf = 'OneMinus<HistogramQuantileAffinity<RegionGraphType, 50, ScoreValue, 256>>'
                            segmentation = list(waterz.agglomerate(volume_pred, [0.50],
                                                                   fragments=fragments,
                                                                   scoring_function=sf,
                                                                   discretize_queue=256))[0]
                            arand = adapted_rand_ref(gt_seg, segmentation, ignore_labels=(0))[0]
                            voi_split, voi_merge = voi_ref(gt_seg, segmentation, ignore_labels=(0))
                            voi_sum = voi_split + voi_merge
                        else:
                            voi_merge = 0.0
                            voi_split = 0.0
                            voi_sum = 0.0
                            arand = 0.0
                            print('model-%d, segmentation failed!' % iters)
                    else:
                        voi_merge = 0.0
                        voi_split = 0.0
                        voi_sum = 0.0
                        arand = 0.0

                    all_metrics['voi_split'].append(voi_split)
                    all_metrics['voi_merge'].append(voi_merge)
                    all_metrics['voi_sum'].append(voi_sum)
                    all_metrics['arand'].append(arand)

                    # 为下一个volume重置输出（所有数据集都需要）
                    valid_provider.reset_output()

                    # 写入volume级日志
                    f_valid_volumes_txt.write(
                        f'Volume {vol_idx}: loss={avg_volume_loss:.6f}, MSE={mse:.6f}, BCE={bce_mean:.6f}, F1={f1:.6f}, VOI-split={voi_split:.6f}, VOI-merge={voi_merge:.6f}, VOI-sum={voi_sum:.6f}, ARAND={arand:.6f}\n'
                    )

                f_valid_volumes_txt.close()

                # 计算总体平均指标
                epoch_loss = sum(all_metrics['loss']) / len(all_metrics['loss'])
                whole_mse = sum(all_metrics['mse']) / len(all_metrics['mse'])
                whole_bce = sum(all_metrics['bce']) / len(all_metrics['bce'])
                whole_f1 = sum(all_metrics['f1']) / len(all_metrics['f1'])
                voi_split = sum(all_metrics['voi_split']) / len(all_metrics['voi_split'])
                voi_merge = sum(all_metrics['voi_merge']) / len(all_metrics['voi_merge'])
                voi_sum = sum(all_metrics['voi_sum']) / len(all_metrics['voi_sum'])
                arand = sum(all_metrics['arand']) / len(all_metrics['arand'])

                print(
                    f'model-{iters}, valid-loss={epoch_loss:.6f}, MSE-loss={whole_mse:.6f}, BCE-loss={whole_bce:.6f}, F1-score={whole_f1:.6f}, VOI-split={voi_split:.6f}, VOI-merge={voi_merge:.6f}, VOI-sum={voi_sum:.6f}, ARAND={arand:.6f}',
                    flush=True)

                writer.add_scalar('valid/epoch_loss', epoch_loss, iters)
                writer.add_scalar('valid/mse_loss', whole_mse, iters)
                writer.add_scalar('valid/bce_loss', whole_bce, iters)
                writer.add_scalar('valid/f1_score', whole_f1, iters)
                writer.add_scalar('valid/voi_split', voi_split, iters)
                writer.add_scalar('valid/voi_merge', voi_merge, iters)
                writer.add_scalar('valid/voi_sum', voi_sum, iters)
                writer.add_scalar('valid/arand', arand, iters)

                f_valid_txt.write(
                    f'model-{iters}, valid-loss={epoch_loss:.6f}, MSE-loss={whole_mse:.6f}, BCE-loss={whole_bce:.6f}, F1-score={whole_f1:.6f}, VOI-split={voi_split:.6f}, VOI-merge={voi_merge:.6f}, VOI-sum={voi_sum:.6f}, ARAND={arand:.6f}'
                )
                f_valid_txt.write('\n')
                f_valid_txt.flush()
                torch.cuda.empty_cache()

                ##############
                # segmentation
                if cfg.TRAIN.if_seg:
                    if iters > 1:
                        fragments = watershed(out_affs, 'maxima_distance')
                        sf = 'OneMinus<HistogramQuantileAffinity<RegionGraphType, 50, ScoreValue, 256>>'
                        segmentation = list(waterz.agglomerate(out_affs, [0.50],
                                                               fragments=fragments,
                                                               scoring_function=sf,
                                                               discretize_queue=256))[0]
                        arand = adapted_rand_ref(gt_seg, segmentation, ignore_labels=(0))[0]
                        voi_split, voi_merge = voi_ref(gt_seg, segmentation, ignore_labels=(0))
                        voi_sum = voi_split + voi_merge
                    else:
                        voi_merge = 0.0
                        voi_split = 0.0
                        voi_sum = 0.0
                        arand = 0.0
                        print('model-%d, segmentation failed!' % iters)
                else:
                    voi_merge = 0.0
                    voi_split = 0.0
                    voi_sum = 0.0
                    arand = 0.0
                ##############

                # MSE
                whole_mse = np.sum(np.square(out_affs - gt_affs)) / np.size(gt_affs)
                out_affs = np.clip(out_affs, 0.000001, 0.999999)
                bce = -(gt_affs * np.log(out_affs) + (1 - gt_affs) * np.log(1 - out_affs))
                whole_bce = np.sum(bce) / np.size(gt_affs)
                out_affs[out_affs <= 0.5] = 0
                out_affs[out_affs > 0.5] = 1
                # whole_f1 = 1 - f1_score(gt_affs.astype(np.uint8).flatten(), out_affs.astype(np.uint8).flatten())
                whole_f1 = f1_score(1 - gt_affs.astype(np.uint8).flatten(), 1 - out_affs.astype(np.uint8).flatten())
                print(
                    'model-%d, valid-loss=%.6f, MSE-loss=%.6f, BCE-loss=%.6f, F1-score=%.6f, VOI-split=%.6f, VOI-merge=%.6f, VOI-sum=%.6f, ARAND=%.6f' % \
                    (iters, epoch_loss, whole_mse, whole_bce, whole_f1, voi_split, voi_merge, voi_sum, arand), flush=True)
                writer.add_scalar('valid/epoch_loss', epoch_loss, iters)
                writer.add_scalar('valid/mse_loss', whole_mse, iters)
                writer.add_scalar('valid/bce_loss', whole_bce, iters)
                writer.add_scalar('valid/f1_score', whole_f1, iters)
                writer.add_scalar('valid/voi_split', voi_split, iters)
                writer.add_scalar('valid/voi_merge', voi_merge, iters)
                writer.add_scalar('valid/voi_sum', voi_sum, iters)
                writer.add_scalar('valid/arand', arand, iters)
                f_valid_txt.write(
                    'model-%d, valid-loss=%.6f, MSE-loss=%.6f, BCE-loss=%.6f, F1-score=%.6f, VOI-split=%.6f, VOI-merge=%.6f, VOI-sum=%.6f, ARAND=%.6f' % \
                    (iters, epoch_loss, whole_mse, whole_bce, whole_f1, voi_split, voi_merge, voi_sum, arand))
                f_valid_txt.write('\n')
                f_valid_txt.flush()
                torch.cuda.empty_cache()

        # save
        if iters % cfg.TRAIN.save_freq == 0:
            states = {'current_iter': iters, 'valid_result': None,
                      'model_weights': model.state_dict()}
            states_loss_weight_model = {'current_iter': iters, 'valid_result': None,
                      'model_weights': loss_weight_model.state_dict(),
                      'hcl_parameters': loss_weight_model.get_parameters_info()}
            torch.save(states, os.path.join(cfg.save_path, 'model-%06d.ckpt' % iters))
            torch.save(states_loss_weight_model, os.path.join(cfg.save_path, 'loss_weight_model-%06d.ckpt' % iters))
            print('***************save modol, iters = %d.***************' % (iters), flush=True)
    f_loss_txt.close()
    f_valid_txt.close()


if __name__ == "__main__":
    # mp.set_start_method('spawn')
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--cfg', type=str, default='seg_3d_ac34_b2_skele_dt_11_bce_mse_hcl', help='path to config file')
    # parser.add_argument('-c', '--cfg', type=str, default='seg_inpainting', help='path to config file')
    parser.add_argument('-m', '--mode', type=str, default='train', help='path to config file')
    parser.add_argument('-p', '--pretrain_path', type=str, default='', help='path to pretraining model')  # 20240513
    args = parser.parse_args()

    cfg_file = args.cfg + '.yaml'
    print('cfg_file: ' + cfg_file)
    print('mode: ' + args.mode)

    with open(os.path.join(os.path.dirname(__file__), 'config', cfg_file), 'r') as f:
        # cfg = AttrDict(yaml.load(f))
        cfg = AttrDict(yaml.safe_load(f))
    print(cfg)
    
    timeArray = time.localtime()
    time_stamp = time.strftime('%Y-%m-%d--%H-%M-%S', timeArray)
    print('time stamp:', time_stamp)

    cfg.path = cfg_file
    cfg.time = time_stamp
    if cfg.DATA.shift_channels is None:
        assert cfg.MODEL.output_nc == 3, "output_nc must be 3"  # output_nc意味着什么
        cfg.shift = None
    else:
        assert cfg.MODEL.output_nc == cfg.DATA.shift_channels, "output_nc must be equal to shift_channels"
        cfg.shift = shift_func(cfg.DATA.shift_channels)

    if args.mode == 'train':
        writer = init_project(cfg)
        train_provider, valid_provider = load_dataset(cfg)
        model = build_model(cfg, writer)
        loss_weight_model = build_lossWeight_model(cfg, writer)


        # if args.pretrain_path:
            # checkpoint = torch.load(args.pretrain_path, map_location='cpu')
            # for k in list(checkpoint['model'].keys()):
            #     if k.startswith('module.'):
            #         checkpoint['model'][k[7:]] = checkpoint['model'].pop(k)
            #     if k in model.state_dict() and checkpoint['model'][k].shape != model.state_dict()[k].shape:
            #         print(f"Removing key {k} from pretrained checkpoint")
            #         del checkpoint['model'][k]
            # model.load_state_dict(checkpoint['model'], strict=False)
            # print("Load pre-trained checkpoint from: %s" % args.pretrain_path)
        
        if args.pretrain_path:
            checkpoint = torch.load(args.pretrain_path, map_location='cpu')
            for k in list(checkpoint['model_weights'].keys()):
                if k.startswith('module.'):
                    checkpoint['model_weights'][k[7:]] = checkpoint['model_weights'].pop(k)
                if k in model.state_dict() and checkpoint['model_weights'][k].shape != model.state_dict()[k].shape:
                    print(f"Removing key {k} from pretrained checkpoint")
                    del checkpoint['model_weights'][k]
            model.load_state_dict(checkpoint['model_weights'], strict=False)
            print("Load pre-trained checkpoint from: %s" % args.pretrain_path)


        # optimizer = torch.optim.Adam(model.parameters(), lr=cfg.TRAIN.base_lr, betas=(0.9, 0.999),
                                    #  eps=0.01, weight_decay=1e-6, amsgrad=True)
                                    
        optimizer = torch.optim.Adam(list(model.parameters()) + list(loss_weight_model.parameters()), lr=cfg.TRAIN.base_lr, betas=(0.9, 0.999),
                                     eps=0.01, weight_decay=1e-6, amsgrad=True)
        # optimizer = torch.optim.Adam(model.parameters(), lr=cfg.TRAIN.base_lr, betas=(0.9, 0.999),
        #                              eps=0.01, weight_decay=1e-6, amsgrad=True)

        # optimizer = optim.Adam(model.parameters(), lr=cfg.TRAIN.base_lr, betas=(0.9, 0.999), eps=1e-8, amsgrad=False)
        # optimizer = optim.Adamax(model.parameters(), lr=cfg.TRAIN.base_l, eps=1e-8)
        model, optimizer, init_iters = resume_params(cfg, model, optimizer, cfg.TRAIN.resume)
        loop(cfg, train_provider, valid_provider, model, loss_weight_model, nn.L1Loss(), optimizer, init_iters, writer)
        writer.close()
    else:
        pass
    print('***Done***')
