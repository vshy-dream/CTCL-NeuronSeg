# complete_fixed_structural_pruning_v3.py
import torch
import torch.nn as nn
import numpy as np
from collections import OrderedDict
import copy

class CompleteStructuralPruner:
    def __init__(self, model):
        self.model = model
        self.device = next(model.parameters()).device  # 获取模型所在的设备
        self.gradient_accumulator = {}
        self.importance_scores = {}
        self.layer_info = {}
        self.hooks = []
        
    def inspect_model_structure(self):
        """检查模型结构，获取实际的层名和信息"""
        print("检查模型结构...")
        print("-" * 80)
        
        conv_layers = []
        
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Conv3d):
                conv_layers.append(name)
                self.layer_info[name] = {
                    'in_channels': module.in_channels,
                    'out_channels': module.out_channels,
                    'kernel_size': module.kernel_size,
                    'stride': module.stride,
                    'padding': module.padding,
                    'requires_grad': module.weight.requires_grad if hasattr(module, 'weight') else False
                }
                print(f"Conv3d层: {name:<30} | "
                      f"输入: {module.in_channels:<3} | "
                      f"输出: {module.out_channels:<3} | "
                      f"需要梯度: {module.weight.requires_grad if hasattr(module, 'weight') else False}")
        
        print("-" * 80)
        print(f"总共找到 {len(conv_layers)} 个卷积层")
        print(f"模型设备: {self.device}")
        
        # 检查模型参数状态
        total_params = 0
        trainable_params = 0
        for param in self.model.parameters():
            total_params += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
        
        print(f"总参数: {total_params:,}, 可训练参数: {trainable_params:,}")
        
        return conv_layers
    
    def ensure_model_trainable(self):
        """确保模型可训练"""
        print("确保模型处于可训练状态...")
        
        # 设置为训练模式
        self.model.train()
        
        # 确保所有参数需要梯度
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                print(f"启用梯度: {name}")
                param.requires_grad = True
        
        # 检查BN层
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.BatchNorm3d, nn.BatchNorm2d, nn.BatchNorm1d)):
                module.train()
                if hasattr(module, 'track_running_stats'):
                    module.track_running_stats = True
    
    def collect_gradients_direct_method(self, train_provider, cfg, num_batches=5):
        """直接方法收集梯度 - 最可靠的方法"""
        print(f"使用直接方法收集梯度，batch数: {num_batches}")
        
        # 检查模型结构
        conv_layers = self.inspect_model_structure()
        
        if len(conv_layers) == 0:
            raise ValueError("没有找到任何卷积层！")
        
        # 确保模型可训练
        self.ensure_model_trainable()
        
        # 初始化梯度累积器
        for layer_name in conv_layers:
            self.gradient_accumulator[layer_name] = []
        
        # 使用直接的梯度收集方法
        criterion = nn.BCELoss()
        if cfg.MODEL.if_dt:
            criterion_dt = nn.MSELoss()
        if cfg.MODEL.if_skele:
            criterion_skele = nn.BCELoss()
        
        successful_batches = 0
        
        for batch_idx in range(num_batches):
            try:
                print(f"\n处理Batch {batch_idx + 1}/{num_batches}")
                
                # 获取数据
                if cfg.MODEL.if_skele and cfg.MODEL.if_dt:
                    inputs, target, weightmap, skeletons, distanceTransform = train_provider.next()
                    skeletons = skeletons.to(self.device)
                    distanceTransform = distanceTransform.to(self.device)
                elif cfg.MODEL.if_skele:
                    inputs, target, weightmap, skeletons = train_provider.next()
                    skeletons = skeletons.to(self.device)
                elif cfg.MODEL.if_dt:
                    inputs, target, weightmap, distanceTransform = train_provider.next()
                    distanceTransform = distanceTransform.to(self.device)
                else:
                    inputs, target, weightmap = train_provider.next()
                
                inputs = inputs.to(self.device)
                target = target.to(self.device)
                
                # 清零梯度
                self.model.zero_grad()
                
                # 前向传播
                pred = self.model(inputs)
                print(f"  模型输出形状: {pred.shape}")
                
                # 计算损失 - 关键修改：使用所有输出来确保梯度流
                if cfg.MODEL.if_skele and cfg.MODEL.if_dt:
                    # 三个任务的输出
                    # skele_pred = pred[:, :1]
                    # dt_pred = pred[:, 1:2] 
                    affinity_pred = pred[:, 2:]
                    
                    # 计算各自的损失
                    # loss_skele = criterion_skele(skele_pred, skeletons)
                    # loss_dt = criterion_dt(dt_pred, distanceTransform)
                    loss_affinity = criterion(affinity_pred, target)
                    
                    # 总损失 - 确保所有分支都有梯度
                    # total_loss = loss_affinity + 0.5 * loss_skele + 0.5 * loss_dt
                    total_loss = loss_affinity
                    
                    # print(f"  损失: affinity={loss_affinity:.6f}, skele={loss_skele:.6f}, dt={loss_dt:.6f}")
                    
                    print(f" 仅affinity 损失: affinity={loss_affinity:.6f}")
                    
                elif cfg.MODEL.if_skele:
                    skele_pred = pred[:, :1]
                    affinity_pred = pred[:, 1:]
                    
                    loss_skele = criterion_skele(skele_pred, skeletons)
                    loss_affinity = criterion(affinity_pred, target)
                    
                    total_loss = loss_affinity + 0.5 * loss_skele
                    
                elif cfg.MODEL.if_dt:
                    dt_pred = pred[:, :1]
                    affinity_pred = pred[:, 1:]
                    
                    loss_dt = criterion_dt(dt_pred, distanceTransform)
                    loss_affinity = criterion(affinity_pred, target)
                    
                    total_loss = loss_affinity + 0.5 * loss_dt
                    
                else:
                    total_loss = criterion(pred, target)
                
                print(f"  总损失: {total_loss:.6f}")
                
                # 反向传播
                total_loss.backward()
                
                # 直接收集每层的梯度
                gradient_collected = 0
                for name, module in self.model.named_modules():
                    if isinstance(module, nn.Conv3d):
                        if hasattr(module, 'weight') and module.weight.grad is not None:
                            weight_grad = module.weight.grad
                            # 计算每个输出通道的重要性
                            channel_importance = torch.norm(weight_grad.view(weight_grad.size(0), -1), dim=1)
                            self.gradient_accumulator[name].append(channel_importance.detach().cpu())
                            gradient_collected += 1
                            
                            if batch_idx == 0:  # 只在第一个batch打印
                                print(f"    ✓ {name}: 梯度形状 {weight_grad.shape}")
                        else:
                            if batch_idx == 0:
                                print(f"    ✗ {name}: 无梯度")
                
                print(f"  收集到 {gradient_collected} 层的梯度")
                successful_batches += 1
                
            except Exception as e:
                print(f"Batch {batch_idx + 1} 处理失败: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        print(f"\n成功处理了 {successful_batches} 个batch")
        
        # 计算重要性分数
        self._calculate_importance_scores()
        
        return len(self.importance_scores) > 0
    
    def _calculate_importance_scores(self):
        """计算重要性分数"""
        print("\n计算重要性分数...")
        
        valid_layers = 0
        for name, grad_list in self.gradient_accumulator.items():
            if len(grad_list) > 0:
                # 堆叠所有batch的梯度
                gradients = torch.stack(grad_list)
                # 计算平均重要性
                importance = torch.mean(gradients, dim=0)
                self.importance_scores[name] = importance
                valid_layers += 1
                print(f"✓ {name}: {len(grad_list)} 个梯度样本, 重要性分数形状 {importance.shape}")
            else:
                print(f"✗ {name}: 没有收集到梯度")
        
        print(f"总共 {valid_layers} 层有有效的重要性分数")
        
        # 如果没有收集到任何梯度，使用权重大小作为备用方案
        if valid_layers == 0:
            print("\n使用权重大小作为备用方案...")
            self._use_weight_magnitude_as_importance()
    
    def _use_weight_magnitude_as_importance(self):
        """使用权重大小作为重要性指标"""
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Conv3d) and hasattr(module, 'weight'):
                weight = module.weight.data
                # 计算每个输出通道的权重L2范数
                channel_importance = torch.norm(weight.view(weight.size(0), -1), dim=1)
                self.importance_scores[name] = channel_importance.cpu()
                print(f"✓ {name}: 权重重要性形状 {channel_importance.shape}")
        
        print(f"通过权重大小获得 {len(self.importance_scores)} 层的重要性分数")
    
    def generate_pruning_plan(self, pruning_ratio=0.3):
        """生成剪枝方案"""
        print(f"\n生成剪枝方案，目标剪枝比例: {pruning_ratio}")
        
        if len(self.importance_scores) == 0:
            raise ValueError("没有重要性分数，无法生成剪枝方案")
        
        pruning_plan = {}
        
        for name, importance in self.importance_scores.items():
            num_channels = len(importance)
            
            # 确保至少保留足够的通道
            min_keep = max(1, int(num_channels * 0.25))
            num_keep = max(min_keep, int(num_channels * (1 - pruning_ratio)))
            
            # 找到最重要的通道
            _, sorted_indices = torch.sort(importance, descending=True)
            keep_indices = sorted_indices[:num_keep].sort()[0]  # 保持索引有序
            
            reduction_ratio = 1 - num_keep / num_channels
            
            pruning_plan[name] = {
                'keep_indices': keep_indices,
                'original_channels': num_channels,
                'pruned_channels': num_keep,
                'reduction_ratio': reduction_ratio
            }
            
            print(f"{name}: {num_channels} -> {num_keep} channels (减少 {reduction_ratio*100:.1f}%)")
        
        return pruning_plan
    
    def create_pruned_model(self, cfg, pruning_plan):
        """创建剪枝后的模型"""
        print("\n创建剪枝后的模型...")
        
        if len(pruning_plan) == 0:
            raise ValueError("剪枝方案为空，无法创建模型")
        
        # 计算新的filter配置
        reduction_ratios = [plan['reduction_ratio'] for plan in pruning_plan.values()]
        avg_reduction = np.mean(reduction_ratios)
        
        print(f"平均减少比例: {avg_reduction:.3f}")
        
        # 保守地调整filters
        original_filters = cfg.MODEL.filters
        pruned_filters = []
        
        for i, original_filter in enumerate(original_filters):
            # 不同位置使用不同的减少策略
            if i == 0:  # 第一层减少得最少
                reduction_factor = avg_reduction * 0.3
            elif i < len(original_filters) // 2:  # 前半部分
                reduction_factor = avg_reduction * 0.6
            else:  # 后半部分
                reduction_factor = avg_reduction * 0.8
            
            new_filter_count = max(8, int(original_filter * (1 - reduction_factor)))
            pruned_filters.append(new_filter_count)
        
        print(f"原始filters: {original_filters}")
        print(f"剪枝后filters: {pruned_filters}")
        
        # 创建新模型 - 关键修复：确保模型在正确的设备上
        from model_superhuman2_multitask import UNet_PNI
        
        print(f"在设备 {self.device} 上创建模型...")
        
        pruned_model = UNet_PNI(
            in_planes=cfg.MODEL.input_nc,
            out_planes=3,  # 只输出affinity
            filters=pruned_filters,
            upsample_mode=cfg.MODEL.upsample_mode,
            decode_ratio=cfg.MODEL.decode_ratio,
            merge_mode=cfg.MODEL.merge_mode,
            pad_mode=cfg.MODEL.pad_mode,
            bn_mode=cfg.MODEL.bn_mode,
            relu_mode=cfg.MODEL.relu_mode,
            init_mode=cfg.MODEL.init_mode,
            if_skele=False,
            if_dt=False,
        ).to(self.device)  # 关键修复：将模型移动到正确的设备

        pruned_model.pruned_filters = pruned_filters
        
        print("✓ 成功创建剪枝后的模型")
        print(f"   剪枝后模型设备: {next(pruned_model.parameters()).device}")
        
        # 传输权重
        self._transfer_weights(pruned_model, pruning_plan)
        
        return pruned_model
    
    def _transfer_weights(self, pruned_model, pruning_plan):
        """传输权重到剪枝后的模型"""
        print("传输权重...")
        
        original_state_dict = self.model.state_dict()
        pruned_state_dict = pruned_model.state_dict()
        
        transferred = 0
        skipped = 0
        
        for name, param in pruned_state_dict.items():
            if name in original_state_dict:
                original_param = original_state_dict[name]
                
                # 如果形状相同，直接复制
                if param.shape == original_param.shape:
                    param.data.copy_(original_param.data)
                    transferred += 1
                else:
                    # 尝试按照剪枝方案选择权重
                    layer_name = '.'.join(name.split('.')[:-1])
                    if layer_name in pruning_plan and 'weight' in name:
                        try:
                            keep_indices = pruning_plan[layer_name]['keep_indices']
                            if len(original_param.shape) >= 1:
                                # 选择保留的输出通道
                                selected_param = original_param[keep_indices]
                                if selected_param.shape == param.shape:
                                    param.data.copy_(selected_param)
                                    transferred += 1
                                    print(f"  ✓ 剪枝传输: {name}")
                                    continue
                        except Exception as e:
                            print(f"  剪枝传输失败 {name}: {e}")
                    elif layer_name in pruning_plan and 'bias' in name:
                        try:
                            keep_indices = pruning_plan[layer_name]['keep_indices']
                            selected_param = original_param[keep_indices]
                            if selected_param.shape == param.shape:
                                param.data.copy_(selected_param)
                                transferred += 1
                                print(f"  ✓ 剪枝传输bias: {name}")
                                continue
                        except Exception as e:
                            print(f"  剪枝传输bias失败 {name}: {e}")
                    
                    skipped += 1
                    if skipped <= 5:  # 只打印前5个
                        print(f"  跳过: {name} {original_param.shape} -> {param.shape}")
            else:
                skipped += 1
        
        print(f"权重传输完成: 成功 {transferred}, 跳过 {skipped}")

def execute_complete_pruning(cfg_file, checkpoint_path, pruning_ratio=0.3, data_iteartion=10000):
    """执行完整的剪枝过程"""
    import yaml
    from attrdict import AttrDict
    from data_provider_labeled_hcl import Provider
    import os
    
    print("=" * 80)
    print("开始完整剪枝过程")
    print("=" * 80)
    
    # 加载配置
    with open(os.path.join(os.path.dirname(__file__), 'config', cfg_file + '.yaml'), 'r') as f:
        cfg = AttrDict(yaml.safe_load(f))
    
    device = torch.device('cuda:0')
    print(f"使用设备: {device}")
    
    # 加载原始模型
    print("1. 加载原始模型...")
    from model_superhuman2_multitask import UNet_PNI
    
    original_model = UNet_PNI(
        in_planes=cfg.MODEL.input_nc,
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
    
    # 加载检查点
    print(f"   加载检查点: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    original_model.load_state_dict(checkpoint['model_weights'])
    
    # 验证模型在正确设备上
    print(f"   原始模型设备: {next(original_model.parameters()).device}")
    
    # 统计原始模型参数
    orig_total = sum(p.numel() for p in original_model.parameters())
    print(f"   原始模型参数: {orig_total:,}")
    
    # 创建数据提供器
    print("2. 准备数据...")
    train_provider = Provider('train', cfg)
    
    # 创建剪枝器
    print("3. 创建剪枝器...")
    pruner = CompleteStructuralPruner(original_model)
    
    num_batches_setting = data_iteartion
    # 收集梯度
    print("4. 收集梯度信息...")
    success = pruner.collect_gradients_direct_method(train_provider, cfg, num_batches=num_batches_setting)
    
    if not success:
        raise RuntimeError("梯度收集失败")
    
    # 生成剪枝方案
    print("5. 生成剪枝方案...")
    pruning_plan = pruner.generate_pruning_plan(pruning_ratio)
    
    # 创建剪枝后的模型
    print("6. 创建剪枝后的模型...")
    pruned_model = pruner.create_pruned_model(cfg, pruning_plan)
    
    # 获取pruned_filters
    pruned_filters = getattr(pruned_model, 'pruned_filters', None)
    if pruned_filters is None:
        raise AttributeError('No pruned_filters')
        # 如果没有找到，从剪枝方案中重新计算
        reduction_ratios = [plan['reduction_ratio'] for plan in pruning_plan.values()]
        avg_reduction = np.mean(reduction_ratios)
        original_filters = cfg.MODEL.filters
        pruned_filters = []
        
        for i, original_filter in enumerate(original_filters):
            if i == 0:
                reduction_factor = avg_reduction * 0.3
            elif i < len(original_filters) // 2:
                reduction_factor = avg_reduction * 0.6
            else:
                reduction_factor = avg_reduction * 0.8
            
            new_filter_count = max(8, int(original_filter * (1 - reduction_factor)))
            pruned_filters.append(new_filter_count)

    # 统计剪枝后模型参数
    pruned_total = sum(p.numel() for p in pruned_model.parameters())
    reduction = (orig_total - pruned_total) / orig_total
    
    print("\n" + "=" * 80)
    print("剪枝结果汇总:")
    print(f"原始模型参数: {orig_total:,}")
    print(f"剪枝后参数: {pruned_total:,}")
    print(f"减少参数: {orig_total - pruned_total:,}")
    print(f"减少比例: {reduction*100:.2f}%")
    print("=" * 80)
    
    # 测试模型
    print("7. 测试剪枝后的模型...")
    test_input = torch.randn(1, cfg.MODEL.input_nc, 18, 160, 160).to(device)
    print(f"   测试输入设备: {test_input.device}")
    print(f"   剪枝后模型设备: {next(pruned_model.parameters()).device}")
    
    try:
        with torch.no_grad():
            original_output = original_model(test_input)
            pruned_output = pruned_model(test_input)
            print(f"   ✓ 测试成功")
            print(f"   原始输出形状: {original_output.shape}")
            print(f"   剪枝后输出形状: {pruned_output.shape}")
    except Exception as e:
        print(f"   ✗ 测试失败: {e}")
        print(f"   调试信息:")
        print(f"     - 测试输入设备: {test_input.device}")
        print(f"     - 原始模型设备: {next(original_model.parameters()).device}")
        print(f"     - 剪枝模型设备: {next(pruned_model.parameters()).device}")
        raise
    
    # 保存模型
    print("8. 保存剪枝后的模型...")
    save_dir = 'outputs/models_pruning'
    os.makedirs(save_dir, exist_ok=True)
    save_path = f'{save_dir}/{cfg.DATA.dataset_name}_pruned_model_ratio_{pruning_ratio}_{num_batches_setting}.pth'
    
    # 将模型移动到CPU以便保存
    pruned_model_cpu = pruned_model.cpu()
    
    torch.save({
        'model_weights': pruned_model_cpu.state_dict(),
        'pruning_plan': pruning_plan,
        'original_params': orig_total,
        'pruned_params': pruned_total,
        'reduction_ratio': reduction,
        'config': cfg,
        # 'filters': getattr(pruned_model_cpu, 'filters', None),
        'filters': pruned_filters,
        'pruning_info': {
            'method': 'gradient_based_structural',
            'target_ratio': pruning_ratio,
            'actual_ratio': reduction,
            'device': str(device)
        }
    }, save_path)
    
    # 将模型移回GPU
    pruned_model = pruned_model_cpu.to(device)
    
    print(f"   模型保存到: {save_path}")
    print("=" * 80)
    print("剪枝过程完成！")
    print("=" * 80)
    
    return save_path, pruned_model

if __name__ == "__main__":
    import sys
    import os
    import argparse

    # 添加当前目录到路径
    sys.path.append(os.getcwd())
    
    parser = argparse.ArgumentParser(description='模型剪枝程序')
    parser.add_argument('--cfg', type=str, 
                        default='seg_3d_ac34_b2_skele_dt_11_bce_mse_hcl',
                        help='配置文件名')
    parser.add_argument('--checkpoint', type=str, 
                        required=True,
                        help='检查点文件路径')
    parser.add_argument('--ratio', type=float, 
                        default=0.3,
                        help='剪枝比例')
    parser.add_argument('--data', type=int, 
                        default=25000,
                        help='迭代次数')
    
    args = parser.parse_args()
    
    cfg_file = args.cfg
    checkpoint_path = args.checkpoint
    pruning_ratio = args.ratio
    data_iteration = args.data
    
    print(f"配置文件: {cfg_file}")
    print(f"检查点: {checkpoint_path}")
    print(f"剪枝比例: {pruning_ratio}")
    
    try:
        pruned_model_path, pruned_model = execute_complete_pruning(
            cfg_file, checkpoint_path, pruning_ratio, data_iteration
        )
        print("剪枝成功完成！")
    except Exception as e:
        print(f"剪枝失败: {e}")
        import traceback
        traceback.print_exc()