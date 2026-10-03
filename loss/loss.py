from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.nn.functional as F

#######################################################
# 0. Main loss functions
#######################################################

class PhasedPriorsCurriculum(nn.Module):
    """
    Phased Priors Curriculum (PPC) 框架
    Phase 1: Morphological Scaffolding (形态脚手架)
    Phase 2: Topological Refinement (拓扑精化)  
    Phase 3: Collaborative Harmonization (协同和谐化)
    """
    def __init__(self, if_skele=False, if_dt=False, 
                 phase1_iters=30000, phase2_iters=60000,
                 w_dt_phase1=10.0, w_skel_phase2=10.0):
        super().__init__()
        
        self.if_skele = if_skele
        self.if_dt = if_dt
        self.phase1_iters = phase1_iters  # Phase 1 结束的迭代数
        self.phase2_iters = phase2_iters  # Phase 2 结束的迭代数
        self.w_dt_phase1 = w_dt_phase1    # Phase 1 中 DT 的权重
        self.w_skel_phase2 = w_skel_phase2 # Phase 2 中 Skeleton 的权重
        
        # Phase 3 的不确定性权重参数
        self.nb_task = 3 if (self.if_skele and self.if_dt) else 2
        self.log_vars = nn.ParameterList([
            nn.Parameter(torch.zeros(1), requires_grad=True) for _ in range(self.nb_task)
        ])

    def get_parameters_info(self):
            """获取当前参数信息，用于保存和日志记录"""
            info = {
                'phase1_iters': self.phase1_iters,
                'phase2_iters': self.phase2_iters,
                'w_dt_phase1': self.w_dt_phase1,
                'w_skel_phase2': self.w_skel_phase2,
                'if_skele': self.if_skele,
                'if_dt': self.if_dt,
                'nb_task': self.nb_task
            }
            
            # 添加不确定性参数的当前值
            task_names = ['aff']
            if self.if_skele:
                task_names.append('skel')
            if self.if_dt:
                task_names.append('dt')
            
            uncertainty_params = {}
            for i, task_name in enumerate(task_names):
                if i < len(self.log_vars):
                    log_var_value = self.log_vars[i].item()
                    sigma_value = torch.exp(self.log_vars[i]).item()
                    uncertainty_params[f'log_var_{task_name}'] = log_var_value
                    uncertainty_params[f'sigma_{task_name}'] = sigma_value
            
            info['uncertainty_params'] = uncertainty_params
            return info        

    def get_current_phase(self, current_iter):
        """获取当前训练阶段"""
        if current_iter <= self.phase1_iters:
            return 1
        elif current_iter <= self.phase2_iters:
            return 2
        else:
            return 3
    
    def forward(self, loss_aff, loss_skel=None, loss_dt=None, current_iter=0):
        """
        根据当前迭代数选择相应的损失计算策略
        """
        current_phase = self.get_current_phase(current_iter)
        
        if current_phase == 1:
            # Phase 1: Morphological Scaffolding
            # L_Phase1 = L_aff + w_dt * L_dt
            total_loss = loss_aff
            if self.if_dt and loss_dt is not None:
                total_loss = total_loss + self.w_dt_phase1 * loss_dt
            return total_loss, current_phase
            
        elif current_phase == 2:
            # Phase 2: Topological Refinement  
            # L_Phase2 = L_aff + w_skel * L_skel
            total_loss = loss_aff
            if self.if_skele and loss_skel is not None:
                total_loss = total_loss + self.w_skel_phase2 * loss_skel
            return total_loss, current_phase
            
        else:
            # Phase 3: Collaborative Harmonization
            # 使用不确定性加权: L_Phase3 = Σ(1/(2σ²) * L_i + log(σ))
            losses = [loss_aff]
            if self.if_skele and loss_skel is not None:
                losses.append(loss_skel)
            if self.if_dt and loss_dt is not None:
                losses.append(loss_dt)
            
            total_loss = 0
            for i, loss in enumerate(losses):
                precision = torch.exp(-self.log_vars[i])
                total_loss += precision * loss + self.log_vars[i]
            
            return total_loss, current_phase
    
    def get_phase_info(self, current_iter):
        """获取当前阶段的详细信息，用于日志记录"""
        phase = self.get_current_phase(current_iter)
        info = {'phase': phase}
        
        if phase == 1:
            info['description'] = 'Morphological Scaffolding'
            info['active_tasks'] = ['aff', 'dt'] if self.if_dt else ['aff']
            info['weights'] = {'dt': self.w_dt_phase1} if self.if_dt else {}
            
        elif phase == 2:
            info['description'] = 'Topological Refinement'
            info['active_tasks'] = ['aff', 'skel'] if self.if_skele else ['aff']
            info['weights'] = {'skel': self.w_skel_phase2} if self.if_skele else {}
            
        else:
            info['description'] = 'Collaborative Harmonization'
            info['active_tasks'] = ['aff']
            if self.if_skele:
                info['active_tasks'].append('skel')
            if self.if_dt:
                info['active_tasks'].append('dt')
            
            # 获取当前的不确定性参数
            uncertainty_info = {}
            task_names = ['aff']
            if self.if_skele:
                task_names.append('skel')
            if self.if_dt:
                task_names.append('dt')
                
            for i, task_name in enumerate(task_names):
                if i < len(self.log_vars):
                    sigma = torch.exp(self.log_vars[i]).item()
                    uncertainty_info[f'sigma_{task_name}'] = sigma
            info['uncertainty_params'] = uncertainty_info
            
        return info



class HierarchicalCalibrationLoss(nn.Module):
    """Hierarchical Calibration Loss (HCL) for multi-task learning"""
    
    def __init__(self, if_skele=False, if_dt=False):
        super().__init__()
        
        self.if_skele = if_skele
        self.if_dt = if_dt
        
        # Global uncertainty parameter (log σ for numerical stability)
        self.log_sigma = nn.Parameter(torch.zeros(1), requires_grad=True)
        
        # Learnable relative importance parameters (log λ for ensuring positivity)
        if self.if_skele:
            self.log_lambda_skel = nn.Parameter(torch.zeros(1), requires_grad=True)
        if self.if_dt:
            self.log_lambda_dt = nn.Parameter(torch.zeros(1), requires_grad=True)
    
    def forward(self, loss_aff, loss_skel=None, loss_dt=None):
        # 确保所有损失都是 [1] 形状
        if loss_aff.shape == torch.Size([]):
            loss_aff = loss_aff.unsqueeze(0)
        
        if loss_skel is not None and loss_skel.shape == torch.Size([]):
            loss_skel = loss_skel.unsqueeze(0)
        
        if loss_dt is not None and loss_dt.shape == torch.Size([]):
            loss_dt = loss_dt.unsqueeze(0)

        # 直接使用，无需类型检查
        weighted_loss = loss_aff
        
        if self.if_skele and loss_skel is not None:
            lambda_skel = torch.exp(self.log_lambda_skel)
            weighted_loss += lambda_skel * loss_skel
        
        if self.if_dt and loss_dt is not None:
            lambda_dt = torch.exp(self.log_lambda_dt)
            weighted_loss += lambda_dt * loss_dt
        
        sigma_squared = torch.exp(2 * self.log_sigma)
        # HCL formulation: (1/2σ²) * weighted_loss + log(σ)
        hcl_loss = (1.0 / (2.0 * sigma_squared)) * weighted_loss + self.log_sigma
        
        return hcl_loss
    
    def get_parameters_info(self):
        """Get current parameter values for logging"""
        info = {
            'sigma': torch.exp(self.log_sigma).item(),
            'sigma_squared': torch.exp(2 * self.log_sigma).item()
        }
        
        if self.if_skele:
            info['lambda_skel'] = torch.exp(self.log_lambda_skel).item()
        if self.if_dt:
            info['lambda_dt'] = torch.exp(self.log_lambda_dt).item()
            
        return info

class PixelwiseUncertaintyWeights(nn.Module):
    def __init__(self, if_skele=False, if_dt=False):
        super().__init__()
        
        self.if_skele = if_skele
        self.if_dt = if_dt
        self.nb_task = 3 if (self.if_skele and self.if_dt) else 2
        
        # 为每个任务创建pixel-wise的不确定性预测头
        # 这些将在主模型中定义，这里只是处理权重计算
        
    def forward(self, losses, log_vars_pixelwise):
        """
        losses: list of [loss_aff, loss_skele, loss_dt] (pixel-wise losses)
        log_vars_pixelwise: list of pixel-wise log variance maps for each task
        """
        assert len(losses) == self.nb_task
        assert len(log_vars_pixelwise) == self.nb_task
        
        total_loss = 0
        for i in range(self.nb_task):
            # 计算pixel-wise的precision
            precision = torch.exp(-log_vars_pixelwise[i])
            
            # pixel-wise加权损失
            weighted_loss = precision * losses[i] + log_vars_pixelwise[i]
            total_loss += torch.mean(weighted_loss)
            
        return total_loss

class PixelwiseClampUncertaintyWeights(nn.Module):
    def __init__(self, if_skele=False, if_dt=False, log_var_clamp_min=-5.0, log_var_clamp_max=5.0, lambda_reg=1):
        super().__init__()
        
        self.if_skele = if_skele
        self.if_dt = if_dt
        self.nb_task = 3 if (self.if_skele and self.if_dt) else 2
        
        # 为每个任务创建pixel-wise的不确定性预测头
        # 这些将在主模型中定义，这里只是处理权重计算
        
        self.log_var_clamp_min = log_var_clamp_min
        self.log_var_clamp_max = log_var_clamp_max
        self.lambda_reg = lambda_reg

    def forward(self, losses, log_vars_pixelwise):
        """
        losses: list of [loss_aff, loss_skele, loss_dt] (pixel-wise losses)
        log_vars_pixelwise: list of pixel-wise log variance maps for each task
        """
        assert len(losses) == self.nb_task
        assert len(log_vars_pixelwise) == self.nb_task
        
        total_loss = 0
        for i in range(self.nb_task):
            # 计算pixel-wise的precision
            log_var = torch.clamp(log_vars_pixelwise[i], self.log_var_clamp_min, self.log_var_clamp_max)
            precision = torch.exp(-log_var)
            
            # pixel-wise加权损失
            weighted_loss = precision * losses[i] + self.lambda_reg * log_var
            total_loss += torch.mean(weighted_loss)
            
        return total_loss


class UncertaintyWeights(nn.Module):
    def __init__(self,
                 if_skele=False,
                 if_dt=False):
        super().__init__()
        
        self.if_skele = if_skele
        self.if_dt = if_dt
        self.nb_task = 3 if (self.if_skele and self.if_dt) else 2
        self.log_vars = nn.ParameterList([
            nn.Parameter(torch.zeros(1), requires_grad=True) for _ in range(self.nb_task)
        ])
        self.relu = torch.nn.ReLU()

        
    def forward(self, losses=None):

        assert len(losses) == self.nb_task

        loss = 0
        for i in range(self.nb_task):
            # s1 = torch.prod(torch.tensor(losses[i].size()[2:]).float())
            # s2 = losses[i].size()[0]
            # norm_term = (s1 * s2).cuda()
            # precision = torch.exp(-self.log_vars[i])
            # # precision = self.relu(precision)
            # # loss += torch.sum(precision * losses[i] + self.log_vars[i]) / norm_term
            # # loss += (torch.sum(precision * losses[i]) + self.log_vars[i]) / norm_term 
            # loss += torch.sum(precision * losses[i]) / norm_term + self.log_vars[i]
            
            # precision = torch.exp(-self.log_vars[i])
            # loss += precision * losses[i] + self.log_vars[i]

            precision = 2 * torch.exp(self.log_vars[i])**2
            loss += losses[i] / precision  + self.log_vars[i]


        return torch.mean(loss)

class MVGCConsistencyLoss(nn.Module):
    """
    Multi-View Geometric Consistency Loss
    实现MVGC的完整损失函数：L_MVGC = L_affinity + λ₁×L_skeleton + λ₂×L_distance + λ₃×L_consistency
    """
    def __init__(self, alpha=1.0, beta=1.0, learnable_weights=True):
        super().__init__()
        
        # 可学习的主要权重参数 λ₁, λ₂, λ₃
        self.lambda_skeleton = nn.Parameter(torch.tensor(1.0))  # λ₁
        self.lambda_distance = nn.Parameter(torch.tensor(1.0))  # λ₂  
        self.lambda_consistency = nn.Parameter(torch.tensor(0.5))  # λ₃
        
        if learnable_weights:
            # 可学习的一致性内部权重参数（log参数化保证正值）
            self.log_alpha = nn.Parameter(torch.tensor(0.0))  # exp(0) = 1.0
            self.log_beta = nn.Parameter(torch.tensor(0.0))
        else:
            # 固定权重
            self.register_buffer('log_alpha', torch.log(torch.tensor(alpha)))
            self.register_buffer('log_beta', torch.log(torch.tensor(beta)))
        
        self.learnable_weights = learnable_weights
    
    def compute_consistency_loss(self, pred_skeleton, pred_distance, pred_affinity):
        """
        计算一致性损失 L_consistency
        """
        # 1. 预处理：将亲和力转换为统一表示
        pred_connectivity = torch.mean(pred_affinity, dim=1, keepdim=True)
        
        # 2. 应用sigmoid确保[0,1]范围（如果模型输出没有sigmoid）
        pred_skeleton = torch.sigmoid(pred_skeleton)
        pred_distance = torch.sigmoid(pred_distance)
        pred_connectivity = torch.sigmoid(pred_connectivity)
        
        # 3. 计算拓扑-几何一致性
        diff_topo_geo = pred_skeleton - pred_distance
        loss_topo_geo = torch.mean(diff_topo_geo ** 2)
        
        # 4. 计算几何-连接一致性  
        diff_geo_conn = pred_distance - pred_connectivity
        loss_geo_conn = torch.mean(diff_geo_conn ** 2)
        
        # 5. 获取当前权重
        alpha = torch.exp(self.log_alpha)
        beta = torch.exp(self.log_beta)
        
        # 6. 加权组合得到一致性损失
        consistency_loss = alpha * loss_topo_geo + beta * loss_geo_conn
        
        # 7. 生成一致性图（用于可视化和分析）
        consistency_map = alpha * (diff_topo_geo ** 2) + beta * (diff_geo_conn ** 2)
        
        return consistency_loss, consistency_map, {
            'loss_topo_geo': loss_topo_geo,
            'loss_geo_conn': loss_geo_conn,
            'alpha': alpha,
            'beta': beta
        }
    
    def forward(self, loss_affinity, loss_skeleton, loss_distance, 
                pred_skeleton, pred_distance, pred_affinity):
        """
        计算完整的MVGC损失
        
        Args:
            loss_affinity: 亲和力损失 L_affinity
            loss_skeleton: 骨架损失 L_skeleton  
            loss_distance: 距离损失 L_distance
            pred_skeleton: [B, 1, D, H, W] 骨架预测（用于计算一致性）
            pred_distance: [B, 1, D, H, W] 距离预测
            pred_affinity: [B, 3, D, H, W] 亲和力预测
        
        Returns:
            total_loss: 总的MVGC损失
            loss_details: 详细的损失分解
            consistency_map: 一致性分数图
        """
        # 计算一致性损失
        consistency_loss, consistency_map, consistency_details = self.compute_consistency_loss(
            pred_skeleton, pred_distance, pred_affinity
        )
        
        # 获取可学习的权重参数（使用softplus确保正值）
        lambda_skeleton = F.softplus(self.lambda_skeleton)
        lambda_distance = F.softplus(self.lambda_distance) 
        lambda_consistency = F.softplus(self.lambda_consistency)
        
        # 计算总损失：L_MVGC = L_affinity + λ₁×L_skeleton + λ₂×L_distance + λ₃×L_consistency
        total_loss = (loss_affinity + 
                     lambda_skeleton * loss_skeleton +
                     lambda_distance * loss_distance +
                     lambda_consistency * consistency_loss)
        
        # 详细损失信息
        loss_details = {
            'loss_affinity': loss_affinity,
            'loss_skeleton': loss_skeleton,
            'loss_distance': loss_distance,
            'consistency_loss': consistency_loss,
            'lambda_skeleton': lambda_skeleton,
            'lambda_distance': lambda_distance,
            'lambda_consistency': lambda_consistency,
            'total_loss': total_loss,
            **consistency_details
        }
        
        return total_loss, loss_details, consistency_map
    
    def get_all_weights(self):
        """获取所有权重参数"""
        return {
            'lambda_skeleton': F.softplus(self.lambda_skeleton).item(),
            'lambda_distance': F.softplus(self.lambda_distance).item(),
            'lambda_consistency': F.softplus(self.lambda_consistency).item(),
            'alpha': torch.exp(self.log_alpha).item(),
            'beta': torch.exp(self.log_beta).item()
        }


class JaccardLoss(nn.Module):
    """Jaccard loss.
    """
    # binary case

    def __init__(self, size_average=True, reduce=True, smooth=1.0):
        super(JaccardLoss, self).__init__()
        self.smooth = smooth
        self.reduce = reduce

    def jaccard_loss(self, pred, target):
        loss = 0.
        # for each sample in the batch
        for index in range(pred.size()[0]):
            iflat = pred[index].view(-1)
            tflat = target[index].view(-1)
            intersection = (iflat * tflat).sum()
            loss += 1 - ((intersection + self.smooth) / 
                    ( iflat.sum() + tflat.sum() - intersection + self.smooth))
            #print('loss:',intersection, iflat.sum(), tflat.sum())

        # size_average=True for the jaccard loss
        return loss / float(pred.size()[0])

    def jaccard_loss_batch(self, pred, target):
        iflat = pred.view(-1)
        tflat = target.view(-1)
        intersection = (iflat * tflat).sum()
        loss = 1 - ((intersection + self.smooth) / 
               ( iflat.sum() + tflat.sum() - intersection + self.smooth))
        #print('loss:',intersection, iflat.sum(), tflat.sum())
        return loss

    def forward(self, pred, target):
        #_assert_no_grad(target)
        if not (target.size() == pred.size()):
            raise ValueError("Target size ({}) must be the same as pred size ({})".format(target.size(), pred.size()))
        if self.reduce:
            loss = self.jaccard_loss(pred, target)
        else:    
            loss = self.jaccard_loss_batch(pred, target)
        return loss

class DiceLoss(nn.Module):
    """DICE loss.
    """
    # https://lars76.github.io/neural-networks/object-detection/losses-for-segmentation/

    def __init__(self, size_average=True, reduce=True, smooth=100.0, power=1):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        self.reduce = reduce
        self.power = power

    def dice_loss(self, pred, target):
        loss = 0.

        for index in range(pred.size()[0]):
            iflat = pred[index].view(-1)
            tflat = target[index].view(-1)
            intersection = (iflat * tflat).sum()
            if self.power==1:
                loss += 1 - ((2. * intersection + self.smooth) / 
                        ( iflat.sum() + tflat.sum() + self.smooth))
            else:
                loss += 1 - ((2. * intersection + self.smooth) / 
                        ( (iflat**self.power).sum() + (tflat**self.power).sum() + self.smooth))

        # size_average=True for the dice loss
        return loss / float(pred.size()[0])

    def dice_loss_batch(self, pred, target):
        iflat = pred.view(-1)
        tflat = target.view(-1)
        intersection = (iflat * tflat).sum()
        
        if self.power==1:
            loss = 1 - ((2. * intersection + self.smooth) / 
                   (iflat.sum() + tflat.sum() + self.smooth))
        else:
            loss = 1 - ((2. * intersection + self.smooth) / 
                   ( (iflat**self.power).sum() + (tflat**self.power).sum() + self.smooth))
        return loss

    def forward(self, pred, target):
        #_assert_no_grad(target)
        if not (target.size() == pred.size()):
            raise ValueError("Target size ({}) must be the same as pred size ({})".format(target.size(), pred.size()))

        if self.reduce:
            loss = self.dice_loss(pred, target)
        else:    
            loss = self.dice_loss_batch(pred, target)
        return loss

class WeightedMSE(nn.Module):
    """Weighted mean-squared error.
    """

    def __init__(self):
        super().__init__()

    def weighted_mse_loss(self, pred, target, weight):
        s1 = torch.prod(torch.tensor(pred.size()[2:]).float())
        s2 = pred.size()[0]
        norm_term = (s1 * s2).cuda()
        if weight is None:
            return torch.sum((pred - target) ** 2) / norm_term
        else:
            return torch.sum(weight * (pred - target) ** 2) / norm_term

    def forward(self, pred, target, weight=None):
        #_assert_no_grad(target)
        return self.weighted_mse_loss(pred, target, weight)

class MSELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.criterion = nn.MSELoss()
    
    def forward(self, pred, target, weight=None):
        return self.criterion(pred, target)

class BCELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.criterion = nn.BCELoss()
    
    def forward(self, pred, target, weight=None):
        return self.criterion(pred, target)

class WeightedBCE(nn.Module):
    """Weighted binary cross-entropy.
    """
    def __init__(self, size_average=True, reduce=True):
        super().__init__()
        self.size_average = size_average
        self.reduce = reduce

    def forward(self, pred, target, weight=None):
        #_assert_no_grad(target)
        return F.binary_cross_entropy(pred, target, weight)
    
class PixelwiseWeightedBCE(nn.Module):
    """Weighted binary cross-entropy.
    """
    def __init__(self, size_average=True, reduce=True):
        super().__init__()
        self.size_average = size_average
        self.reduce = reduce
        self.bce = nn.BCELoss(reduction='none')

    def forward(self, pred, target, weight=None):
        #_assert_no_grad(target)
        # return F.binary_cross_entropy(pred, target, weight)
        loss = self.bce(pred, target)
        loss = loss * weight

        return loss

class WeightedCE(nn.Module):
    """Mask weighted multi-class cross-entropy (CE) loss.
    """
    def __init__(self):
        super().__init__()

    def forward(self, pred, target, weight_mask=None):
        # Different from, F.binary_cross_entropy, the "weight" parameter
        # in F.cross_entropy is a manual rescaling weight given to each 
        # class. Therefore we need to multiply the weight mask after the
        # loss calculation.
        loss = F.cross_entropy(pred, target, reduction='none')
        if weight_mask is not None:
            loss = loss * weight_mask
        return loss.mean()

#######################################################
# 1. Regularization
#######################################################

class BinaryReg(nn.Module):
    """Regularization for encouraging the outputs to be binary.
    """
    def __init__(self, alpha=0.1):
        super().__init__()
        self.alpha = alpha
    
    def forward(self, pred):
        diff = pred - 0.5
        diff = torch.clamp(torch.abs(diff), min=1e-2)
        loss = (1.0 / diff).mean()
        return self.alpha * loss
