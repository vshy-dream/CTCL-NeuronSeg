# deployed model without much flexibility
# useful for stand-alone test, model translation, quantization
import torch.nn as nn
import torch.nn.functional as F
import torch

from model.basic import conv3dBlock, upsampleBlock
from model.residual import resBlock_pni
from model.model_para import model_structure


class UNet_PNI(nn.Module):  # deployed PNI model
    # Superhuman Accuracy on the SNEMI3D Connectomics Challenge. Lee et al.
    # https://arxiv.org/abs/1706.00120
    def __init__(self, in_planes=1,
                 out_planes=3,
                 filters=[28, 36, 48, 64, 80],  # [28, 36, 48, 64, 80], [32, 64, 128, 256, 512]
                 upsample_mode='transposeS',  # transposeS, bilinear
                 decode_ratio=1,
                 merge_mode='cat',
                 pad_mode='zero',
                 bn_mode='async',  # async or sync
                 relu_mode='elu',
                 init_mode='kaiming_normal',
                 bn_momentum=0.001,
                 do_embed=True,
                 if_sigmoid=True,
                 show_feature=False,
                 if_skele=False,
                 if_dt=False,
                 if_only_skele=False,
                 if_uncertainty=False
                 ):
        # filter_ratio: #filter_decode/#filter_encode
        super(UNet_PNI, self).__init__()
        filters2 = filters[:1] + filters
        self.merge_mode = merge_mode
        self.do_embed = do_embed
        self.depth = len(filters2) - 2
        self.if_sigmoid = if_sigmoid
        self.show_feature = show_feature
        self.if_skele = if_skele
        self.if_dt = if_dt
        self.skele_pseudo_loss = None
        self.dt_pseudo_loss = None
        self.temp_affs = None
        self.if_only_skele = if_only_skele
        self.if_uncertainty = if_uncertainty

        # 2D conv for anisotropic
        self.embed_in = conv3dBlock([in_planes],
                                    [filters2[0]],
                                    [(1, 5, 5)],
                                    [1],
                                    [(0, 2, 2)],
                                    [True],
                                    [pad_mode],
                                    [''],
                                    [relu_mode],
                                    init_mode,
                                    bn_momentum)

        # downsample stream
        self.conv0 = resBlock_pni(filters2[0], filters2[1], pad_mode, bn_mode, relu_mode, init_mode, bn_momentum)
        self.pool0 = nn.MaxPool3d((1, 2, 2), (1, 2, 2))

        self.conv1 = resBlock_pni(filters2[1], filters2[2], pad_mode, bn_mode, relu_mode, init_mode, bn_momentum)
        self.pool1 = nn.MaxPool3d((1, 2, 2), (1, 2, 2))

        self.conv2 = resBlock_pni(filters2[2], filters2[3], pad_mode, bn_mode, relu_mode, init_mode, bn_momentum)
        self.pool2 = nn.MaxPool3d((1, 2, 2), (1, 2, 2))

        self.conv3 = resBlock_pni(filters2[3], filters2[4], pad_mode, bn_mode, relu_mode, init_mode, bn_momentum)
        self.pool3 = nn.MaxPool3d((1, 2, 2), (1, 2, 2))

        self.center = resBlock_pni(filters2[4], filters2[5], pad_mode, bn_mode, relu_mode, init_mode, bn_momentum)

        # upsample stream
        self.up0 = upsampleBlock(filters2[5], filters2[4], (1, 2, 2), upsample_mode, init_mode=init_mode)
        if self.merge_mode == 'add':
            self.cat0 = conv3dBlock([0], [filters2[4]], bn_mode=[bn_mode], relu_mode=[relu_mode],
                                    bn_momentum=bn_momentum)
            self.conv4 = resBlock_pni(filters2[4], filters2[4], pad_mode, bn_mode, relu_mode, init_mode, bn_momentum)
        else:
            self.cat0 = conv3dBlock([0], [filters2[4] * 2], bn_mode=[bn_mode], relu_mode=[relu_mode],
                                    bn_momentum=bn_momentum)
            self.conv4 = resBlock_pni(filters2[4] * 2, filters2[4], pad_mode, bn_mode, relu_mode, init_mode,
                                      bn_momentum)

        self.up1 = upsampleBlock(filters2[4], filters2[3], (1, 2, 2), upsample_mode, init_mode=init_mode)
        if self.merge_mode == 'add':
            self.cat1 = conv3dBlock([0], [filters2[3]], bn_mode=[bn_mode], relu_mode=[relu_mode],
                                    bn_momentum=bn_momentum)
            self.conv5 = resBlock_pni(filters2[3], filters2[3], pad_mode, bn_mode, relu_mode, init_mode, bn_momentum)
        else:
            self.cat1 = conv3dBlock([0], [filters2[3] * 2], bn_mode=[bn_mode], relu_mode=[relu_mode],
                                    bn_momentum=bn_momentum)
            self.conv5 = resBlock_pni(filters2[3] * 2, filters2[3], pad_mode, bn_mode, relu_mode, init_mode,
                                      bn_momentum)

        self.up2 = upsampleBlock(filters2[3], filters2[2], (1, 2, 2), upsample_mode, init_mode=init_mode)
        if self.merge_mode == 'add':
            self.cat2 = conv3dBlock([0], [filters2[2]], bn_mode=[bn_mode], relu_mode=[relu_mode],
                                    bn_momentum=bn_momentum)
            self.conv6 = resBlock_pni(filters2[2], filters2[2], pad_mode, bn_mode, relu_mode, init_mode, bn_momentum)
        else:
            self.cat2 = conv3dBlock([0], [filters2[2] * 2], bn_mode=[bn_mode], relu_mode=[relu_mode],
                                    bn_momentum=bn_momentum)
            self.conv6 = resBlock_pni(filters2[2] * 2, filters2[2], pad_mode, bn_mode, relu_mode, init_mode,
                                      bn_momentum)

        self.up3 = upsampleBlock(filters2[2], filters2[1], (1, 2, 2), upsample_mode, init_mode=init_mode)
        if self.merge_mode == 'add':
            self.cat3 = conv3dBlock([0], [filters2[1]], bn_mode=[bn_mode], relu_mode=[relu_mode],
                                    bn_momentum=bn_momentum)
            self.conv7 = resBlock_pni(filters2[1], filters2[1], pad_mode, bn_mode, relu_mode, init_mode, bn_momentum)
        else:
            self.cat3 = conv3dBlock([0], [filters2[1] * 2], bn_mode=[bn_mode], relu_mode=[relu_mode],
                                    bn_momentum=bn_momentum)
            self.conv7 = resBlock_pni(filters2[1] * 2, filters2[1], pad_mode, bn_mode, relu_mode, init_mode,
                                      bn_momentum)

        self.embed_out = conv3dBlock([int(filters2[0])],
                                     [int(filters2[0])],
                                     [(1, 5, 5)],
                                     [1],
                                     [(0, 2, 2)],
                                     [True],
                                     [pad_mode],
                                     [''],
                                     [relu_mode],
                                     init_mode,
                                     bn_momentum)

        self.out_put = conv3dBlock([int(filters2[0])], [out_planes], [(1, 1, 1)], init_mode=init_mode)
        self.skeleton_out_put = conv3dBlock([int(filters2[0])], [1], [(1, 1, 1)], init_mode=init_mode)
        self.distanceTransform_out_put = conv3dBlock([int(filters2[0])], [1], [(1, 1, 1)], init_mode=init_mode)

        if self.if_uncertainty:
            self.affinity_uncertainty = conv3dBlock([int(filters2[0])], [out_planes], [(1, 1, 1)], init_mode=init_mode)
            if self.if_skele:
                self.skeleton_uncertainty = conv3dBlock([int(filters2[0])], [1], [(1, 1, 1)], init_mode=init_mode)
            if self.if_dt:
                self.dt_uncertainty = conv3dBlock([int(filters2[0])], [1], [(1, 1, 1)], init_mode=init_mode)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # embedding
        embed_in = self.embed_in(x)
        conv0 = self.conv0(embed_in)
        pool0 = self.pool0(conv0)
        conv1 = self.conv1(pool0)
        pool1 = self.pool1(conv1)
        conv2 = self.conv2(pool1)
        pool2 = self.pool2(conv2)
        conv3 = self.conv3(pool2)
        pool3 = self.pool3(conv3)

        center = self.center(pool3)

        up0 = self.up0(center)
        if self.merge_mode == 'add':
            cat0 = self.cat0(up0 + conv3)
        else:
            cat0 = self.cat0(torch.cat([up0, conv3], dim=1))
        conv4 = self.conv4(cat0)

        up1 = self.up1(conv4)
        if self.merge_mode == 'add':
            cat1 = self.cat1(up1 + conv2)
        else:
            cat1 = self.cat1(torch.cat([up1, conv2], dim=1))
        conv5 = self.conv5(cat1)

        up2 = self.up2(conv5)
        if self.merge_mode == 'add':
            cat2 = self.cat2(up2 + conv1)
        else:
            cat2 = self.cat2(torch.cat([up2, conv1], dim=1))
        conv6 = self.conv6(cat2)

        up3 = self.up3(conv6)
        if self.merge_mode == 'add':
            cat3 = self.cat3(up3 + conv0)
        else:
            cat3 = self.cat3(torch.cat([up3, conv0], dim=1))
        conv7 = self.conv7(cat3)

        embed_out = self.embed_out(conv7)
        if self.if_only_skele == True:
            out = self.skeleton_out_put(embed_out)
            uncertraintise = None
        else:
            if self.if_skele is True and self.if_dt is False:
                out_skele = self.skeleton_out_put(embed_out)
                out_aff = self.out_put(embed_out)
                out = torch.cat((out_skele, out_aff), dim=1)  # skele=out[:,:1] aff=out[:,1:]

                # 不确定性输出
                if self.if_uncertainty:
                    unc_skele = self.skeleton_uncertainty(embed_out)
                    unc_aff = self.affinity_uncertainty(embed_out)
                    uncertainties = [unc_aff, unc_skele]
                else:
                    uncertainties = None

            elif self.if_skele is False and self.if_dt is True:
                out_dt = self.distanceTransform_out_put(embed_out)
                out_aff = self.out_put(embed_out)
                out = torch.cat((out_dt, out_aff), dim=1)  # dt=out[:,:1]  aff=out[:,1:]

                if self.if_uncertainty:
                    unc_dt = self.dt_uncertainty(embed_out)
                    unc_aff = self.affinity_uncertainty(embed_out)
                    uncertainties = [unc_aff, unc_dt]
                else:
                    uncertainties = None


            elif self.if_skele is True and self.if_dt is True:
                out_skele = self.skeleton_out_put(embed_out)
                out_dt = self.distanceTransform_out_put(embed_out)
                out_aff = self.out_put(embed_out)
                out = torch.cat((out_skele, out_dt, out_aff), dim=1) # skele=out[:,:1] dt=out[:,1:2] aff=out[:,2:]

                if self.if_uncertainty:
                    unc_skele = self.skeleton_uncertainty(embed_out)
                    unc_dt = self.dt_uncertainty(embed_out)
                    unc_aff = self.affinity_uncertainty(embed_out)
                    uncertainties = [unc_aff, unc_skele, unc_dt]
                else:
                    uncertainties = None


            else:
                out = self.out_put(embed_out)

                if self.if_uncertainty:
                    uncertainties = [self.affinity_uncertainty(embed_out)]
                else:
                    uncertainties = None


        if self.if_sigmoid:
            # if self.if_skele is True:
            #     out[:, :1] = torch.sigmoid(out_skele)
            #     out[:, 1:] = torch.sigmoid(out_aff)
            # else:
            #     out = torch.sigmoid(out)
            out = torch.sigmoid(out)

        if self.show_feature:
            down_features = [conv0, conv1, conv2, conv3]
            center_features = [center]
            up_features = [conv4, conv5, conv6, conv7]
            return down_features, center_features, up_features, out
        else:
            if self.if_uncertainty:
                return out, uncertainties
            else:
                return out

class UNet_PNI_MOE(nn.Module):  # deployed PNI model
    # Superhuman Accuracy on the SNEMI3D Connectomics Challenge. Lee et al.
    # https://arxiv.org/abs/1706.00120
    def __init__(self, in_planes=1,
                 out_planes=3,
                 filters=[28, 36, 48, 64, 80],  # [28, 36, 48, 64, 80], [32, 64, 128, 256, 512]
                 upsample_mode='transposeS',  # transposeS, bilinear
                 decode_ratio=1,
                 merge_mode='cat',
                 pad_mode='zero',
                 bn_mode='async',  # async or sync
                 relu_mode='elu',
                 init_mode='kaiming_normal',
                 bn_momentum=0.001,
                 do_embed=True,
                 if_sigmoid=True,
                 show_feature=False,
                 if_skele=False,
                 if_dt=False,
                 if_only_skele=False,
                 if_hard=False):
        # filter_ratio: #filter_decode/#filter_encode
        super(UNet_PNI_MOE, self).__init__()
        filters2 = filters[:1] + filters
        self.merge_mode = merge_mode
        self.do_embed = do_embed
        self.depth = len(filters2) - 2
        self.if_sigmoid = if_sigmoid
        self.show_feature = show_feature
        self.if_skele = if_skele
        self.if_dt = if_dt
        self.skele_pseudo_loss = None
        self.dt_pseudo_loss = None
        self.temp_affs = None
        self.if_only_skele = if_only_skele
        self.feature_selector = FeatureSelector(input_dim=3, num_features=3)
        self.feature_weighting = FeatureWeighting(input_dim=3, num_features=3)
        self.if_hard = if_hard

        # 2D conv for anisotropic
        self.embed_in = conv3dBlock([in_planes],
                                    [filters2[0]],
                                    [(1, 5, 5)],
                                    [1],
                                    [(0, 2, 2)],
                                    [True],
                                    [pad_mode],
                                    [''],
                                    [relu_mode],
                                    init_mode,
                                    bn_momentum)

        # downsample stream
        self.conv0 = resBlock_pni(filters2[0], filters2[1], pad_mode, bn_mode, relu_mode, init_mode, bn_momentum)
        self.pool0 = nn.MaxPool3d((1, 2, 2), (1, 2, 2))

        self.conv1 = resBlock_pni(filters2[1], filters2[2], pad_mode, bn_mode, relu_mode, init_mode, bn_momentum)
        self.pool1 = nn.MaxPool3d((1, 2, 2), (1, 2, 2))

        self.conv2 = resBlock_pni(filters2[2], filters2[3], pad_mode, bn_mode, relu_mode, init_mode, bn_momentum)
        self.pool2 = nn.MaxPool3d((1, 2, 2), (1, 2, 2))

        self.conv3 = resBlock_pni(filters2[3], filters2[4], pad_mode, bn_mode, relu_mode, init_mode, bn_momentum)
        self.pool3 = nn.MaxPool3d((1, 2, 2), (1, 2, 2))

        self.center = resBlock_pni(filters2[4], filters2[5], pad_mode, bn_mode, relu_mode, init_mode, bn_momentum)

        # upsample stream
        self.up0 = upsampleBlock(filters2[5], filters2[4], (1, 2, 2), upsample_mode, init_mode=init_mode)
        if self.merge_mode == 'add':
            self.cat0 = conv3dBlock([0], [filters2[4]], bn_mode=[bn_mode], relu_mode=[relu_mode],
                                    bn_momentum=bn_momentum)
            self.conv4 = resBlock_pni(filters2[4], filters2[4], pad_mode, bn_mode, relu_mode, init_mode, bn_momentum)
        else:
            self.cat0 = conv3dBlock([0], [filters2[4] * 2], bn_mode=[bn_mode], relu_mode=[relu_mode],
                                    bn_momentum=bn_momentum)
            self.conv4 = resBlock_pni(filters2[4] * 2, filters2[4], pad_mode, bn_mode, relu_mode, init_mode,
                                      bn_momentum)

        self.up1 = upsampleBlock(filters2[4], filters2[3], (1, 2, 2), upsample_mode, init_mode=init_mode)
        if self.merge_mode == 'add':
            self.cat1 = conv3dBlock([0], [filters2[3]], bn_mode=[bn_mode], relu_mode=[relu_mode],
                                    bn_momentum=bn_momentum)
            self.conv5 = resBlock_pni(filters2[3], filters2[3], pad_mode, bn_mode, relu_mode, init_mode, bn_momentum)
        else:
            self.cat1 = conv3dBlock([0], [filters2[3] * 2], bn_mode=[bn_mode], relu_mode=[relu_mode],
                                    bn_momentum=bn_momentum)
            self.conv5 = resBlock_pni(filters2[3] * 2, filters2[3], pad_mode, bn_mode, relu_mode, init_mode,
                                      bn_momentum)

        self.up2 = upsampleBlock(filters2[3], filters2[2], (1, 2, 2), upsample_mode, init_mode=init_mode)
        if self.merge_mode == 'add':
            self.cat2 = conv3dBlock([0], [filters2[2]], bn_mode=[bn_mode], relu_mode=[relu_mode],
                                    bn_momentum=bn_momentum)
            self.conv6 = resBlock_pni(filters2[2], filters2[2], pad_mode, bn_mode, relu_mode, init_mode, bn_momentum)
        else:
            self.cat2 = conv3dBlock([0], [filters2[2] * 2], bn_mode=[bn_mode], relu_mode=[relu_mode],
                                    bn_momentum=bn_momentum)
            self.conv6 = resBlock_pni(filters2[2] * 2, filters2[2], pad_mode, bn_mode, relu_mode, init_mode,
                                      bn_momentum)

        self.up3 = upsampleBlock(filters2[2], filters2[1], (1, 2, 2), upsample_mode, init_mode=init_mode)
        if self.merge_mode == 'add':
            self.cat3 = conv3dBlock([0], [filters2[1]], bn_mode=[bn_mode], relu_mode=[relu_mode],
                                    bn_momentum=bn_momentum)
            self.conv7 = resBlock_pni(filters2[1], filters2[1], pad_mode, bn_mode, relu_mode, init_mode, bn_momentum)
        else:
            self.cat3 = conv3dBlock([0], [filters2[1] * 2], bn_mode=[bn_mode], relu_mode=[relu_mode],
                                    bn_momentum=bn_momentum)
            self.conv7 = resBlock_pni(filters2[1] * 2, filters2[1], pad_mode, bn_mode, relu_mode, init_mode,
                                      bn_momentum)

        self.embed_out = conv3dBlock([int(filters2[0])],
                                     [int(filters2[0])],
                                     [(1, 5, 5)],
                                     [1],
                                     [(0, 2, 2)],
                                     [True],
                                     [pad_mode],
                                     [''],
                                     [relu_mode],
                                     init_mode,
                                     bn_momentum)

        self.out_put = conv3dBlock([int(filters2[0])], [out_planes], [(1, 1, 1)], init_mode=init_mode)
        self.skeleton_out_put = conv3dBlock([int(filters2[0])], [1], [(1, 1, 1)], init_mode=init_mode)
        self.distanceTransform_out_put = conv3dBlock([int(filters2[0])], [1], [(1, 1, 1)], init_mode=init_mode)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # MOE feature selector
        if self.if_hard is True:
            x = self.feature_selector(x)
        else:
            x, x_weight = self.feature_weighting(x)
        # embedding
        embed_in = self.embed_in(x)
        conv0 = self.conv0(embed_in)
        pool0 = self.pool0(conv0)
        conv1 = self.conv1(pool0)
        pool1 = self.pool1(conv1)
        conv2 = self.conv2(pool1)
        pool2 = self.pool2(conv2)
        conv3 = self.conv3(pool2)
        pool3 = self.pool3(conv3)

        center = self.center(pool3)

        up0 = self.up0(center)
        if self.merge_mode == 'add':
            cat0 = self.cat0(up0 + conv3)
        else:
            cat0 = self.cat0(torch.cat([up0, conv3], dim=1))
        conv4 = self.conv4(cat0)

        up1 = self.up1(conv4)
        if self.merge_mode == 'add':
            cat1 = self.cat1(up1 + conv2)
        else:
            cat1 = self.cat1(torch.cat([up1, conv2], dim=1))
        conv5 = self.conv5(cat1)

        up2 = self.up2(conv5)
        if self.merge_mode == 'add':
            cat2 = self.cat2(up2 + conv1)
        else:
            cat2 = self.cat2(torch.cat([up2, conv1], dim=1))
        conv6 = self.conv6(cat2)

        up3 = self.up3(conv6)
        if self.merge_mode == 'add':
            cat3 = self.cat3(up3 + conv0)
        else:
            cat3 = self.cat3(torch.cat([up3, conv0], dim=1))
        conv7 = self.conv7(cat3)

        embed_out = self.embed_out(conv7)
        if self.if_only_skele == True:
            out = self.skeleton_out_put(embed_out)
        else:
            if self.if_skele is True and self.if_dt is False:
                out_skele = self.skeleton_out_put(embed_out)
                out_aff = self.out_put(embed_out)
                out = torch.cat((out_skele, out_aff), dim=1)  # skele=out[:,:1] aff=out[:,1:]
            elif self.if_skele is False and self.if_dt is True:
                out_dt = self.distanceTransform_out_put(embed_out)
                out_aff = self.out_put(embed_out)
                out = torch.cat((out_dt, out_aff), dim=1)  # dt=out[:,:1]  aff=out[:,1:]
            elif self.if_skele is True and self.if_dt is True:
                out_skele = self.skeleton_out_put(embed_out)
                out_dt = self.distanceTransform_out_put(embed_out)
                out_aff = self.out_put(embed_out)
                out = torch.cat((out_skele, out_dt, out_aff), dim=1) # skele=out[:,:1] dt=out[:,1:2] aff=out[:,2:]
            else:
                out = self.out_put(embed_out)

        if self.if_sigmoid:
            # if self.if_skele is True:
            #     out[:, :1] = torch.sigmoid(out_skele)
            #     out[:, 1:] = torch.sigmoid(out_aff)
            # else:
            #     out = torch.sigmoid(out)
            out = torch.sigmoid(out)

        if self.show_feature:
            down_features = [conv0, conv1, conv2, conv3]
            center_features = [center]
            up_features = [conv4, conv5, conv6, conv7]
            return down_features, center_features, up_features, out
        else:
            return out


class UNet_PNI_MOE_Add(nn.Module):  # deployed PNI model
    # Superhuman Accuracy on the SNEMI3D Connectomics Challenge. Lee et al.
    # https://arxiv.org/abs/1706.00120
    def __init__(self, in_planes=1,
                 out_planes=3,
                 filters=[28, 36, 48, 64, 80],  # [28, 36, 48, 64, 80], [32, 64, 128, 256, 512]
                 upsample_mode='transposeS',  # transposeS, bilinear
                 decode_ratio=1,
                 merge_mode='cat',
                 pad_mode='zero',
                 bn_mode='async',  # async or sync
                 relu_mode='elu',
                 init_mode='kaiming_normal',
                 bn_momentum=0.001,
                 do_embed=True,
                 if_sigmoid=True,
                 show_feature=False,
                 if_skele=False,
                 if_dt=False,
                 if_only_skele=False,
                 if_hard=False,
                 if_residual=False):
        # filter_ratio: #filter_decode/#filter_encode
        super(UNet_PNI_MOE_Add, self).__init__()
        filters2 = filters[:1] + filters
        self.merge_mode = merge_mode
        self.do_embed = do_embed
        self.depth = len(filters2) - 2
        self.if_sigmoid = if_sigmoid
        self.show_feature = show_feature
        self.if_skele = if_skele
        self.if_dt = if_dt
        self.skele_pseudo_loss = None
        self.dt_pseudo_loss = None
        self.temp_affs = None
        self.if_only_skele = if_only_skele
        self.feature_selector = FeatureSelectorConvAdd(input_dim=3, num_features=3)
        self.feature_weighting = FeatureWeighting(input_dim=3, num_features=3)
        self.if_hard = if_hard
        self.if_residual = if_residual

        # 2D conv for anisotropic
        self.embed_in = conv3dBlock([in_planes],
                                    [filters2[0]],
                                    [(1, 5, 5)],
                                    [1],
                                    [(0, 2, 2)],
                                    [True],
                                    [pad_mode],
                                    [''],
                                    [relu_mode],
                                    init_mode,
                                    bn_momentum)

        # downsample stream
        self.conv0 = resBlock_pni(filters2[0], filters2[1], pad_mode, bn_mode, relu_mode, init_mode, bn_momentum)
        self.pool0 = nn.MaxPool3d((1, 2, 2), (1, 2, 2))

        self.conv1 = resBlock_pni(filters2[1], filters2[2], pad_mode, bn_mode, relu_mode, init_mode, bn_momentum)
        self.pool1 = nn.MaxPool3d((1, 2, 2), (1, 2, 2))

        self.conv2 = resBlock_pni(filters2[2], filters2[3], pad_mode, bn_mode, relu_mode, init_mode, bn_momentum)
        self.pool2 = nn.MaxPool3d((1, 2, 2), (1, 2, 2))

        self.conv3 = resBlock_pni(filters2[3], filters2[4], pad_mode, bn_mode, relu_mode, init_mode, bn_momentum)
        self.pool3 = nn.MaxPool3d((1, 2, 2), (1, 2, 2))

        self.center = resBlock_pni(filters2[4], filters2[5], pad_mode, bn_mode, relu_mode, init_mode, bn_momentum)

        # upsample stream
        self.up0 = upsampleBlock(filters2[5], filters2[4], (1, 2, 2), upsample_mode, init_mode=init_mode)
        if self.merge_mode == 'add':
            self.cat0 = conv3dBlock([0], [filters2[4]], bn_mode=[bn_mode], relu_mode=[relu_mode],
                                    bn_momentum=bn_momentum)
            self.conv4 = resBlock_pni(filters2[4], filters2[4], pad_mode, bn_mode, relu_mode, init_mode, bn_momentum)
        else:
            self.cat0 = conv3dBlock([0], [filters2[4] * 2], bn_mode=[bn_mode], relu_mode=[relu_mode],
                                    bn_momentum=bn_momentum)
            self.conv4 = resBlock_pni(filters2[4] * 2, filters2[4], pad_mode, bn_mode, relu_mode, init_mode,
                                      bn_momentum)

        self.up1 = upsampleBlock(filters2[4], filters2[3], (1, 2, 2), upsample_mode, init_mode=init_mode)
        if self.merge_mode == 'add':
            self.cat1 = conv3dBlock([0], [filters2[3]], bn_mode=[bn_mode], relu_mode=[relu_mode],
                                    bn_momentum=bn_momentum)
            self.conv5 = resBlock_pni(filters2[3], filters2[3], pad_mode, bn_mode, relu_mode, init_mode, bn_momentum)
        else:
            self.cat1 = conv3dBlock([0], [filters2[3] * 2], bn_mode=[bn_mode], relu_mode=[relu_mode],
                                    bn_momentum=bn_momentum)
            self.conv5 = resBlock_pni(filters2[3] * 2, filters2[3], pad_mode, bn_mode, relu_mode, init_mode,
                                      bn_momentum)

        self.up2 = upsampleBlock(filters2[3], filters2[2], (1, 2, 2), upsample_mode, init_mode=init_mode)
        if self.merge_mode == 'add':
            self.cat2 = conv3dBlock([0], [filters2[2]], bn_mode=[bn_mode], relu_mode=[relu_mode],
                                    bn_momentum=bn_momentum)
            self.conv6 = resBlock_pni(filters2[2], filters2[2], pad_mode, bn_mode, relu_mode, init_mode, bn_momentum)
        else:
            self.cat2 = conv3dBlock([0], [filters2[2] * 2], bn_mode=[bn_mode], relu_mode=[relu_mode],
                                    bn_momentum=bn_momentum)
            self.conv6 = resBlock_pni(filters2[2] * 2, filters2[2], pad_mode, bn_mode, relu_mode, init_mode,
                                      bn_momentum)

        self.up3 = upsampleBlock(filters2[2], filters2[1], (1, 2, 2), upsample_mode, init_mode=init_mode)
        if self.merge_mode == 'add':
            self.cat3 = conv3dBlock([0], [filters2[1]], bn_mode=[bn_mode], relu_mode=[relu_mode],
                                    bn_momentum=bn_momentum)
            self.conv7 = resBlock_pni(filters2[1], filters2[1], pad_mode, bn_mode, relu_mode, init_mode, bn_momentum)
        else:
            self.cat3 = conv3dBlock([0], [filters2[1] * 2], bn_mode=[bn_mode], relu_mode=[relu_mode],
                                    bn_momentum=bn_momentum)
            self.conv7 = resBlock_pni(filters2[1] * 2, filters2[1], pad_mode, bn_mode, relu_mode, init_mode,
                                      bn_momentum)

        self.embed_out = conv3dBlock([int(filters2[0])],
                                     [int(filters2[0])],
                                     [(1, 5, 5)],
                                     [1],
                                     [(0, 2, 2)],
                                     [True],
                                     [pad_mode],
                                     [''],
                                     [relu_mode],
                                     init_mode,
                                     bn_momentum)

        self.out_put = conv3dBlock([int(filters2[0])], [out_planes], [(1, 1, 1)], init_mode=init_mode)
        self.skeleton_out_put = conv3dBlock([int(filters2[0])], [1], [(1, 1, 1)], init_mode=init_mode)
        self.distanceTransform_out_put = conv3dBlock([int(filters2[0])], [1], [(1, 1, 1)], init_mode=init_mode)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, first_stage_aff=None):
        # MOE feature selector
        if self.if_hard is True:
            x = self.feature_selector(x)
        else:
            x, x_weight = self.feature_weighting(x)
        # embedding
        embed_in = self.embed_in(x)
        conv0 = self.conv0(embed_in)
        pool0 = self.pool0(conv0)
        conv1 = self.conv1(pool0)
        pool1 = self.pool1(conv1)
        conv2 = self.conv2(pool1)
        pool2 = self.pool2(conv2)
        conv3 = self.conv3(pool2)
        pool3 = self.pool3(conv3)

        center = self.center(pool3)

        up0 = self.up0(center)
        if self.merge_mode == 'add':
            cat0 = self.cat0(up0 + conv3)
        else:
            cat0 = self.cat0(torch.cat([up0, conv3], dim=1))
        conv4 = self.conv4(cat0)

        up1 = self.up1(conv4)
        if self.merge_mode == 'add':
            cat1 = self.cat1(up1 + conv2)
        else:
            cat1 = self.cat1(torch.cat([up1, conv2], dim=1))
        conv5 = self.conv5(cat1)

        up2 = self.up2(conv5)
        if self.merge_mode == 'add':
            cat2 = self.cat2(up2 + conv1)
        else:
            cat2 = self.cat2(torch.cat([up2, conv1], dim=1))
        conv6 = self.conv6(cat2)

        up3 = self.up3(conv6)
        if self.merge_mode == 'add':
            cat3 = self.cat3(up3 + conv0)
        else:
            cat3 = self.cat3(torch.cat([up3, conv0], dim=1))
        conv7 = self.conv7(cat3)

        embed_out = self.embed_out(conv7)
        if self.if_only_skele == True:
            out = self.skeleton_out_put(embed_out)
        else:
            if self.if_skele is True and self.if_dt is False:
                out_skele = self.skeleton_out_put(embed_out)
                out_aff = self.out_put(embed_out)
                out = torch.cat((out_skele, out_aff), dim=1)  # skele=out[:,:1] aff=out[:,1:]
            elif self.if_skele is False and self.if_dt is True:
                out_dt = self.distanceTransform_out_put(embed_out)
                out_aff = self.out_put(embed_out)
                out = torch.cat((out_dt, out_aff), dim=1)  # dt=out[:,:1]  aff=out[:,1:]
            elif self.if_skele is True and self.if_dt is True:
                out_skele = self.skeleton_out_put(embed_out)
                out_dt = self.distanceTransform_out_put(embed_out)
                out_aff = self.out_put(embed_out)
                out = torch.cat((out_skele, out_dt, out_aff), dim=1) # skele=out[:,:1] dt=out[:,1:2] aff=out[:,2:]
            else:
                out = self.out_put(embed_out)

        if self.if_residual and first_stage_aff is not None:
            # residual_out = self.residual_conv(first_stage_aff)
            residual_out = first_stage_aff
            out = out + residual_out  # 残差连接

        if self.if_sigmoid:
            # if self.if_skele is True:
            #     out[:, :1] = torch.sigmoid(out_skele)
            #     out[:, 1:] = torch.sigmoid(out_aff)
            # else:
            #     out = torch.sigmoid(out)
            out = torch.sigmoid(out)

        if self.show_feature:
            down_features = [conv0, conv1, conv2, conv3]
            center_features = [center]
            up_features = [conv4, conv5, conv6, conv7]
            return down_features, center_features, up_features, out
        else:
            return out

class UNet_PNI_Cascaded(nn.Module):
    def __init__(self, in_planes=1,
                 out_planes=3,
                 filters=[28, 36, 48, 64, 80],
                 upsample_mode='transposeS',
                 decode_ratio=1,
                 merge_mode='cat',
                 pad_mode='zero',
                 bn_mode='async',
                 relu_mode='elu',
                 init_mode='kaiming_normal',
                 bn_momentum=0.001,
                 do_embed=True,
                 if_sigmoid=True,
                 show_feature=False,
                 if_skele=False,
                 if_dt=False,
                 if_only_skele=False,
                 if_residual=True):  # 新增残差学习标志
        
        super(UNet_PNI_Cascaded, self).__init__()
        filters2 = filters[:1] + filters
        self.merge_mode = merge_mode
        self.do_embed = do_embed
        self.depth = len(filters2) - 2
        self.if_sigmoid = if_sigmoid
        self.show_feature = show_feature
        self.if_skele = if_skele
        self.if_dt = if_dt
        self.if_only_skele = if_only_skele
        self.if_residual = if_residual
        
        # 自适应特征选择器 - 替换原来的FeatureSelector
        self.feature_selector = FusedFeatureSelector()
        
        # 残差连接层 - 用于第一阶段affinity的残差学习
        # if self.if_residual:
        #     self.residual_conv = conv3dBlock([out_planes], [out_planes], [(1, 1, 1)], 
        #                                    init_mode=init_mode)

        # 2D conv for anisotropic - 修改输入通道数以适应融合后的特征
        self.embed_in = conv3dBlock([1],  # 融合后是单通道
                                    [filters2[0]],
                                    [(1, 5, 5)],
                                    [1],
                                    [(0, 2, 2)],
                                    [True],
                                    [pad_mode],
                                    [''],
                                    [relu_mode],
                                    init_mode,
                                    bn_momentum)

        # 保持原有的网络结构
        # downsample stream
        self.conv0 = resBlock_pni(filters2[0], filters2[1], pad_mode, bn_mode, relu_mode, init_mode, bn_momentum)
        self.pool0 = nn.MaxPool3d((1, 2, 2), (1, 2, 2))

        self.conv1 = resBlock_pni(filters2[1], filters2[2], pad_mode, bn_mode, relu_mode, init_mode, bn_momentum)
        self.pool1 = nn.MaxPool3d((1, 2, 2), (1, 2, 2))

        self.conv2 = resBlock_pni(filters2[2], filters2[3], pad_mode, bn_mode, relu_mode, init_mode, bn_momentum)
        self.pool2 = nn.MaxPool3d((1, 2, 2), (1, 2, 2))

        self.conv3 = resBlock_pni(filters2[3], filters2[4], pad_mode, bn_mode, relu_mode, init_mode, bn_momentum)
        self.pool3 = nn.MaxPool3d((1, 2, 2), (1, 2, 2))

        self.center = resBlock_pni(filters2[4], filters2[5], pad_mode, bn_mode, relu_mode, init_mode, bn_momentum)

        # upsample stream
        self.up0 = upsampleBlock(filters2[5], filters2[4], (1, 2, 2), upsample_mode, init_mode=init_mode)
        if self.merge_mode == 'add':
            self.cat0 = conv3dBlock([0], [filters2[4]], bn_mode=[bn_mode], relu_mode=[relu_mode],
                                    bn_momentum=bn_momentum)
            self.conv4 = resBlock_pni(filters2[4], filters2[4], pad_mode, bn_mode, relu_mode, init_mode, bn_momentum)
        else:
            self.cat0 = conv3dBlock([0], [filters2[4] * 2], bn_mode=[bn_mode], relu_mode=[relu_mode],
                                    bn_momentum=bn_momentum)
            self.conv4 = resBlock_pni(filters2[4] * 2, filters2[4], pad_mode, bn_mode, relu_mode, init_mode,
                                      bn_momentum)

        self.up1 = upsampleBlock(filters2[4], filters2[3], (1, 2, 2), upsample_mode, init_mode=init_mode)
        if self.merge_mode == 'add':
            self.cat1 = conv3dBlock([0], [filters2[3]], bn_mode=[bn_mode], relu_mode=[relu_mode],
                                    bn_momentum=bn_momentum)
            self.conv5 = resBlock_pni(filters2[3], filters2[3], pad_mode, bn_mode, relu_mode, init_mode, bn_momentum)
        else:
            self.cat1 = conv3dBlock([0], [filters2[3] * 2], bn_mode=[bn_mode], relu_mode=[relu_mode],
                                    bn_momentum=bn_momentum)
            self.conv5 = resBlock_pni(filters2[3] * 2, filters2[3], pad_mode, bn_mode, relu_mode, init_mode,
                                      bn_momentum)

        self.up2 = upsampleBlock(filters2[3], filters2[2], (1, 2, 2), upsample_mode, init_mode=init_mode)
        if self.merge_mode == 'add':
            self.cat2 = conv3dBlock([0], [filters2[2]], bn_mode=[bn_mode], relu_mode=[relu_mode],
                                    bn_momentum=bn_momentum)
            self.conv6 = resBlock_pni(filters2[2], filters2[2], pad_mode, bn_mode, relu_mode, init_mode, bn_momentum)
        else:
            self.cat2 = conv3dBlock([0], [filters2[2] * 2], bn_mode=[bn_mode], relu_mode=[relu_mode],
                                    bn_momentum=bn_momentum)
            self.conv6 = resBlock_pni(filters2[2] * 2, filters2[2], pad_mode, bn_mode, relu_mode, init_mode,
                                      bn_momentum)

        self.up3 = upsampleBlock(filters2[2], filters2[1], (1, 2, 2), upsample_mode, init_mode=init_mode)
        if self.merge_mode == 'add':
            self.cat3 = conv3dBlock([0], [filters2[1]], bn_mode=[bn_mode], relu_mode=[relu_mode],
                                    bn_momentum=bn_momentum)
            self.conv7 = resBlock_pni(filters2[1], filters2[1], pad_mode, bn_mode, relu_mode, init_mode, bn_momentum)
        else:
            self.cat3 = conv3dBlock([0], [filters2[1] * 2], bn_mode=[bn_mode], relu_mode=[relu_mode],
                                    bn_momentum=bn_momentum)
            self.conv7 = resBlock_pni(filters2[1] * 2, filters2[1], pad_mode, bn_mode, relu_mode, init_mode,
                                      bn_momentum)

        self.embed_out = conv3dBlock([int(filters2[0])],
                                     [int(filters2[0])],
                                     [(1, 5, 5)],
                                     [1],
                                     [(0, 2, 2)],
                                     [True],
                                     [pad_mode],
                                     [''],
                                     [relu_mode],
                                     init_mode,
                                     bn_momentum)

        self.out_put = conv3dBlock([int(filters2[0])], [out_planes], [(1, 1, 1)], init_mode=init_mode)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, first_stage_aff=None):
        """
        Args:
            x: [B, 3, D, H, W] 包含skeleton, distance_transform, raw_image
            first_stage_aff: [B, 3, D, H, W] 第一阶段的affinity输出，用于残差学习
        """
        # 使用自适应特征选择器融合特征
        fused_feature = self.feature_selector(x)  # [B, 1, D, H, W]
        
        # embedding
        embed_in = self.embed_in(fused_feature)
        conv0 = self.conv0(embed_in)
        pool0 = self.pool0(conv0)
        conv1 = self.conv1(pool0)
        pool1 = self.pool1(conv1)
        conv2 = self.conv2(pool1)
        pool2 = self.pool2(conv2)
        conv3 = self.conv3(pool2)
        pool3 = self.pool3(conv3)

        center = self.center(pool3)

        up0 = self.up0(center)
        if self.merge_mode == 'add':
            cat0 = self.cat0(up0 + conv3)
        else:
            cat0 = self.cat0(torch.cat([up0, conv3], dim=1))
        conv4 = self.conv4(cat0)

        up1 = self.up1(conv4)
        if self.merge_mode == 'add':
            cat1 = self.cat1(up1 + conv2)
        else:
            cat1 = self.cat1(torch.cat([up1, conv2], dim=1))
        conv5 = self.conv5(cat1)

        up2 = self.up2(conv5)
        if self.merge_mode == 'add':
            cat2 = self.cat2(up2 + conv1)
        else:
            cat2 = self.cat2(torch.cat([up2, conv1], dim=1))
        conv6 = self.conv6(cat2)

        up3 = self.up3(conv6)
        if self.merge_mode == 'add':
            cat3 = self.cat3(up3 + conv0)
        else:
            cat3 = self.cat3(torch.cat([up3, conv0], dim=1))
        conv7 = self.conv7(cat3)

        embed_out = self.embed_out(conv7)
        out = self.out_put(embed_out)
        
        # 残差学习：如果提供了第一阶段的affinity，进行残差连接
        if self.if_residual and first_stage_aff is not None:
            # residual_out = self.residual_conv(first_stage_aff)
            residual_out = first_stage_aff
            out = out + residual_out  # 残差连接

        if self.if_sigmoid:
            out = torch.sigmoid(out)

        if self.show_feature:
            down_features = [conv0, conv1, conv2, conv3]
            center_features = [center]
            up_features = [conv4, conv5, conv6, conv7]
            return down_features, center_features, up_features, out
        else:
            return out

class AdaptiveKFeatureSelector(nn.Module):
    def __init__(self):
        super().__init__()
        
        # K值预测网络：分析三个特征决定每个位置需要几个特征
        self.k_predictor = nn.Sequential(
            nn.Conv3d(3, 16, kernel_size=3, padding=1),  # 输入3个特征
            nn.ReLU(),
            nn.Conv3d(16, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv3d(8, 3, kernel_size=1)  # 输出3个K值的概率 (K=1,2,3)
        )
        
        # 特征重要性评估网络：为每个特征打分
        self.feature_scorer = nn.Sequential(
            nn.Conv3d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv3d(16, 3, kernel_size=1)  # 为3个特征分别打分
        )
        
        # 特征融合网络：将选中的特征融合
        self.feature_fusion = nn.Sequential(
            nn.Conv3d(3, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv3d(8, 1, kernel_size=1)  # 输出融合后的单一特征
        )

    def forward(self, raw_image, skeleton, distance_transform):
        """
        Args:
            raw_image: [B, 1, D, H, W]
            skeleton: [B, 1, D, H, W] 
            distance_transform: [B, 1, D, H, W]
        Returns:
            fused_feature: [B, 1, D, H, W]
            k_distribution: [B, 3, D, H, W] - K值选择概率
        """
        # 1. 合并输入特征用于分析
        combined_features = torch.cat([raw_image, skeleton, distance_transform], dim=1)  # [B, 3, D, H, W]
        
        # 2. 预测每个位置的最优K值
        k_logits = self.k_predictor(combined_features)  # [B, 3, D, H, W]
        k_probs = F.softmax(k_logits, dim=1)  # K=1,2,3的概率分布
        
        # 3. 计算每个特征的重要性分数
        feature_scores = self.feature_scorer(combined_features)  # [B, 3, D, H, W]
        feature_weights = F.softmax(feature_scores, dim=1)  # 归一化权重
        
        # 4. 自适应特征选择和融合
        selected_features = self.adaptive_feature_selection(
            combined_features, k_probs, feature_weights
        )
        
        # 5. 最终融合
        fused_feature = self.feature_fusion(selected_features)
        
        return fused_feature, k_probs

    def adaptive_feature_selection(self, features, k_probs, feature_weights):
        """
        根据K值概率和特征权重进行自适应选择
        """
        B, C, D, H, W = features.shape
        
        # 为不同K值计算选择结果
        # K=1: 选择权重最高的单个特征
        max_weight_idx = torch.argmax(feature_weights, dim=1, keepdim=True)  # [B, 1, D, H, W]
        selected_k1 = torch.gather(features, 1, max_weight_idx).expand(-1, 3, -1, -1, -1)
        
        # K=2: 选择权重最高的两个特征
        top2_weights, top2_indices = torch.topk(feature_weights, 2, dim=1)  # [B, 2, D, H, W]
        top2_weights = F.softmax(top2_weights, dim=1)  # 重新归一化
        
        selected_k2 = torch.zeros_like(features)
        selected_k2.scatter_(1, top2_indices, torch.gather(features, 1, top2_indices) * top2_weights)
        
        # K=3: 使用所有特征
        selected_k3 = features * feature_weights
        
        # 根据K值概率进行加权组合
        final_selection = (
            k_probs[:, 0:1] * selected_k1 +     # K=1的贡献
            k_probs[:, 1:2] * selected_k2 +     # K=2的贡献  
            k_probs[:, 2:3] * selected_k3       # K=3的贡献
        )
        
        return final_selection


# 在 FeatureSelector 的 forward 方法中，确保正确保存 k_distribution
class FusedFeatureSelector(nn.Module):
    def __init__(self, input_dim=None, num_features=None):
        """保持接口兼容性"""
        super().__init__()
        self.adaptive_selector = AdaptiveKFeatureSelector()
        self.last_k_distribution = None  # 初始化
        
    def forward(self, x):
        """
        x: [B, 3, D, H, W] 其中包含skeleton, distance, raw_image
        """
        # 分离三个特征
        skeleton = x[:, 0:1]      # [B, 1, D, H, W]
        distance = x[:, 1:2]      # [B, 1, D, H, W] 
        raw_image = x[:, 2:3]     # [B, 1, D, H, W]
        
        # 使用自适应选择器
        fused_feature, k_distribution = self.adaptive_selector(
            raw_image, skeleton, distance
        )
        
        # 保存分析数据用于可视化
        self.last_k_distribution = k_distribution.detach()  # 添加 detach() 避免梯度问题
        
        return fused_feature  # [B, 1, D, H, W]

class FeatureSelectorConvAdd(nn.Module):
    def __init__(self, input_dim, num_features):
        """
        input_dim: 每个特征通道的输入维度（D）
        num_features: 特征通道数（C）
        """
        super(FeatureSelectorConvAdd, self).__init__()
        self.num_features = num_features
        self.fusion_conv = nn.Conv3d(2, 1, kernel_size=1, bias=False)
        self.scoring_layer = nn.Linear(input_dim, num_features)  # 计算每个特征的得分

    def forward(self, x):
        """
        x: [B, C, D, H, W] 输入特征
        返回:
            top2_features: [B, 2, D, H, W] 选择的 Top-2 特征
        """
        B, C, D, H, W = x.shape
        assert C == self.num_features, "输入的特征通道数应与 num_features 匹配"

        # Step 1: 计算特征得分（对每个 C 通道计算一个得分）
        x_flat = x.mean(dim=[2, 3, 4])  # [B, C]，对 D, H, W 维度求均值，得到每个特征通道的全局信息
        scores = self.scoring_layer(x_flat)  # [B, C]
        scores = F.softmax(scores, dim=-1)  # 归一化得分

        # Step 2: 选择 Top-2 特征通道
        top2_scores, top2_indices = torch.topk(scores, k=2, dim=-1)  # [B, 2]

        # Step 3: 选择对应的特征通道
        top2_features = torch.zeros(B, 2, D, H, W, device=x.device)
        for b in range(B):
            top2_features[b] = x[b, top2_indices[b]]  # 选择对应的通道


        # 在forward中使用
        fused_features = self.fusion_conv(top2_features)  # [B, 1, D, H, W]

        return fused_features  # [B, 2, D, H, W]

class FeatureSelector(nn.Module):
    def __init__(self, input_dim, num_features):
        """
        input_dim: 每个特征通道的输入维度（D）
        num_features: 特征通道数（C）
        """
        super(FeatureSelector, self).__init__()
        self.num_features = num_features
        self.scoring_layer = nn.Linear(input_dim, num_features)  # 计算每个特征的得分

    def forward(self, x):
        """
        x: [B, C, D, H, W] 输入特征
        返回:
            top2_features: [B, 2, D, H, W] 选择的 Top-2 特征
        """
        B, C, D, H, W = x.shape
        assert C == self.num_features, "输入的特征通道数应与 num_features 匹配"

        # Step 1: 计算特征得分（对每个 C 通道计算一个得分）
        x_flat = x.mean(dim=[2, 3, 4])  # [B, C]，对 D, H, W 维度求均值，得到每个特征通道的全局信息
        scores = self.scoring_layer(x_flat)  # [B, C]
        scores = F.softmax(scores, dim=-1)  # 归一化得分

        # Step 2: 选择 Top-2 特征通道
        top2_scores, top2_indices = torch.topk(scores, k=2, dim=-1)  # [B, 2]

        # Step 3: 选择对应的特征通道
        top2_features = torch.zeros(B, 2, D, H, W, device=x.device)
        for b in range(B):
            top2_features[b] = x[b, top2_indices[b]]  # 选择对应的通道

        # self.fusion_conv = nn.Conv3d(2, 1, kernel_size=1, bias=False)

        # # 在forward中使用
        # fused_features = self.fusion_conv(top2_features)  # [B, 1, D, H, W]

        return top2_features  # [B, 2, D, H, W]
    
        # Step 2: 选择 Top-1 特征通道
        top1_scores, top1_indices = torch.topk(scores, k=1, dim=-1)  # [B, 1]

        # Step 3: 选择对应的特征通道
        top1_features = x.gather(1, top1_indices.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand(-1, -1, D, H, W))

        return top1_features  # [B, 1, D, H, W]

class FeatureWeighting(nn.Module):
    def __init__(self, input_dim, num_features):
        """
        input_dim: 每个特征通道的输入维度（D）
        num_features: 特征通道数（C）
        """
        super(FeatureWeighting, self).__init__()
        self.num_features = num_features
        self.weight_layer = nn.Linear(input_dim, num_features)  # 计算每个特征的权重

    def forward(self, x):
        """
        x: [B, C, D, H, W] 输入特征
        返回:
            weighted_x: [B, C, D, H, W] 加权后的特征
            weights: [B, C] 每个特征通道的权重
        """
        B, C, D, H, W = x.shape
        assert C == self.num_features, "输入的特征通道数应与 num_features 匹配"

        # Step 1: 计算特征通道的全局信息（均值池化）
        x_flat = x.mean(dim=[2, 3, 4])  # [B, C]，对 D, H, W 维度求均值

        # Step 2: 计算每个特征通道的权重
        weights = self.weight_layer(x_flat)  # [B, C]
        weights = F.softmax(weights, dim=-1)  # 归一化，使得权重总和为 1

        # Step 3: 对每个特征通道进行加权
        weights = weights.view(B, C, 1, 1, 1)  # 调整维度以便广播
        weighted_x = x * weights  # [B, C, D, H, W] * [B, C, 1, 1, 1] → [B, C, D, H, W]

        return weighted_x, weights  # 返回加权后的特征和权重

class UNet_PNI_Noskip(nn.Module):  # deployed PNI model
    # Superhuman Accuracy on the SNEMI3D Connectomics Challenge. Lee et al.
    # https://arxiv.org/abs/1706.00120
    def __init__(self, in_planes=1,
                 out_planes=3,
                 filters=[28, 36, 48, 64, 80],  # [28, 36, 48, 64, 80], [32, 64, 128, 256, 512]
                 upsample_mode='bilinear',
                 decode_ratio=1,
                 merge_mode='cat',
                 pad_mode='zero',
                 bn_mode='async',  # async or sync
                 relu_mode='elu',
                 init_mode='kaiming_normal',
                 bn_momentum=0.001,
                 do_embed=True,
                 if_sigmoid=True,
                 show_feature=False):
        # filter_ratio: #filter_decode/#filter_encode
        super(UNet_PNI_Noskip, self).__init__()
        filters2 = filters[:1] + filters
        self.merge_mode = merge_mode
        self.do_embed = do_embed
        self.depth = len(filters2) - 2
        self.if_sigmoid = if_sigmoid
        self.show_feature = show_feature

        # 2D conv for anisotropic
        self.embed_in = conv3dBlock([in_planes],
                                    [filters2[0]],
                                    [(1, 5, 5)],
                                    [1],
                                    [(0, 2, 2)],
                                    [True],
                                    [pad_mode],
                                    [''],
                                    [relu_mode],
                                    init_mode,
                                    bn_momentum)

        # downsample stream
        self.conv0 = resBlock_pni(filters2[0], filters2[1], pad_mode, bn_mode, relu_mode, init_mode, bn_momentum)
        self.pool0 = nn.MaxPool3d((1, 2, 2), (1, 2, 2))

        self.conv1 = resBlock_pni(filters2[1], filters2[2], pad_mode, bn_mode, relu_mode, init_mode, bn_momentum)
        self.pool1 = nn.MaxPool3d((1, 2, 2), (1, 2, 2))

        self.conv2 = resBlock_pni(filters2[2], filters2[3], pad_mode, bn_mode, relu_mode, init_mode, bn_momentum)
        self.pool2 = nn.MaxPool3d((1, 2, 2), (1, 2, 2))

        self.conv3 = resBlock_pni(filters2[3], filters2[4], pad_mode, bn_mode, relu_mode, init_mode, bn_momentum)
        self.pool3 = nn.MaxPool3d((1, 2, 2), (1, 2, 2))

        self.center = resBlock_pni(filters2[4], filters2[5], pad_mode, bn_mode, relu_mode, init_mode, bn_momentum)

        # upsample stream
        self.up0 = upsampleBlock(filters2[5], filters2[4], (1, 2, 2), upsample_mode, init_mode=init_mode)
        self.cat0 = conv3dBlock([0], [filters2[4]], bn_mode=[bn_mode], relu_mode=[relu_mode], bn_momentum=bn_momentum)
        self.conv4 = resBlock_pni(filters2[4], filters2[4], pad_mode, bn_mode, relu_mode, init_mode, bn_momentum)

        self.up1 = upsampleBlock(filters2[4], filters2[3], (1, 2, 2), upsample_mode, init_mode=init_mode)
        self.cat1 = conv3dBlock([0], [filters2[3]], bn_mode=[bn_mode], relu_mode=[relu_mode], bn_momentum=bn_momentum)
        self.conv5 = resBlock_pni(filters2[3], filters2[3], pad_mode, bn_mode, relu_mode, init_mode, bn_momentum)

        self.up2 = upsampleBlock(filters2[3], filters2[2], (1, 2, 2), upsample_mode, init_mode=init_mode)
        self.cat2 = conv3dBlock([0], [filters2[2]], bn_mode=[bn_mode], relu_mode=[relu_mode], bn_momentum=bn_momentum)
        self.conv6 = resBlock_pni(filters2[2], filters2[2], pad_mode, bn_mode, relu_mode, init_mode, bn_momentum)

        self.up3 = upsampleBlock(filters2[2], filters2[1], (1, 2, 2), upsample_mode, init_mode=init_mode)
        self.cat3 = conv3dBlock([0], [filters2[1]], bn_mode=[bn_mode], relu_mode=[relu_mode], bn_momentum=bn_momentum)
        self.conv7 = resBlock_pni(filters2[1], filters2[1], pad_mode, bn_mode, relu_mode, init_mode, bn_momentum)

        self.embed_out = conv3dBlock([int(filters2[0])],
                                     [int(filters2[0])],
                                     [(1, 5, 5)],
                                     [1],
                                     [(0, 2, 2)],
                                     [True],
                                     [pad_mode],
                                     [''],
                                     [relu_mode],
                                     init_mode,
                                     bn_momentum)

        self.out_put = conv3dBlock([int(filters2[0])], [out_planes], [(1, 1, 1)], init_mode=init_mode)
        self.sigmoid = nn.Sigmoid()

    def encoder(self, x):
        # embedding
        embed_in = self.embed_in(x)
        conv0 = self.conv0(embed_in)
        pool0 = self.pool0(conv0)
        conv1 = self.conv1(pool0)
        pool1 = self.pool1(conv1)
        conv2 = self.conv2(pool1)
        pool2 = self.pool2(conv2)
        conv3 = self.conv3(pool2)
        pool3 = self.pool3(conv3)

        center = self.center(pool3)
        down_features = [conv0, conv1, conv2, conv3]

        return center, down_features

    def decoder(self, center):
        up0 = self.up0(center)
        cat0 = self.cat0(up0)
        conv4 = self.conv4(cat0)

        up1 = self.up1(conv4)
        cat1 = self.cat1(up1)
        conv5 = self.conv5(cat1)

        up2 = self.up2(conv5)
        cat2 = self.cat2(up2)
        conv6 = self.conv6(cat2)

        up3 = self.up3(conv6)
        cat3 = self.cat3(up3)
        conv7 = self.conv7(cat3)

        embed_out = self.embed_out(conv7)
        out = self.out_put(embed_out)
        up_features = [conv4, conv5, conv6, conv7]

        return out, up_features

    def forward(self, x):
        center, down_features = self.encoder(x)
        out, up_features = self.decoder(center)

        if self.if_sigmoid:
            out = torch.sigmoid(out)

        if self.show_feature:
            center_features = [center]
            return down_features, center_features, up_features, out
        else:
            return out


class UNet_PNI_Noskip2(nn.Module):  # deployed PNI model
    # Superhuman Accuracy on the SNEMI3D Connectomics Challenge. Lee et al.
    # https://arxiv.org/abs/1706.00120
    def __init__(self, in_planes=1,
                 out_planes=3,
                 filters=[28, 36, 48, 64, 80],  # [28, 36, 48, 64, 80], [32, 64, 128, 256, 512]
                 upsample_mode='bilinear',
                 decode_ratio=1,
                 merge_mode='cat',
                 pad_mode='zero',
                 bn_mode='async',  # async or sync
                 relu_mode='elu',
                 init_mode='kaiming_normal',
                 bn_momentum=0.001,
                 do_embed=True,
                 if_sigmoid=True,
                 show_feature=False):
        # filter_ratio: #filter_decode/#filter_encode
        super(UNet_PNI_Noskip2, self).__init__()
        filters2 = filters[:1] + filters
        self.merge_mode = merge_mode
        self.do_embed = do_embed
        self.depth = len(filters2) - 2
        self.if_sigmoid = if_sigmoid
        self.show_feature = show_feature

        # 2D conv for anisotropic
        self.embed_in = conv3dBlock([in_planes],
                                    [filters2[0]],
                                    [(1, 5, 5)],
                                    [1],
                                    [(0, 2, 2)],
                                    [True],
                                    [pad_mode],
                                    [''],
                                    [relu_mode],
                                    init_mode,
                                    bn_momentum)

        # downsample stream
        self.conv0 = resBlock_pni(filters2[0], filters2[1], pad_mode, bn_mode, relu_mode, init_mode, bn_momentum)
        self.pool0 = nn.MaxPool3d((1, 2, 2), (1, 2, 2))

        self.conv1 = resBlock_pni(filters2[1], filters2[2], pad_mode, bn_mode, relu_mode, init_mode, bn_momentum)
        self.pool1 = nn.MaxPool3d((1, 2, 2), (1, 2, 2))

        self.conv2 = resBlock_pni(filters2[2], filters2[3], pad_mode, bn_mode, relu_mode, init_mode, bn_momentum)
        self.pool2 = nn.MaxPool3d((1, 2, 2), (1, 2, 2))

        self.conv3 = resBlock_pni(filters2[3], filters2[4], pad_mode, bn_mode, relu_mode, init_mode, bn_momentum)
        self.pool3 = nn.MaxPool3d((1, 2, 2), (1, 2, 2))

        self.center = resBlock_pni(filters2[4], filters2[5], pad_mode, bn_mode, relu_mode, init_mode, bn_momentum)

        # upsample stream
        self.up0 = upsampleBlock(filters2[5], filters2[4], (1, 2, 2), upsample_mode, init_mode=init_mode)
        self.cat0 = conv3dBlock([0], [filters2[4]], bn_mode=[bn_mode], relu_mode=[relu_mode], bn_momentum=bn_momentum)
        # self.conv4 = resBlock_pni(filters2[4], filters2[4], pad_mode, bn_mode, relu_mode, init_mode, bn_momentum)
        self.conv4 = conv3dBlock([filters2[4]], [filters2[4]], [(3, 3, 3)], [1], [(1, 1, 1)], [True], [pad_mode],
                                 [bn_mode], [relu_mode], init_mode, bn_momentum)

        self.up1 = upsampleBlock(filters2[4], filters2[3], (1, 2, 2), upsample_mode, init_mode=init_mode)
        self.cat1 = conv3dBlock([0], [filters2[3]], bn_mode=[bn_mode], relu_mode=[relu_mode], bn_momentum=bn_momentum)
        # self.conv5 = resBlock_pni(filters2[3], filters2[3], pad_mode, bn_mode, relu_mode, init_mode, bn_momentum)
        self.conv5 = conv3dBlock([filters2[3]], [filters2[3]], [(3, 3, 3)], [1], [(1, 1, 1)], [True], [pad_mode],
                                 [bn_mode], [relu_mode], init_mode, bn_momentum)

        self.up2 = upsampleBlock(filters2[3], filters2[2], (1, 2, 2), upsample_mode, init_mode=init_mode)
        self.cat2 = conv3dBlock([0], [filters2[2]], bn_mode=[bn_mode], relu_mode=[relu_mode], bn_momentum=bn_momentum)
        # self.conv6 = resBlock_pni(filters2[2], filters2[2], pad_mode, bn_mode, relu_mode, init_mode, bn_momentum)
        self.conv6 = conv3dBlock([filters2[2]], [filters2[2]], [(3, 3, 3)], [1], [(1, 1, 1)], [True], [pad_mode],
                                 [bn_mode], [relu_mode], init_mode, bn_momentum)

        self.up3 = upsampleBlock(filters2[2], filters2[1], (1, 2, 2), upsample_mode, init_mode=init_mode)
        self.cat3 = conv3dBlock([0], [filters2[1]], bn_mode=[bn_mode], relu_mode=[relu_mode], bn_momentum=bn_momentum)
        # self.conv7 = resBlock_pni(filters2[1], filters2[1], pad_mode, bn_mode, relu_mode, init_mode, bn_momentum)
        self.conv7 = conv3dBlock([filters2[1]], [filters2[1]], [(3, 3, 3)], [1], [(1, 1, 1)], [True], [pad_mode],
                                 [bn_mode], [relu_mode], init_mode, bn_momentum)

        self.embed_out = conv3dBlock([int(filters2[0])],
                                     [int(filters2[0])],
                                     [(1, 5, 5)],
                                     [1],
                                     [(0, 2, 2)],
                                     [True],
                                     [pad_mode],
                                     [''],
                                     [relu_mode],
                                     init_mode,
                                     bn_momentum)

        self.out_put = conv3dBlock([int(filters2[0])], [out_planes], [(1, 1, 1)], init_mode=init_mode)
        self.sigmoid = nn.Sigmoid()

    def encoder(self, x):
        # embedding
        embed_in = self.embed_in(x)
        conv0 = self.conv0(embed_in)
        pool0 = self.pool0(conv0)
        conv1 = self.conv1(pool0)
        pool1 = self.pool1(conv1)
        conv2 = self.conv2(pool1)
        pool2 = self.pool2(conv2)
        conv3 = self.conv3(pool2)
        pool3 = self.pool3(conv3)

        center = self.center(pool3)
        down_features = [conv0, conv1, conv2, conv3]

        return center, down_features

    def decoder(self, center):
        up0 = self.up0(center)
        cat0 = self.cat0(up0)
        conv4 = self.conv4(cat0)

        up1 = self.up1(conv4)
        cat1 = self.cat1(up1)
        conv5 = self.conv5(cat1)

        up2 = self.up2(conv5)
        cat2 = self.cat2(up2)
        conv6 = self.conv6(cat2)

        up3 = self.up3(conv6)
        cat3 = self.cat3(up3)
        conv7 = self.conv7(cat3)

        embed_out = self.embed_out(conv7)
        out = self.out_put(embed_out)
        up_features = [conv4, conv5, conv6, conv7]

        return out, up_features

    def forward(self, x):
        center, down_features = self.encoder(x)
        out, up_features = self.decoder(center)

        if self.if_sigmoid:
            out = torch.sigmoid(out)

        if self.show_feature:
            center_features = [center]
            return down_features, center_features, up_features, out
        else:
            return out


class UNet_PNI_FT1(nn.Module):  # deployed PNI model
    # Superhuman Accuracy on the SNEMI3D Connectomics Challenge. Lee et al.
    # https://arxiv.org/abs/1706.00120
    def __init__(self, pretrained_model,
                 out_planes=3,
                 filters=[28, 36, 48, 64, 80],  # [28, 36, 48, 64, 80], [32, 64, 128, 256, 512]
                 upsample_mode='transposeS',
                 decode_ratio=1,
                 merge_mode='cat',
                 pad_mode='zero',
                 bn_mode='async',  # async or sync
                 relu_mode='elu',
                 init_mode='kaiming_normal',
                 bn_momentum=0.001,
                 do_embed=True,
                 if_sigmoid=True,
                 show_feature=False):
        # filter_ratio: #filter_decode/#filter_encode
        super(UNet_PNI_FT1, self).__init__()
        self.pretrained_model = pretrained_model
        filters2 = filters[:1] + filters
        self.merge_mode = merge_mode
        self.do_embed = do_embed
        self.depth = len(filters2) - 2
        self.if_sigmoid = if_sigmoid
        self.show_feature = show_feature

        self.center_ft = resBlock_pni(filters2[5], filters2[5], pad_mode, bn_mode, relu_mode, init_mode, bn_momentum)

        # upsample stream
        self.up0_ft = upsampleBlock(filters2[5], filters2[4], (1, 2, 2), upsample_mode, init_mode=init_mode)
        if self.merge_mode == 'add' or self.merge_mode == 'add2' or self.merge_mode == 'add3':
            self.cat0_ft = conv3dBlock([0], [filters2[4]], bn_mode=[bn_mode], relu_mode=[relu_mode],
                                       bn_momentum=bn_momentum)
            self.conv4_ft = resBlock_pni(filters2[4], filters2[4], pad_mode, bn_mode, relu_mode, init_mode, bn_momentum)
        elif self.merge_mode == 'cat' or self.merge_mode == 'cat2':
            self.cat0_ft = conv3dBlock([0], [filters2[4] * 2], bn_mode=[bn_mode], relu_mode=[relu_mode],
                                       bn_momentum=bn_momentum)
            self.conv4_ft = resBlock_pni(filters2[4] * 2, filters2[4], pad_mode, bn_mode, relu_mode, init_mode,
                                         bn_momentum)
        elif self.merge_mode == 'cat3':
            self.cat0_ft = conv3dBlock([0], [filters2[4] * 3], bn_mode=[bn_mode], relu_mode=[relu_mode],
                                       bn_momentum=bn_momentum)
            self.conv4_ft = resBlock_pni(filters2[4] * 3, filters2[4], pad_mode, bn_mode, relu_mode, init_mode,
                                         bn_momentum)
        else:
            raise AttributeError('No this merge mode!')

        self.up1_ft = upsampleBlock(filters2[4], filters2[3], (1, 2, 2), upsample_mode, init_mode=init_mode)
        if self.merge_mode == 'add' or self.merge_mode == 'add2' or self.merge_mode == 'add3':
            self.cat1_ft = conv3dBlock([0], [filters2[3]], bn_mode=[bn_mode], relu_mode=[relu_mode],
                                       bn_momentum=bn_momentum)
            self.conv5_ft = resBlock_pni(filters2[3], filters2[3], pad_mode, bn_mode, relu_mode, init_mode, bn_momentum)
        elif self.merge_mode == 'cat' or self.merge_mode == 'cat2':
            self.cat1_ft = conv3dBlock([0], [filters2[3] * 2], bn_mode=[bn_mode], relu_mode=[relu_mode],
                                       bn_momentum=bn_momentum)
            self.conv5_ft = resBlock_pni(filters2[3] * 2, filters2[3], pad_mode, bn_mode, relu_mode, init_mode,
                                         bn_momentum)
        elif self.merge_mode == 'cat3':
            self.cat1_ft = conv3dBlock([0], [filters2[3] * 3], bn_mode=[bn_mode], relu_mode=[relu_mode],
                                       bn_momentum=bn_momentum)
            self.conv5_ft = resBlock_pni(filters2[3] * 3, filters2[3], pad_mode, bn_mode, relu_mode, init_mode,
                                         bn_momentum)
        else:
            raise AttributeError('No this merge mode!')

        self.up2_ft = upsampleBlock(filters2[3], filters2[2], (1, 2, 2), upsample_mode, init_mode=init_mode)
        if self.merge_mode == 'add' or self.merge_mode == 'add2' or self.merge_mode == 'add3':
            self.cat2_ft = conv3dBlock([0], [filters2[2]], bn_mode=[bn_mode], relu_mode=[relu_mode],
                                       bn_momentum=bn_momentum)
            self.conv6_ft = resBlock_pni(filters2[2], filters2[2], pad_mode, bn_mode, relu_mode, init_mode, bn_momentum)
        elif self.merge_mode == 'cat' or self.merge_mode == 'cat2':
            self.cat2_ft = conv3dBlock([0], [filters2[2] * 2], bn_mode=[bn_mode], relu_mode=[relu_mode],
                                       bn_momentum=bn_momentum)
            self.conv6_ft = resBlock_pni(filters2[2] * 2, filters2[2], pad_mode, bn_mode, relu_mode, init_mode,
                                         bn_momentum)
        elif self.merge_mode == 'cat3':
            self.cat2_ft = conv3dBlock([0], [filters2[2] * 3], bn_mode=[bn_mode], relu_mode=[relu_mode],
                                       bn_momentum=bn_momentum)
            self.conv6_ft = resBlock_pni(filters2[2] * 3, filters2[2], pad_mode, bn_mode, relu_mode, init_mode,
                                         bn_momentum)
        else:
            raise AttributeError('No this merge mode!')

        self.up3_ft = upsampleBlock(filters2[2], filters2[1], (1, 2, 2), upsample_mode, init_mode=init_mode)
        if self.merge_mode == 'add' or self.merge_mode == 'add2' or self.merge_mode == 'add3':
            self.cat3_ft = conv3dBlock([0], [filters2[1]], bn_mode=[bn_mode], relu_mode=[relu_mode],
                                       bn_momentum=bn_momentum)
            self.conv7_ft = resBlock_pni(filters2[1], filters2[1], pad_mode, bn_mode, relu_mode, init_mode, bn_momentum)
        elif self.merge_mode == 'cat' or self.merge_mode == 'cat2':
            self.cat3_ft = conv3dBlock([0], [filters2[1] * 2], bn_mode=[bn_mode], relu_mode=[relu_mode],
                                       bn_momentum=bn_momentum)
            self.conv7_ft = resBlock_pni(filters2[1] * 2, filters2[1], pad_mode, bn_mode, relu_mode, init_mode,
                                         bn_momentum)
        elif self.merge_mode == 'cat3':
            self.cat3_ft = conv3dBlock([0], [filters2[1] * 3], bn_mode=[bn_mode], relu_mode=[relu_mode],
                                       bn_momentum=bn_momentum)
            self.conv7_ft = resBlock_pni(filters2[1] * 3, filters2[1], pad_mode, bn_mode, relu_mode, init_mode,
                                         bn_momentum)
        else:
            raise AttributeError('No this merge mode!')

        self.embed_out_ft = conv3dBlock([int(filters2[0])],
                                        [int(filters2[0])],
                                        [(1, 5, 5)],
                                        [1],
                                        [(0, 2, 2)],
                                        [True],
                                        [pad_mode],
                                        [''],
                                        [relu_mode],
                                        init_mode,
                                        bn_momentum)

        self.out_put_ft = conv3dBlock([int(filters2[0])], [out_planes], [(1, 1, 1)], init_mode=init_mode)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        with torch.no_grad():
            down_features, center_features, up_features, _ = self.pretrained_model(x)
            pre_conv0, pre_conv1, pre_conv2, pre_conv3 = down_features
            pre_center = center_features[0]
            pre_conv4, pre_conv5, pre_conv6, pre_conv7 = up_features

        center_ft = self.center_ft(pre_center)

        up0_ft = self.up0_ft(center_ft)
        if self.merge_mode == 'add' or self.merge_mode == 'add2':
            cat0_ft = self.cat0_ft(up0_ft + pre_conv4)
        elif self.merge_mode == 'add3':
            cat0_ft = self.cat0_ft(up0_ft + pre_conv4 + pre_conv3)
        elif self.merge_mode == 'cat' or self.merge_mode == 'cat2':
            cat0_ft = self.cat0_ft(torch.cat([up0_ft, pre_conv4], dim=1))
        else:
            cat0_ft = self.cat0_ft(torch.cat([up0_ft, pre_conv4, pre_conv3], dim=1))
        conv4_ft = self.conv4_ft(cat0_ft)

        up1_ft = self.up1_ft(conv4_ft)
        if self.merge_mode == 'add' or self.merge_mode == 'add2':
            cat1_ft = self.cat1_ft(up1_ft + pre_conv5)
        elif self.merge_mode == 'add3':
            cat1_ft = self.cat1_ft(up1_ft + pre_conv5 + pre_conv2)
        elif self.merge_mode == 'cat' or self.merge_mode == 'cat2':
            cat1_ft = self.cat1_ft(torch.cat([up1_ft, pre_conv5], dim=1))
        else:
            cat1_ft = self.cat1_ft(torch.cat([up1_ft, pre_conv5, pre_conv2], dim=1))
        conv5_ft = self.conv5_ft(cat1_ft)

        up2_ft = self.up2_ft(conv5_ft)
        if self.merge_mode == 'add' or self.merge_mode == 'add2':
            cat2_ft = self.cat2_ft(up2_ft + pre_conv6)
        elif self.merge_mode == 'add3':
            cat2_ft = self.cat2_ft(up2_ft + pre_conv6 + pre_conv1)
        elif self.merge_mode == 'cat' or self.merge_mode == 'cat2':
            cat2_ft = self.cat2_ft(torch.cat([up2_ft, pre_conv6], dim=1))
        else:
            cat2_ft = self.cat2_ft(torch.cat([up2_ft, pre_conv6, pre_conv1], dim=1))
        conv6_ft = self.conv6_ft(cat2_ft)

        up3_ft = self.up3_ft(conv6_ft)
        if self.merge_mode == 'add' or self.merge_mode == 'add2':
            cat3_ft = self.cat3_ft(up3_ft + pre_conv7)
        elif self.merge_mode == 'add3':
            cat3_ft = self.cat3_ft(up3_ft + pre_conv7 + pre_conv0)
        elif self.merge_mode == 'cat' or self.merge_mode == 'cat2':
            cat3_ft = self.cat3_ft(torch.cat([up3_ft, pre_conv7], dim=1))
        else:
            cat3_ft = self.cat3_ft(torch.cat([up3_ft, pre_conv7, pre_conv0], dim=1))
        conv7_ft = self.conv7_ft(cat3_ft)

        embed_out_ft = self.embed_out_ft(conv7_ft)
        out_ft = self.out_put_ft(embed_out_ft)

        if self.if_sigmoid:
            out_ft = torch.sigmoid(out_ft)

        if self.show_feature:
            center_features_ft = [center_ft]
            up_features_ft = [conv4_ft, conv5_ft, conv6_ft, conv7_ft]
            return down_features, center_features_ft, up_features_ft, out_ft
        else:
            return out_ft


class UNet_PNI_FT2(nn.Module):  # deployed PNI model
    # Superhuman Accuracy on the SNEMI3D Connectomics Challenge. Lee et al.
    # https://arxiv.org/abs/1706.00120
    def __init__(self, in_planes=1,
                 out_planes=3,
                 filters=[28, 36, 48, 64, 80],  # [28, 36, 48, 64, 80], [32, 64, 128, 256, 512]
                 upsample_mode='bilinear',
                 decode_ratio=1,
                 merge_mode='cat',
                 pad_mode='zero',
                 bn_mode='async',  # async or sync
                 relu_mode='elu',
                 init_mode='kaiming_normal',
                 bn_momentum=0.001,
                 do_embed=True,
                 if_sigmoid=True,
                 show_feature=False,
                 encoder_update=False):
        # filter_ratio: #filter_decode/#filter_encode
        super(UNet_PNI_FT2, self).__init__()
        filters2 = filters[:1] + filters
        self.merge_mode = merge_mode
        self.do_embed = do_embed
        self.depth = len(filters2) - 2
        self.if_sigmoid = if_sigmoid
        self.show_feature = show_feature
        self.encoder_update = encoder_update

        # 2D conv for anisotropic
        self.embed_in = conv3dBlock([in_planes],
                                    [filters2[0]],
                                    [(1, 5, 5)],
                                    [1],
                                    [(0, 2, 2)],
                                    [True],
                                    [pad_mode],
                                    [''],
                                    [relu_mode],
                                    init_mode,
                                    bn_momentum)

        # downsample stream
        self.conv0 = resBlock_pni(filters2[0], filters2[1], pad_mode, bn_mode, relu_mode, init_mode, bn_momentum)
        self.pool0 = nn.MaxPool3d((1, 2, 2), (1, 2, 2))

        self.conv1 = resBlock_pni(filters2[1], filters2[2], pad_mode, bn_mode, relu_mode, init_mode, bn_momentum)
        self.pool1 = nn.MaxPool3d((1, 2, 2), (1, 2, 2))

        self.conv2 = resBlock_pni(filters2[2], filters2[3], pad_mode, bn_mode, relu_mode, init_mode, bn_momentum)
        self.pool2 = nn.MaxPool3d((1, 2, 2), (1, 2, 2))

        self.conv3 = resBlock_pni(filters2[3], filters2[4], pad_mode, bn_mode, relu_mode, init_mode, bn_momentum)
        self.pool3 = nn.MaxPool3d((1, 2, 2), (1, 2, 2))

        self.center = resBlock_pni(filters2[4], filters2[5], pad_mode, bn_mode, relu_mode, init_mode, bn_momentum)

        # upsample stream
        self.up0 = upsampleBlock(filters2[5], filters2[4], (1, 2, 2), upsample_mode, init_mode=init_mode)
        if self.merge_mode == 'add':
            self.cat0 = conv3dBlock([0], [filters2[4]], bn_mode=[bn_mode], relu_mode=[relu_mode],
                                    bn_momentum=bn_momentum)
            self.conv4 = resBlock_pni(filters2[4], filters2[4], pad_mode, bn_mode, relu_mode, init_mode, bn_momentum)
        else:
            self.cat0 = conv3dBlock([0], [filters2[4] * 2], bn_mode=[bn_mode], relu_mode=[relu_mode],
                                    bn_momentum=bn_momentum)
            self.conv4 = resBlock_pni(filters2[4] * 2, filters2[4], pad_mode, bn_mode, relu_mode, init_mode,
                                      bn_momentum)

        self.up1 = upsampleBlock(filters2[4], filters2[3], (1, 2, 2), upsample_mode, init_mode=init_mode)
        if self.merge_mode == 'add':
            self.cat1 = conv3dBlock([0], [filters2[3]], bn_mode=[bn_mode], relu_mode=[relu_mode],
                                    bn_momentum=bn_momentum)
            self.conv5 = resBlock_pni(filters2[3], filters2[3], pad_mode, bn_mode, relu_mode, init_mode, bn_momentum)
        else:
            self.cat1 = conv3dBlock([0], [filters2[3] * 2], bn_mode=[bn_mode], relu_mode=[relu_mode],
                                    bn_momentum=bn_momentum)
            self.conv5 = resBlock_pni(filters2[3] * 2, filters2[3], pad_mode, bn_mode, relu_mode, init_mode,
                                      bn_momentum)

        self.up2 = upsampleBlock(filters2[3], filters2[2], (1, 2, 2), upsample_mode, init_mode=init_mode)
        if self.merge_mode == 'add':
            self.cat2 = conv3dBlock([0], [filters2[2]], bn_mode=[bn_mode], relu_mode=[relu_mode],
                                    bn_momentum=bn_momentum)
            self.conv6 = resBlock_pni(filters2[2], filters2[2], pad_mode, bn_mode, relu_mode, init_mode, bn_momentum)
        else:
            self.cat2 = conv3dBlock([0], [filters2[2] * 2], bn_mode=[bn_mode], relu_mode=[relu_mode],
                                    bn_momentum=bn_momentum)
            self.conv6 = resBlock_pni(filters2[2] * 2, filters2[2], pad_mode, bn_mode, relu_mode, init_mode,
                                      bn_momentum)

        self.up3 = upsampleBlock(filters2[2], filters2[1], (1, 2, 2), upsample_mode, init_mode=init_mode)
        if self.merge_mode == 'add':
            self.cat3 = conv3dBlock([0], [filters2[1]], bn_mode=[bn_mode], relu_mode=[relu_mode],
                                    bn_momentum=bn_momentum)
            self.conv7 = resBlock_pni(filters2[1], filters2[1], pad_mode, bn_mode, relu_mode, init_mode, bn_momentum)
        else:
            self.cat3 = conv3dBlock([0], [filters2[1] * 2], bn_mode=[bn_mode], relu_mode=[relu_mode],
                                    bn_momentum=bn_momentum)
            self.conv7 = resBlock_pni(filters2[1] * 2, filters2[1], pad_mode, bn_mode, relu_mode, init_mode,
                                      bn_momentum)

        self.embed_out = conv3dBlock([int(filters2[0])],
                                     [int(filters2[0])],
                                     [(1, 5, 5)],
                                     [1],
                                     [(0, 2, 2)],
                                     [True],
                                     [pad_mode],
                                     [''],
                                     [relu_mode],
                                     init_mode,
                                     bn_momentum)

        self.out_put = conv3dBlock([int(filters2[0])], [out_planes], [(1, 1, 1)], init_mode=init_mode)
        self.sigmoid = nn.Sigmoid()

    def encoder(self, x):
        # embedding
        embed_in = self.embed_in(x)
        conv0 = self.conv0(embed_in)
        pool0 = self.pool0(conv0)
        conv1 = self.conv1(pool0)
        pool1 = self.pool1(conv1)
        conv2 = self.conv2(pool1)
        pool2 = self.pool2(conv2)
        conv3 = self.conv3(pool2)
        pool3 = self.pool3(conv3)

        center = self.center(pool3)
        down_features = [conv0, conv1, conv2, conv3]

        return center, down_features

    def forward(self, x):
        if self.encoder_update:
            center, down_features = self.encoder(x)
            pre_conv0, pre_conv1, pre_conv2, pre_conv3 = down_features
        else:
            with torch.no_grad():
                center, down_features = self.encoder(x)
                pre_conv0, pre_conv1, pre_conv2, pre_conv3 = down_features

        up0 = self.up0(center)
        if self.merge_mode == 'add':
            cat0 = self.cat0(up0 + pre_conv3)
        else:
            cat0 = self.cat0(torch.cat([up0, pre_conv3], dim=1))
        conv4 = self.conv4(cat0)

        up1 = self.up1(conv4)
        if self.merge_mode == 'add':
            cat1 = self.cat1(up1 + pre_conv2)
        else:
            cat1 = self.cat1(torch.cat([up1, pre_conv2], dim=1))
        conv5 = self.conv5(cat1)

        up2 = self.up2(conv5)
        if self.merge_mode == 'add':
            cat2 = self.cat2(up2 + pre_conv1)
        else:
            cat2 = self.cat2(torch.cat([up2, pre_conv1], dim=1))
        conv6 = self.conv6(cat2)

        up3 = self.up3(conv6)
        if self.merge_mode == 'add':
            cat3 = self.cat3(up3 + pre_conv0)
        else:
            cat3 = self.cat3(torch.cat([up3, pre_conv0], dim=1))
        conv7 = self.conv7(cat3)

        embed_out = self.embed_out(conv7)
        out = self.out_put(embed_out)

        if self.if_sigmoid:
            out = torch.sigmoid(out)

        if self.show_feature:
            center_features = [center]
            up_features = [conv4, conv5, conv6, conv7]
            return down_features, center_features, up_features, out
        else:
            return out


class UNet_PNI_encoder(nn.Module):
    def __init__(self, in_planes=1,
                 filters=[32, 64, 128, 256, 512],
                 pad_mode='zero',
                 bn_mode='async',  # async or sync
                 relu_mode='elu',
                 init_mode='kaiming_normal',
                 bn_momentum=0.001,
                 do_embed=True,
                 num_classes=None):
        super(UNet_PNI_encoder, self).__init__()
        filters2 = filters[:1] + filters
        self.do_embed = do_embed
        self.depth = len(filters2) - 2
        self.num_classes = num_classes

        # 2D conv for anisotropic
        self.embed_in = conv3dBlock([in_planes],
                                    [filters2[0]],
                                    [(1, 5, 5)],
                                    [1],
                                    [(0, 2, 2)],
                                    [True],
                                    [pad_mode],
                                    [''],
                                    [relu_mode],
                                    init_mode,
                                    bn_momentum)

        # downsample stream
        self.conv0 = resBlock_pni(filters2[0], filters2[1], pad_mode, bn_mode, relu_mode, init_mode, bn_momentum)
        self.pool0 = nn.MaxPool3d((1, 2, 2), (1, 2, 2))

        self.conv1 = resBlock_pni(filters2[1], filters2[2], pad_mode, bn_mode, relu_mode, init_mode, bn_momentum)
        self.pool1 = nn.MaxPool3d((1, 2, 2), (1, 2, 2))

        self.conv2 = resBlock_pni(filters2[2], filters2[3], pad_mode, bn_mode, relu_mode, init_mode, bn_momentum)
        self.pool2 = nn.MaxPool3d((1, 2, 2), (1, 2, 2))

        self.conv3 = resBlock_pni(filters2[3], filters2[4], pad_mode, bn_mode, relu_mode, init_mode, bn_momentum)
        self.pool3 = nn.MaxPool3d((1, 2, 2), (1, 2, 2))

        self.center = resBlock_pni(filters2[4], filters2[5], pad_mode, bn_mode, relu_mode, init_mode, bn_momentum)

        if self.num_classes is not None:
            self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
            self.fc = nn.Linear(filters[-1], num_classes)

    def encoder(self, x):
        # embedding
        embed_in = self.embed_in(x)
        conv0 = self.conv0(embed_in)
        pool0 = self.pool0(conv0)
        conv1 = self.conv1(pool0)
        pool1 = self.pool1(conv1)
        conv2 = self.conv2(pool1)
        pool2 = self.pool2(conv2)
        conv3 = self.conv3(pool2)
        pool3 = self.pool3(conv3)

        center = self.center(pool3)
        return center

    def forward(self, x):
        x = self.encoder(x)
        if self.num_classes is not None:
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.fc(x)
        return x


if __name__ == "__main__":
    import os
    import numpy as np
    from collections import OrderedDict
    # in_spec  = {'input':(1,1,18,160,160)}
    # out_spec = {'affinity':(1,12,18,160,160)}
    # model = RSUNet(1, 12, depth=4, upsample='transpose').to('cuda:0')
    # model = UNet_PNI_encoder(filters=[28, 36, 48, 64, 80]).to('cuda:0')
    # model = UNet_PNI_encoder(filters=[32, 64, 128, 256, 512]).to('cuda:0')
    # model = UNet_PNI_Noskip2(filters=[32, 64, 128, 256, 512], show_feature=False).to('cuda:0')

    # model_structure(model)
    # model_dict = model.state_dict()
    # from utils.encoder_dict import ENCODER_DICT, ENCODER_DICT2
    from utils.encoder_dict import freeze_layers, difflr_optimizer

    input = np.random.random((1, 1, 18, 160, 160)).astype(np.float32)
    x = torch.tensor(input).to('cuda:0')

    # out = model(x)
    # model_ft = UNet_PNI_FT1(model, filters=[28, 36, 48, 64, 80], merge_mode='add3', show_feature=False).to('cuda:0')
    # model_ft = UNet_PNI_FT1(model, filters=[32, 64, 128, 256, 512], merge_mode='cat3', show_feature=False).to('cuda:0')
    # model_ft = UNet_PNI_FT2(filters=[32, 64, 128, 256, 512], merge_mode='cat', show_feature=False, encoder_update=True).to('cuda:0')
    # model = UNet_PNI(filters=[32, 64, 128, 256, 512], upsample_mode='transposeS', merge_mode='cat').to('cuda:0')
    model = UNet_PNI(filters=[28, 36, 48, 64, 80], upsample_mode='transposeS', merge_mode='cat').to('cuda:0')

    # print('Load pre-trained model ...')
    # ckpt_path = os.path.join('../models', \
    #     '2020-12-30--06-21-26_ssl_3aug_suhu_mse_lr0001_snemi3d_m512_ulb5', \
    #     'model-%06d.ckpt' % 100000)
    # checkpoint = torch.load(ckpt_path)
    # pretrained_dict = checkpoint['model_weights']
    # trained_gpus = 1
    # if_skip = 'False'
    # if trained_gpus > 1:
    #     pretained_model_dict = OrderedDict()
    #     for k, v in pretrained_dict.items():
    #         name = k[7:] # remove module.
    #         # name = k
    #         pretained_model_dict[name] = v
    # else:
    #     pretained_model_dict = pretrained_dict

    # from utils.encoder_dict import ENCODER_DICT2, ENCODER_DECODER_DICT2
    # model_dict = model.state_dict()
    # encoder_dict = OrderedDict()
    # if if_skip == 'True':
    #     print('Load the parameters of encoder and decoder!')
    #     encoder_dict = {k: v for k, v in pretained_model_dict.items() if k.split('.')[0] in ENCODER_DECODER_DICT2}
    # else:
    #     print('Load the parameters of encoder!')
    #     encoder_dict = {k: v for k, v in pretained_model_dict.items() if k.split('.')[0] in ENCODER_DICT2}
    # model_dict.update(encoder_dict)
    # model.load_state_dict(model_dict)

    # for k, v in pretained_model_dict.items():
    #     if k.split('.')[0] in ENCODER_DECODER_DICT2:
    #         error = torch.sum(v - model.state_dict()[k])
    #         print(k, error.item())
    # encoder_layers = ['embed_in', 'conv0', 'conv1', 'conv2', 'conv3', 'center']
    # decoder_layers = ['up0', 'cat0', 'conv4', 'up1', 'cat1', 'conv5', 'up2', 'cat2', 'conv6', 'up3', 'cat3', 'conv7', 'embed_out', 'out_put']

    # for param in model_ft.embed_in.parameters():
    #     param.requires_grad = False
    # for param in model_ft.conv0.parameters():
    #     param.requires_grad = False
    # for param in model_ft.conv1.parameters():
    #     param.requires_grad = False
    # for param in model_ft.conv2.parameters():
    #     param.requires_grad = False
    # for param in model_ft.conv3.parameters():
    #     param.requires_grad = False
    # for param in model_ft.center.parameters():
    #     param.requires_grad = False

    # model_ft = freeze_layers(model_ft, if_skip='True')

    # encoder_layers_param = []
    # encoder_layers_param += list(map(id, model_ft.embed_in.parameters()))
    # encoder_layers_param += list(map(id, model_ft.conv0.parameters()))
    # encoder_layers_param += list(map(id, model_ft.conv1.parameters()))
    # encoder_layers_param += list(map(id, model_ft.conv2.parameters()))
    # encoder_layers_param += list(map(id, model_ft.conv3.parameters()))
    # encoder_layers_param += list(map(id, model_ft.center.parameters()))
    # decoder_param = filter(lambda p: id(p) not in encoder_layers_param, model_ft.parameters())
    # encoder_param = filter(lambda p: id(p) in encoder_layers_param, model_ft.parameters())

    # paras = dict(model_ft.named_parameters())
    # for k, v in paras.items():
    #     print(k.ljust(30), str(v.shape).ljust(30), 'bias:', v.requires_grad)

    # optimizer = torch.optim.Adam(model_ft.parameters(), lr=0.0001, betas=(0.9, 0.999),
    #                              eps=0.01, weight_decay=1e-6, amsgrad=True)
    # optimizer = torch.optim.Adam([{'params': encoder_param, 'lr': 0.00001},
    #                               {'params': decoder_param}], 
    #                             lr=0.0001, betas=(0.9, 0.999), eps=0.01, weight_decay=1e-6, amsgrad=True)
    # optimizer = difflr_optimizer(model_ft, if_skip='True')
    # for p in optimizer.param_groups:
    #     outputs = ''
    #     for k, v in p.items():
    #         if k is 'params':
    #             outputs += (k + ': ' + str(v[0].shape).ljust(30) + ' ')
    #             for ks in range(len(v)):
    #                 print(v[ks].shape)
    #         else:
    #             outputs += (k + ': ' + str(v).ljust(10) + ' ')
    #     print(outputs)

    # model_ft_state_dict = model_ft.state_dict()
    # new_state_dict = OrderedDict()
    # new_state_dict = {k: v for k, v in model_dict.items() if k.split('.')[0] in ENCODER_DICT2}
    # model_ft_state_dict.update(new_state_dict)
    # model_ft.load_state_dict(model_ft_state_dict)

    model_structure(model)
    out = model(x)

    print(out.shape)



class UNet_PNI_LSD(nn.Module):  # PNI model with LSD support
    """
    UNet_PNI variant for LSD (Local Shape Descriptors) and affinity prediction
    Based on: Superhuman Accuracy on the SNEMI3D Connectomics Challenge. Lee et al.
    https://arxiv.org/abs/1706.00120
    """
    def __init__(self, in_planes=1,
                 out_planes_aff=3,  # 亲和力输出通道数
                 out_planes_lsd=10,  # LSD输出通道数
                 filters=[28, 36, 48, 64, 80],
                 upsample_mode='transposeS',
                 decode_ratio=1,
                 merge_mode='cat',
                 pad_mode='zero',
                 bn_mode='async',
                 relu_mode='elu',
                 init_mode='kaiming_normal',
                 bn_momentum=0.001,
                 do_embed=True,
                 if_sigmoid=True,
                 show_feature=False,
                 if_uncertainty=False
                 ):
        super(UNet_PNI_LSD, self).__init__()
        filters2 = filters[:1] + filters
        self.merge_mode = merge_mode
        self.do_embed = do_embed
        self.depth = len(filters2) - 2
        self.if_sigmoid = if_sigmoid
        self.show_feature = show_feature
        self.if_uncertainty = if_uncertainty
        self.out_planes_aff = out_planes_aff
        self.out_planes_lsd = out_planes_lsd

        # 2D conv for anisotropic
        self.embed_in = conv3dBlock([in_planes],
                                    [filters2[0]],
                                    [(1, 5, 5)],
                                    [1],
                                    [(0, 2, 2)],
                                    [True],
                                    [pad_mode],
                                    [''],
                                    [relu_mode],
                                    init_mode,
                                    bn_momentum)

        # downsample stream
        self.conv0 = resBlock_pni(filters2[0], filters2[1], pad_mode, bn_mode, relu_mode, init_mode, bn_momentum)
        self.pool0 = nn.MaxPool3d((1, 2, 2), (1, 2, 2))

        self.conv1 = resBlock_pni(filters2[1], filters2[2], pad_mode, bn_mode, relu_mode, init_mode, bn_momentum)
        self.pool1 = nn.MaxPool3d((1, 2, 2), (1, 2, 2))

        self.conv2 = resBlock_pni(filters2[2], filters2[3], pad_mode, bn_mode, relu_mode, init_mode, bn_momentum)
        self.pool2 = nn.MaxPool3d((1, 2, 2), (1, 2, 2))

        self.conv3 = resBlock_pni(filters2[3], filters2[4], pad_mode, bn_mode, relu_mode, init_mode, bn_momentum)
        self.pool3 = nn.MaxPool3d((1, 2, 2), (1, 2, 2))

        self.center = resBlock_pni(filters2[4], filters2[5], pad_mode, bn_mode, relu_mode, init_mode, bn_momentum)

        # upsample stream
        self.up0 = upsampleBlock(filters2[5], filters2[4], (1, 2, 2), upsample_mode, init_mode=init_mode)
        if self.merge_mode == 'add':
            self.cat0 = conv3dBlock([0], [filters2[4]], bn_mode=[bn_mode], relu_mode=[relu_mode],
                                    bn_momentum=bn_momentum)
            self.conv4 = resBlock_pni(filters2[4], filters2[4], pad_mode, bn_mode, relu_mode, init_mode, bn_momentum)
        else:
            self.cat0 = conv3dBlock([0], [filters2[4] * 2], bn_mode=[bn_mode], relu_mode=[relu_mode],
                                    bn_momentum=bn_momentum)
            self.conv4 = resBlock_pni(filters2[4] * 2, filters2[4], pad_mode, bn_mode, relu_mode, init_mode,
                                      bn_momentum)

        self.up1 = upsampleBlock(filters2[4], filters2[3], (1, 2, 2), upsample_mode, init_mode=init_mode)
        if self.merge_mode == 'add':
            self.cat1 = conv3dBlock([0], [filters2[3]], bn_mode=[bn_mode], relu_mode=[relu_mode],
                                    bn_momentum=bn_momentum)
            self.conv5 = resBlock_pni(filters2[3], filters2[3], pad_mode, bn_mode, relu_mode, init_mode, bn_momentum)
        else:
            self.cat1 = conv3dBlock([0], [filters2[3] * 2], bn_mode=[bn_mode], relu_mode=[relu_mode],
                                    bn_momentum=bn_momentum)
            self.conv5 = resBlock_pni(filters2[3] * 2, filters2[3], pad_mode, bn_mode, relu_mode, init_mode,
                                      bn_momentum)

        self.up2 = upsampleBlock(filters2[3], filters2[2], (1, 2, 2), upsample_mode, init_mode=init_mode)
        if self.merge_mode == 'add':
            self.cat2 = conv3dBlock([0], [filters2[2]], bn_mode=[bn_mode], relu_mode=[relu_mode],
                                    bn_momentum=bn_momentum)
            self.conv6 = resBlock_pni(filters2[2], filters2[2], pad_mode, bn_mode, relu_mode, init_mode, bn_momentum)
        else:
            self.cat2 = conv3dBlock([0], [filters2[2] * 2], bn_mode=[bn_mode], relu_mode=[relu_mode],
                                    bn_momentum=bn_momentum)
            self.conv6 = resBlock_pni(filters2[2] * 2, filters2[2], pad_mode, bn_mode, relu_mode, init_mode,
                                      bn_momentum)

        self.up3 = upsampleBlock(filters2[2], filters2[1], (1, 2, 2), upsample_mode, init_mode=init_mode)
        if self.merge_mode == 'add':
            self.cat3 = conv3dBlock([0], [filters2[1]], bn_mode=[bn_mode], relu_mode=[relu_mode],
                                    bn_momentum=bn_momentum)
            self.conv7 = resBlock_pni(filters2[1], filters2[1], pad_mode, bn_mode, relu_mode, init_mode, bn_momentum)
        else:
            self.cat3 = conv3dBlock([0], [filters2[1] * 2], bn_mode=[bn_mode], relu_mode=[relu_mode],
                                    bn_momentum=bn_momentum)
            self.conv7 = resBlock_pni(filters2[1] * 2, filters2[1], pad_mode, bn_mode, relu_mode, init_mode,
                                      bn_momentum)

        self.embed_out = conv3dBlock([int(filters2[0])],
                                     [int(filters2[0])],
                                     [(1, 5, 5)],
                                     [1],
                                     [(0, 2, 2)],
                                     [True],
                                     [pad_mode],
                                     [''],
                                     [relu_mode],
                                     init_mode,
                                     bn_momentum)

        # 输出头：LSD和亲和力
        self.lsd_output = conv3dBlock([int(filters2[0])], [out_planes_lsd], [(1, 1, 1)], init_mode=init_mode)
        self.aff_output = conv3dBlock([int(filters2[0])], [out_planes_aff], [(1, 1, 1)], init_mode=init_mode)

        # 不确定性输出（可选）
        if self.if_uncertainty:
            self.lsd_uncertainty = conv3dBlock([int(filters2[0])], [out_planes_lsd], [(1, 1, 1)], init_mode=init_mode)
            self.aff_uncertainty = conv3dBlock([int(filters2[0])], [out_planes_aff], [(1, 1, 1)], init_mode=init_mode)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # embedding
        embed_in = self.embed_in(x)
        conv0 = self.conv0(embed_in)
        pool0 = self.pool0(conv0)
        conv1 = self.conv1(pool0)
        pool1 = self.pool1(conv1)
        conv2 = self.conv2(pool1)
        pool2 = self.pool2(conv2)
        conv3 = self.conv3(pool2)
        pool3 = self.pool3(conv3)

        center = self.center(pool3)

        up0 = self.up0(center)
        if self.merge_mode == 'add':
            cat0 = self.cat0(up0 + conv3)
        else:
            cat0 = self.cat0(torch.cat([up0, conv3], dim=1))
        conv4 = self.conv4(cat0)

        up1 = self.up1(conv4)
        if self.merge_mode == 'add':
            cat1 = self.cat1(up1 + conv2)
        else:
            cat1 = self.cat1(torch.cat([up1, conv2], dim=1))
        conv5 = self.conv5(cat1)

        up2 = self.up2(conv5)
        if self.merge_mode == 'add':
            cat2 = self.cat2(up2 + conv1)
        else:
            cat2 = self.cat2(torch.cat([up2, conv1], dim=1))
        conv6 = self.conv6(cat2)

        up3 = self.up3(conv6)
        if self.merge_mode == 'add':
            cat3 = self.cat3(up3 + conv0)
        else:
            cat3 = self.cat3(torch.cat([up3, conv0], dim=1))
        conv7 = self.conv7(cat3)

        embed_out = self.embed_out(conv7)
        
        # 多任务输出
        out_lsd = self.lsd_output(embed_out)
        out_aff = self.aff_output(embed_out)
        
        # 拼接输出 [LSD: 10通道, AFF: 3通道]
        out = torch.cat([out_lsd, out_aff], dim=1)

        if self.if_sigmoid:
            out = torch.sigmoid(out)

        if self.show_feature:
            down_features = [conv0, conv1, conv2, conv3]
            center_features = [center]
            up_features = [conv4, conv5, conv6, conv7]
            return down_features, center_features, up_features, out
        else:
            if self.if_uncertainty:
                unc_lsd = self.lsd_uncertainty(embed_out)
                unc_aff = self.aff_uncertainty(embed_out)
                uncertainties = [unc_lsd, unc_aff]
                return out, uncertainties
            else:
                return out