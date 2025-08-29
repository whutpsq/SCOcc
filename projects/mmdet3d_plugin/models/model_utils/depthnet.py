import torch
import torch.nn as nn
import torch.nn.functional as F
from mmdet.models.backbones.resnet import BasicBlock
from mmcv.cnn import build_conv_layer
from torch.cuda.amp.autocast_mode import autocast
from torch.utils.checkpoint import checkpoint


# class EnhancedCBAM(nn.Module):
#     """增强版CBAM，结合多尺度特征和通道重标定"""
#
#     def __init__(self, channels, reduction_ratio=16, use_spatial=True):
#         super(EnhancedCBAM, self).__init__()
#         self.channel_att = ChannelAttention(channels, reduction_ratio)
#         self.use_spatial = use_spatial
#         if use_spatial:
#             self.spatial_att = SpatialAttention()
#
#         # 多尺度特征提取
#         self.conv3x3 = nn.Conv2d(channels, channels, 3, padding=1, groups=channels)
#         self.conv5x5 = nn.Conv2d(channels, channels, 5, padding=2, groups=channels)
#         self.conv_combine = nn.Sequential(
#             nn.Conv2d(channels * 3, channels, 1),
#             nn.BatchNorm2d(channels),
#             nn.ReLU()
#         )
#
#     def forward(self, x):
#         # 多尺度特征融合
#         x3 = F.relu(self.conv3x3(x))
#         x5 = F.relu(self.conv5x5(x))
#         x_combined = torch.cat([x, x3, x5], dim=1)
#         x = x + self.conv_combine(x_combined)
#
#         # 注意力机制
#         x = self.channel_att(x)
#         if self.use_spatial:
#             x = self.spatial_att(x)
#         return x
#
#
# class DepthAwareAttention(nn.Module):
#     """深度感知注意力机制"""
#
#     def __init__(self, in_channels):
#         super(DepthAwareAttention, self).__init__()
#         self.conv_depth = nn.Conv2d(88, in_channels, kernel_size=7, padding=3)
#         self.conv_feat = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
#         self.att_conv = nn.Sequential(
#             nn.Conv2d(in_channels * 2, in_channels, 1),
#             nn.Sigmoid()
#         )
#
#     def forward(self, feat, depth):
#         # 深度特征提取
#         depth_feat = self.conv_depth(depth)
#         # 特征提取
#         feat_processed = self.conv_feat(feat)
#         # 注意力融合
#         att = self.att_conv(torch.cat([feat_processed, depth_feat], dim=1))
#         return feat * att
# class CBAM(nn.Module):
#     def __init__(self, channels, reduction_ratio=16):
#         super(CBAM, self).__init__()
#         self.channel_att = ChannelAttention(channels, reduction_ratio)
#         self.spatial_att = SpatialAttention()
#
#     def forward(self, x):
#         x = self.channel_att(x)
#         x = self.spatial_att(x)
#         return x


# class ChannelAttention(nn.Module):
#     def __init__(self, in_planes, ratio=16):
#         super(ChannelAttention, self).__init__()
#         self.avg_pool = nn.AdaptiveAvgPool2d(1)
#         self.max_pool = nn.AdaptiveMaxPool2d(1)
#
#         self.shared_MLP = nn.Sequential(
#             nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
#             nn.ReLU(),
#             nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
#         )
#         self.sigmoid = nn.Sigmoid()
#
#     def forward(self, x):
#         avg_out = self.shared_MLP(self.avg_pool(x))
#         max_out = self.shared_MLP(self.max_pool(x))
#         out = avg_out + max_out
#         return x * self.sigmoid(out)
#
#
# class SpatialAttention(nn.Module):
#     def __init__(self, kernel_size=7):
#         super(SpatialAttention, self).__init__()
#         self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
#         self.sigmoid = nn.Sigmoid()
#
#     def forward(self, x):
#         avg_out = torch.mean(x, dim=1, keepdim=True)
#         max_out, _ = torch.max(x, dim=1, keepdim=True)
#         x_out = torch.cat([avg_out, max_out], dim=1)
#         x_out = self.conv(x_out)
#         return x * self.sigmoid(x_out)
class _ASPPModule(nn.Module):
    def __init__(self, inplanes, planes, kernel_size, padding, dilation,
                 BatchNorm):
        super(_ASPPModule, self).__init__()
        self.atrous_conv = nn.Conv2d(
            inplanes,
            planes,
            kernel_size=kernel_size,
            stride=1,
            padding=padding,
            dilation=dilation,
            bias=False)
        self.bn = BatchNorm(planes)
        self.relu = nn.ReLU()

        self._init_weight()

    def forward(self, x):
        x = self.atrous_conv(x)
        x = self.bn(x)

        return self.relu(x)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class ASPP(nn.Module):
    def __init__(self, inplanes, mid_channels=256, BatchNorm=nn.BatchNorm2d):
        super(ASPP, self).__init__()
        dilations = [1, 6, 12, 18]
        self.aspp1 = _ASPPModule(
            inplanes,
            mid_channels,
            1,
            padding=0,
            dilation=dilations[0],
            BatchNorm=BatchNorm)
        self.aspp2 = _ASPPModule(
            inplanes,
            mid_channels,
            3,
            padding=dilations[1],
            dilation=dilations[1],
            BatchNorm=BatchNorm)
        self.aspp3 = _ASPPModule(
            inplanes,
            mid_channels,
            3,
            padding=dilations[2],
            dilation=dilations[2],
            BatchNorm=BatchNorm)
        self.aspp4 = _ASPPModule(
            inplanes,
            mid_channels,
            3,
            padding=dilations[3],
            dilation=dilations[3],
            BatchNorm=BatchNorm)

        self.global_avg_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(inplanes, mid_channels, 1, stride=1, bias=False),
            BatchNorm(mid_channels),
            nn.ReLU(),
        )
        self.conv1 = nn.Conv2d(
            int(mid_channels * 5), inplanes, 1, bias=False)
        self.bn1 = BatchNorm(inplanes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self._init_weight()

    def forward(self, x):
        """
        Args:
            x: (B*N, C, fH, fW)
        Returns:
            x: (B*N, C, fH, fW)
        """
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)
        x5 = self.global_avg_pool(x)
        x5 = F.interpolate(
            x5, size=x4.size()[2:], mode='bilinear', align_corners=True)
        x = torch.cat((x1, x2, x3, x4, x5), dim=1)  # (B*N, 5*C', fH, fW)

        x = self.conv1(x)   # (B*N, C, fH, fW)
        x = self.bn1(x)
        x = self.relu(x)

        return self.dropout(x)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class Mlp(nn.Module):
    def __init__(self,
                 in_features,
                 hidden_features=None,
                 out_features=None,
                 act_layer=nn.ReLU,
                 drop=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x):
        """
        Args:
            x: (B*N_views, 27)
        Returns:
            x: (B*N_views, C)
        """
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class SELayer(nn.Module):
    def __init__(self, channels, act_layer=nn.ReLU, gate_layer=nn.Sigmoid):
        super().__init__()
        self.conv_reduce = nn.Conv2d(channels, channels, 1, bias=True)
        self.act1 = act_layer()
        self.conv_expand = nn.Conv2d(channels, channels, 1, bias=True)
        self.gate = gate_layer()

    def forward(self, x, x_se):
        """
        Args:
            x: (B*N_views, C_mid, fH, fW)
            x_se: (B*N_views, C_mid, 1, 1)
        Returns:
            x: (B*N_views, C_mid, fH, fW)
        """
        x_se = self.conv_reduce(x_se)     # (B*N_views, C_mid, 1, 1)
        x_se = self.act1(x_se)      # (B*N_views, C_mid, 1, 1)
        x_se = self.conv_expand(x_se)   # (B*N_views, C_mid, 1, 1)
        return x * self.gate(x_se)      # (B*N_views, C_mid, fH, fW)


class DepthNet(nn.Module):
    def __init__(self,
                 in_channels,
                 mid_channels,
                 context_channels,
                 depth_channels,
                 use_dcn=True,
                 use_aspp=True,
                 with_cp=False,
                 stereo=False,
                 bias=0.0,
                 aspp_mid_channels=-1):
        super(DepthNet, self).__init__()

        self.reduce_conv = nn.Sequential(
            nn.Conv2d(
                in_channels, mid_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
        )

        # 生成context feature
        self.context_conv = nn.Conv2d(
            mid_channels, context_channels, kernel_size=1, stride=1, padding=0)

        self.bn = nn.BatchNorm1d(27)
        self.depth_mlp = Mlp(in_features=27, hidden_features=mid_channels, out_features=mid_channels)
        self.depth_se = SELayer(channels=mid_channels)  # NOTE: add camera-aware
        self.context_mlp = Mlp(in_features=27, hidden_features=mid_channels, out_features=mid_channels)
        self.context_se = SELayer(channels=mid_channels)  # NOTE: add camera-aware
        depth_conv_input_channels = mid_channels
        downsample = None

        if stereo:
            depth_conv_input_channels += depth_channels
            downsample = nn.Conv2d(depth_conv_input_channels,
                                    mid_channels, 1, 1, 0)
            cost_volumn_net = []
            for stage in range(int(2)):
                cost_volumn_net.extend([
                    nn.Conv2d(depth_channels, depth_channels, kernel_size=3,
                              stride=2, padding=1),
                    nn.BatchNorm2d(depth_channels)])
            self.cost_volumn_net = nn.Sequential(*cost_volumn_net)
            self.bias = bias

        # 3个残差blocks
        depth_conv_list = [BasicBlock(depth_conv_input_channels, mid_channels,
                                      downsample=downsample),
                           BasicBlock(mid_channels, mid_channels),
                           BasicBlock(mid_channels, mid_channels)]
        if use_aspp:
            if aspp_mid_channels < 0:
                aspp_mid_channels = mid_channels
            depth_conv_list.append(ASPP(mid_channels, aspp_mid_channels))
        if use_dcn:
            depth_conv_list.append(
                build_conv_layer(
                    cfg=dict(
                        type='DCN',
                        in_channels=mid_channels,
                        out_channels=mid_channels,
                        kernel_size=3,
                        padding=1,
                        groups=4,
                        im2col_step=128,
                    )))

        depth_conv_list.append(
            nn.Conv2d(
                mid_channels,
                depth_channels,
                kernel_size=1,
                stride=1,
                padding=0))
        self.depth_conv = nn.Sequential(*depth_conv_list)
        self.with_cp = with_cp
        self.depth_channels = depth_channels

    # ----------------------------------------- 用于建立cost volume ----------------------------------
    def gen_grid(self, metas, B, N, D, H, W, hi, wi):
        """
        Args:
            metas: dict{
                k2s_sensor: (B, N_views, 4, 4)
                intrins: (B, N_views, 3, 3)
                post_rots: (B, N_views, 3, 3)
                post_trans: (B, N_views, 3)
                frustum: (D, fH_stereo, fW_stereo, 3)  3:(u, v, d)
                cv_downsample: 4,
                downsample: self.img_view_transformer.downsample=16,
                grid_config: self.img_view_transformer.grid_config,
                cv_feat_list: [feat_prev_iv, stereo_feat]
            }
            B: batchsize
            N: N_views
            D: D
            H: fH_stereo
            W: fW_stereo
            hi: H_img
            wi: W_img
        Returns:
            grid: (B*N_views, D*fH_stereo, fW_stereo, 2)
        """
        frustum = metas['frustum']      # (D, fH_stereo, fW_stereo, 3)  3:(u, v, d)
        # 逆图像增广:
        points = frustum - metas['post_trans'].view(B, N, 1, 1, 1, 3)
        points = torch.inverse(metas['post_rots']).view(B, N, 1, 1, 1, 3, 3) \
            .matmul(points.unsqueeze(-1))   # (B, N_views, D, fH_stereo, fW_stereo, 3, 1)

        # (u, v, d) --> (du, dv, d)
        # (B, N_views, D, fH_stereo, fW_stereo, 3, 1)
        points = torch.cat(
            (points[..., :2, :] * points[..., 2:3, :], points[..., 2:3, :]), 5)

        # cur_pixel --> curr_camera --> prev_camera
        rots = metas['k2s_sensor'][:, :, :3, :3].contiguous()
        trans = metas['k2s_sensor'][:, :, :3, 3].contiguous()
        combine = rots.matmul(torch.inverse(metas['intrins']))
        points = combine.view(B, N, 1, 1, 1, 3, 3).matmul(points)
        points += trans.view(B, N, 1, 1, 1, 3, 1)   # (B, N_views, D, fH_stereo, fW_stereo, 3, 1)

        neg_mask = points[..., 2, 0] < 1e-3
        # prev_camera --> prev_pixel
        points = metas['intrins'].view(B, N, 1, 1, 1, 3, 3).matmul(points)
        # (du, dv, d) --> (u, v)   (B, N_views, D, fH_stereo, fW_stereo, 2, 1)
        points = points[..., :2, :] / points[..., 2:3, :]

        # 图像增广
        points = metas['post_rots'][..., :2, :2].view(B, N, 1, 1, 1, 2, 2).matmul(
            points).squeeze(-1)
        points += metas['post_trans'][..., :2].view(B, N, 1, 1, 1, 2)   # (B, N_views, D, fH_stereo, fW_stereo, 2)

        px = points[..., 0] / (wi - 1.0) * 2.0 - 1.0
        py = points[..., 1] / (hi - 1.0) * 2.0 - 1.0
        px[neg_mask] = -2
        py[neg_mask] = -2
        grid = torch.stack([px, py], dim=-1)    # (B, N_views, D, fH_stereo, fW_stereo, 2)
        grid = grid.view(B * N, D * H, W, 2)    # (B*N_views, D*fH_stereo, fW_stereo, 2)
        return grid

    def calculate_cost_volumn(self, metas):
        """
        Args:
            metas: dict{
                k2s_sensor: (B, N_views, 4, 4)
                intrins: (B, N_views, 3, 3)
                post_rots: (B, N_views, 3, 3)
                post_trans: (B, N_views, 3)
                frustum: (D, fH_stereo, fW_stereo, 3)  3:(u, v, d)
                cv_downsample: 4,
                downsample: self.img_view_transformer.downsample=16,
                grid_config: self.img_view_transformer.grid_config,
                cv_feat_list: [feat_prev_iv, stereo_feat]
            }
        Returns:
            cost_volumn: (B*N_views, D, fH_stereo, fW_stereo)
        """
        prev, curr = metas['cv_feat_list']    # (B*N_views, C_stereo, fH_stereo, fW_stereo)
        group_size = 4
        _, c, hf, wf = curr.shape   #
        hi, wi = hf * 4, wf * 4     # H_img, W_img
        B, N, _ = metas['post_trans'].shape
        D, H, W, _ = metas['frustum'].shape
        grid = self.gen_grid(metas, B, N, D, H, W, hi, wi).to(curr.dtype)   # (B*N_views, D*fH_stereo, fW_stereo, 2)

        prev = prev.view(B * N, -1, H, W)   # (B*N_views, C_stereo, fH_stereo, fW_stereo)
        curr = curr.view(B * N, -1, H, W)   # (B*N_views, C_stereo, fH_stereo, fW_stereo)
        cost_volumn = 0
        # process in group wise to save memory
        for fid in range(curr.shape[1] // group_size):
            # (B*N_views, group_size, fH_stereo, fW_stereo)
            prev_curr = prev[:, fid * group_size:(fid + 1) * group_size, ...]
            wrap_prev = F.grid_sample(prev_curr, grid,
                                      align_corners=True,
                                      padding_mode='zeros')     # (B*N_views, group_size, D*fH_stereo, fW_stereo)
            # (B*N_views, group_size, fH_stereo, fW_stereo)
            curr_tmp = curr[:, fid * group_size:(fid + 1) * group_size, ...]
            # (B*N_views, group_size, 1, fH_stereo, fW_stereo) - (B*N_views, group_size, D, fH_stereo, fW_stereo)
            # --> (B*N_views, group_size, D, fH_stereo, fW_stereo)
            # https://github.com/HuangJunJie2017/BEVDet/issues/278
            cost_volumn_tmp = curr_tmp.unsqueeze(2) - \
                              wrap_prev.view(B * N, -1, D, H, W)
            cost_volumn_tmp = cost_volumn_tmp.abs().sum(dim=1)      # (B*N_views, D, fH_stereo, fW_stereo)
            cost_volumn += cost_volumn_tmp  # (B*N_views, D, fH_stereo, fW_stereo)
        if not self.bias == 0:
            invalid = wrap_prev[:, 0, ...].view(B * N, D, H, W) == 0
            cost_volumn[invalid] = cost_volumn[invalid] + self.bias

        # matching cost --> prob
        cost_volumn = - cost_volumn
        cost_volumn = cost_volumn.softmax(dim=1)
        return cost_volumn
    # ----------------------------------------- 用于建立cost volume --------------------------------------

    def forward(self, x, mlp_input, stereo_metas=None):
        """
        Args:
            x: (B*N_views, C, fH, fW)
            mlp_input: (B, N_views, 27)
            stereo_metas:  None or dict{
                k2s_sensor: (B, N_views, 4, 4)
                intrins: (B, N_views, 3, 3)
                post_rots: (B, N_views, 3, 3)
                post_trans: (B, N_views, 3)
                frustum: (D, fH_stereo, fW_stereo, 3)  3:(u, v, d)
                cv_downsample: 4,
                downsample: self.img_view_transformer.downsample=16,
                grid_config: self.img_view_transformer.grid_config,
                cv_feat_list: [feat_prev_iv, stereo_feat]
            }
        Returns:
            x: (B*N_views, D+C_context, fH, fW)
        """
        mlp_input = self.bn(mlp_input.reshape(-1, mlp_input.shape[-1]))     # (B*N_views, 27)
        x = self.reduce_conv(x)     # (B*N_views, C_mid, fH, fW)

        # # # 添加注意力模块
        # if self.use_attention:
        #     x = self.cbam(x)  # 应用通道和空间注意力


        ######################### V2
        # # 应用增强注意力
        # if self.use_attention:
        #     x = self.enhanced_cbam(x)
        #
        # # 多尺度特征分支
        # multiscale_feat = self.multiscale_branch(x)
        # x = torch.cat([x, multiscale_feat], dim=1)
        # x = self.multiscale_combine(x)
        #
        # # 深度监督点 - 输出D通道
        # depth_supervision = self.depth_supervision_conv(x)
        #
        # # 应用深度感知注意力
        # if self.use_depth_aware_attn:
        #     x = self.depth_aware_attn(x, depth_supervision.sigmoid())
        #########################



        # (B*N_views, 27) --> (B*N_views, C_mid) --> (B*N_views, C_mid, 1, 1)
        context_se = self.context_mlp(mlp_input)[..., None, None]
        context = self.context_se(x, context_se)    # (B*N_views, C_mid, fH, fW)
        context = self.context_conv(context)        # (B*N_views, C_context, fH, fW)

        # (B*N_views, 27) --> (B*N_views, C_mid) --> (B*N_views, C_mid, 1, 1)
        depth_se = self.depth_mlp(mlp_input)[..., None, None]
        depth = self.depth_se(x, depth_se)      # (B*N_views, C_mid, fH, fW)

        if not stereo_metas is None:
            if stereo_metas['cv_feat_list'][0] is None:
                BN, _, H, W = x.shape
                scale_factor = float(stereo_metas['downsample'])/\
                               stereo_metas['cv_downsample']
                cost_volumn = \
                    torch.zeros((BN, self.depth_channels,
                                 int(H*scale_factor),
                                 int(W*scale_factor))).to(x)
            else:
                with torch.no_grad():
                    # https://github.com/HuangJunJie2017/BEVDet/issues/278
                    cost_volumn = self.calculate_cost_volumn(stereo_metas)      # (B*N_views, D, fH_stereo, fW_stereo)
            cost_volumn = self.cost_volumn_net(cost_volumn)     # (B*N_views, D, fH, fW)
            depth = torch.cat([depth, cost_volumn], dim=1)      # (B*N_views, C_mid+D, fH, fW)
        if self.with_cp:
            depth = checkpoint(self.depth_conv, depth)
        else:
            # 3*res blocks +ASPP/DCN + Conv(c_mid-->D)
            depth = self.depth_conv(depth)  # x: (B*N_views, C_mid, fH, fW) --> (B*N_views, D, fH, fW)

        return torch.cat([depth, context], dim=1)


class DepthNetV2(nn.Module):
    def __init__(self,
                 in_channels,
                 mid_channels,
                 context_channels,
                 depth_channels,
                 use_dcn=True,
                 use_aspp=True,
                 with_cp=False,
                 stereo=False,
                 bias=0.0,
                 aspp_mid_channels=-1):
        super(DepthNetV2, self).__init__()

        # 1. 多尺度特征提取
        self.multiscale_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, 3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(),
            self._make_res_block(mid_channels, mid_channels),
            self._make_res_block(mid_channels, mid_channels),
            nn.Conv2d(mid_channels, mid_channels, 3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU()
        )

        # 2. 深度特征分支
        self.reduce_conv = nn.Sequential(
            nn.Conv2d(mid_channels, mid_channels, 3, stride=1, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
        )

        # 3. 上下文特征分支
        self.context_conv = nn.Conv2d(
            mid_channels, context_channels, kernel_size=1, stride=1, padding=0)

        # 4. 相机参数处理
        self.bn = nn.BatchNorm1d(27)
        self.depth_mlp = Mlp(in_features=27, hidden_features=mid_channels,
                             out_features=mid_channels)
        self.context_mlp = Mlp(in_features=27, hidden_features=mid_channels,
                               out_features=mid_channels)

        # 5. 相机感知注意力
        self.depth_se = SELayer(channels=mid_channels)
        self.context_se = SELayer(channels=mid_channels)

        # 6. 深度注意力机制
        self.depth_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(mid_channels, mid_channels // 8, 1),
            nn.ReLU(),
            nn.Conv2d(mid_channels // 8, mid_channels, 1),
            nn.Sigmoid()
        )

        # 7. 深度不确定性估计
        self.uncertainty_conv = nn.Sequential(
            nn.Conv2d(depth_channels, depth_channels // 2, 3, padding=1),
            nn.BatchNorm2d(depth_channels // 2),
            nn.ReLU(),
            nn.Conv2d(depth_channels // 2, 1, 1),
            nn.Sigmoid()
        )

        # 8. 深度特征聚合
        depth_conv_input_channels = mid_channels
        downsample = None

        if stereo:
            depth_conv_input_channels += depth_channels
            downsample = nn.Conv2d(depth_conv_input_channels,
                                   mid_channels, 1, 1, 0)
            cost_volumn_net = []
            for stage in range(int(2)):
                cost_volumn_net.extend([
                    nn.Conv2d(depth_channels, depth_channels, kernel_size=3,
                              stride=2, padding=1),
                    nn.BatchNorm2d(depth_channels)])
            self.cost_volumn_net = nn.Sequential(*cost_volumn_net)
            self.bias = bias

        # 9. 深度特征处理主干
        depth_conv_list = [
            BasicBlock(depth_conv_input_channels, mid_channels,
                       downsample=downsample),
            BasicBlock(mid_channels, mid_channels),
            BasicBlock(mid_channels, mid_channels)
        ]

        # 10. 多尺度特征融合
        depth_conv_list.append(self._make_fusion_block(mid_channels))

        if use_aspp:
            if aspp_mid_channels < 0:
                aspp_mid_channels = mid_channels
            depth_conv_list.append(ASPP(mid_channels, aspp_mid_channels))

        if use_dcn:
            depth_conv_list.append(
                build_conv_layer(
                    cfg=dict(
                        type='DCN',
                        in_channels=mid_channels,
                        out_channels=mid_channels,
                        kernel_size=3,
                        padding=1,
                        groups=4,
                        im2col_step=128,
                    )))

        # 11. 深度预测层
        depth_conv_list.append(
            nn.Conv2d(
                mid_channels,
                depth_channels,
                kernel_size=1,
                stride=1,
                padding=0))

        self.depth_conv = nn.Sequential(*depth_conv_list)
        self.with_cp = with_cp
        self.depth_channels = depth_channels

        # 12. 深度细化卷积
        self.refine_conv = nn.Sequential(
            nn.Conv2d(depth_channels + context_channels, depth_channels, 3, padding=1),
            nn.BatchNorm2d(depth_channels),
            nn.ReLU(),
            nn.Conv2d(depth_channels, depth_channels, 1)
        )


    def gen_grid(self, metas, B, N, D, H, W, hi, wi):
        """
        Args:
            metas: dict{
                k2s_sensor: (B, N_views, 4, 4)
                intrins: (B, N_views, 3, 3)
                post_rots: (B, N_views, 3, 3)
                post_trans: (B, N_views, 3)
                frustum: (D, fH_stereo, fW_stereo, 3)  3:(u, v, d)
                cv_downsample: 4,
                downsample: self.img_view_transformer.downsample=16,
                grid_config: self.img_view_transformer.grid_config,
                cv_feat_list: [feat_prev_iv, stereo_feat]
            }
            B: batchsize
            N: N_views
            D: D
            H: fH_stereo
            W: fW_stereo
            hi: H_img
            wi: W_img
        Returns:
            grid: (B*N_views, D*fH_stereo, fW_stereo, 2)
        """
        frustum = metas['frustum']      # (D, fH_stereo, fW_stereo, 3)  3:(u, v, d)
        # 逆图像增广:
        points = frustum - metas['post_trans'].view(B, N, 1, 1, 1, 3)
        points = torch.inverse(metas['post_rots']).view(B, N, 1, 1, 1, 3, 3) \
            .matmul(points.unsqueeze(-1))   # (B, N_views, D, fH_stereo, fW_stereo, 3, 1)

        # (u, v, d) --> (du, dv, d)
        # (B, N_views, D, fH_stereo, fW_stereo, 3, 1)
        points = torch.cat(
            (points[..., :2, :] * points[..., 2:3, :], points[..., 2:3, :]), 5)

        # cur_pixel --> curr_camera --> prev_camera
        rots = metas['k2s_sensor'][:, :, :3, :3].contiguous()
        trans = metas['k2s_sensor'][:, :, :3, 3].contiguous()
        combine = rots.matmul(torch.inverse(metas['intrins']))
        points = combine.view(B, N, 1, 1, 1, 3, 3).matmul(points)
        points += trans.view(B, N, 1, 1, 1, 3, 1)   # (B, N_views, D, fH_stereo, fW_stereo, 3, 1)

        neg_mask = points[..., 2, 0] < 1e-3
        # prev_camera --> prev_pixel
        points = metas['intrins'].view(B, N, 1, 1, 1, 3, 3).matmul(points)
        # (du, dv, d) --> (u, v)   (B, N_views, D, fH_stereo, fW_stereo, 2, 1)
        points = points[..., :2, :] / points[..., 2:3, :]

        # 图像增广
        points = metas['post_rots'][..., :2, :2].view(B, N, 1, 1, 1, 2, 2).matmul(
            points).squeeze(-1)
        points += metas['post_trans'][..., :2].view(B, N, 1, 1, 1, 2)   # (B, N_views, D, fH_stereo, fW_stereo, 2)

        px = points[..., 0] / (wi - 1.0) * 2.0 - 1.0
        py = points[..., 1] / (hi - 1.0) * 2.0 - 1.0
        px[neg_mask] = -2
        py[neg_mask] = -2
        grid = torch.stack([px, py], dim=-1)    # (B, N_views, D, fH_stereo, fW_stereo, 2)
        grid = grid.view(B * N, D * H, W, 2)    # (B*N_views, D*fH_stereo, fW_stereo, 2)
        return grid

    # 辅助函数：创建残差块
    def _make_res_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )

    # 辅助函数：创建特征融合块
    def _make_fusion_block(self, channels):
        return nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU()
        )

    # 代价体生成函数（改进版）
    # def calculate_cost_volumn(self, metas):
    #     prev, curr = metas['cv_feat_list']
    #     group_size = 4
    #     _, c, hf, wf = curr.shape
    #     hi, wi = hf * 4, wf * 4
    #     B, N, _ = metas['post_trans'].shape
    #     D, H, W, _ = metas['frustum'].shape
    #     grid = self.gen_grid(metas, B, N, D, H, W, hi, wi).to(curr.dtype)
    #
    #     prev = prev.view(B * N, -1, H, W)
    #     curr = curr.view(B * N, -1, H, W)
    #
    #     cost_volumn = 0
    #     for fid in range(curr.shape[1] // group_size):
    #         prev_curr = prev[:, fid * group_size:(fid + 1) * group_size, ...]
    #         wrap_prev = F.grid_sample(prev_curr, grid,
    #                                   align_corners=True,
    #                                   padding_mode='zeros')
    #
    #         curr_tmp = curr[:, fid * group_size:(fid + 1) * group_size, ...]
    #
    #         # 改进：使用相关操作代替绝对差
    #         correlation = torch.sum(wrap_prev * curr_tmp.unsqueeze(2), dim=1)
    #         cost_volumn += correlation
    #
    #     # 添加可学习的匹配代价调整
    #     cost_volumn = self.cost_adjust(cost_volumn)
    #
    #     # 添加不确定性加权
    #     if not self.bias == 0:
    #         invalid = wrap_prev[:, 0, ...].view(B * N, D, H, W) == 0
    #         cost_volumn[invalid] = cost_volumn[invalid] + self.bias
    #
    #     # 匹配代价转换为概率
    #     cost_volumn = -cost_volumn
    #     cost_volumn = F.softmax(cost_volumn, dim=1)
    #     return cost_volumn

    def calculate_cost_volumn(self, metas):
        """
        Args:
            metas: dict{
                k2s_sensor: (B, N_views, 4, 4)
                intrins: (B, N_views, 3, 3)
                post_rots: (B, N_views, 3, 3)
                post_trans: (B, N_views, 3)
                frustum: (D, fH_stereo, fW_stereo, 3)  3:(u, v, d)
                cv_downsample: 4,
                downsample: self.img_view_transformer.downsample=16,
                grid_config: self.img_view_transformer.grid_config,
                cv_feat_list: [feat_prev_iv, stereo_feat]
            }
        Returns:
            cost_volumn: (B*N_views, D, fH_stereo, fW_stereo)
        """
        prev, curr = metas['cv_feat_list']    # (B*N_views, C_stereo, fH_stereo, fW_stereo)
        group_size = 4
        _, c, hf, wf = curr.shape   #
        hi, wi = hf * 4, wf * 4     # H_img, W_img
        B, N, _ = metas['post_trans'].shape
        D, H, W, _ = metas['frustum'].shape
        grid = self.gen_grid(metas, B, N, D, H, W, hi, wi).to(curr.dtype)   # (B*N_views, D*fH_stereo, fW_stereo, 2)

        prev = prev.view(B * N, -1, H, W)   # (B*N_views, C_stereo, fH_stereo, fW_stereo)
        curr = curr.view(B * N, -1, H, W)   # (B*N_views, C_stereo, fH_stereo, fW_stereo)
        cost_volumn = 0
        # process in group wise to save memory
        for fid in range(curr.shape[1] // group_size):
            # (B*N_views, group_size, fH_stereo, fW_stereo)
            prev_curr = prev[:, fid * group_size:(fid + 1) * group_size, ...]
            wrap_prev = F.grid_sample(prev_curr, grid,
                                      align_corners=True,
                                      padding_mode='zeros')     # (B*N_views, group_size, D*fH_stereo, fW_stereo)
            # (B*N_views, group_size, fH_stereo, fW_stereo)
            curr_tmp = curr[:, fid * group_size:(fid + 1) * group_size, ...]
            # (B*N_views, group_size, 1, fH_stereo, fW_stereo) - (B*N_views, group_size, D, fH_stereo, fW_stereo)
            # --> (B*N_views, group_size, D, fH_stereo, fW_stereo)
            # https://github.com/HuangJunJie2017/BEVDet/issues/278
            cost_volumn_tmp = curr_tmp.unsqueeze(2) - \
                              wrap_prev.view(B * N, -1, D, H, W)
            cost_volumn_tmp = cost_volumn_tmp.abs().sum(dim=1)      # (B*N_views, D, fH_stereo, fW_stereo)
            cost_volumn += cost_volumn_tmp  # (B*N_views, D, fH_stereo, fW_stereo)
        if not self.bias == 0:
            invalid = wrap_prev[:, 0, ...].view(B * N, D, H, W) == 0
            cost_volumn[invalid] = cost_volumn[invalid] + self.bias

        # matching cost --> prob
        cost_volumn = - cost_volumn
        cost_volumn = cost_volumn.softmax(dim=1)
        return cost_volumn
    # ----------------------------------------- 用于建立cost volume -----

    def forward(self, x, mlp_input, stereo_metas=None):
        # 1. 多尺度特征提取
        x = self.multiscale_conv(x)

        # 2. 处理相机参数
        mlp_input = self.bn(mlp_input.reshape(-1, mlp_input.shape[-1]))

        # 3. 深度特征分支
        depth_feat = self.reduce_conv(x)

        # 4. 上下文特征分支
        context_feat = x

        # 5. 相机感知注意力
        context_se = self.context_mlp(mlp_input)[..., None, None]
        context_feat = self.context_se(context_feat, context_se)
        context = self.context_conv(context_feat)

        depth_se = self.depth_mlp(mlp_input)[..., None, None]
        depth_feat = self.depth_se(depth_feat, depth_se)

        # 6. 深度注意力机制
        attention_map = self.depth_attention(depth_feat)
        depth_feat = depth_feat * (1 + attention_map)

        # 7. 处理立体视觉信息
        if stereo_metas is not None:
            if stereo_metas['cv_feat_list'][0] is None:
                BN, _, H, W = x.shape
                scale_factor = float(stereo_metas['downsample']) / stereo_metas['cv_downsample']
                cost_volumn = torch.zeros((BN, self.depth_channels,
                                           int(H * scale_factor),
                                           int(W * scale_factor))).to(x)
            else:
                with torch.no_grad():
                    cost_volumn = self.calculate_cost_volumn(stereo_metas)
            cost_volumn = self.cost_volumn_net(cost_volumn)
            depth_feat = torch.cat([depth_feat, cost_volumn], dim=1)

        # 8. 深度特征处理
        if self.with_cp:
            depth = checkpoint(self.depth_conv, depth_feat)
        else:
            depth = self.depth_conv(depth_feat)

        # 9. 深度不确定性估计
        uncertainty = self.uncertainty_conv(depth)

        # 10. 使用不确定性加权深度特征
        depth = depth * (1 + uncertainty)

        # 11. 深度细化（融合上下文信息）
        combined = torch.cat([depth, context], dim=1)
        depth = self.refine_conv(combined) + depth

        return torch.cat([depth, context], dim=1)

class DepthAggregation(nn.Module):
    """pixel cloud feature extraction."""

    def __init__(self, in_channels, mid_channels, out_channels):
        super(DepthAggregation, self).__init__()

        self.reduce_conv = nn.Sequential(
            nn.Conv2d(
                in_channels,
                mid_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
        )

        self.conv = nn.Sequential(
            nn.Conv2d(
                mid_channels,
                mid_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                mid_channels,
                mid_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
        )

        self.out_conv = nn.Sequential(
            nn.Conv2d(
                mid_channels,
                out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=True),
            # nn.BatchNorm3d(out_channels),
            # nn.ReLU(inplace=True),
        )

    @autocast(False)
    def forward(self, x):
        x = checkpoint(self.reduce_conv, x)
        short_cut = x
        x = checkpoint(self.conv, x)
        x = short_cut + x
        x = self.out_conv(x)
        return x