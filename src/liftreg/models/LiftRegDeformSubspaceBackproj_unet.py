import torch
import torch.nn as nn
import torch.nn.functional as F
from ..layers.layers import convBlock, FullyConnectBlock, GaussianSmoothing
from ..utils.net_utils import Bilinear, gen_identity_map
import numpy as np
from ..utils.sdct_projection_utils import backproj_grids_with_poses, backproj_grids_with_SOUV
from diffdrr.detector import Detector
from diffdrr.renderers import Siddon, Trilinear

class ResidualBlock(nn.Module):
    """残差块"""
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, 
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, 
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(out_channels)
        
        # 残差连接的投影层
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1, 
                         stride=stride, bias=False),
                nn.BatchNorm3d(out_channels)
            )
    
    def forward(self, x):
        residual = self.shortcut(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        out = self.relu(out)
        return out


class EncoderBlock(nn.Module):
    """编码器块（下采样）"""
    def __init__(self, in_channels, out_channels, stride=2):
        super(EncoderBlock, self).__init__()
        self.res_block = ResidualBlock(in_channels, out_channels, stride=stride)
    
    def forward(self, x):
        return self.res_block(x)


class DecoderBlock(nn.Module):
    """解码器块（上采样 + 跳跃连接）"""
    def __init__(self, in_channels, skip_channels, out_channels):
        super(DecoderBlock, self).__init__()
        # 上采样
        self.upsample = nn.ConvTranspose3d(
            in_channels, in_channels, 
            kernel_size=2, stride=2
        )
        # 融合跳跃连接后的残差块
        self.res_block = ResidualBlock(
            in_channels + skip_channels, 
            out_channels, 
            stride=1
        )
    
    def forward(self, x, skip):
        x = self.upsample(x)
        # 确保尺寸匹配
        if x.shape != skip.shape:
            x = F.interpolate(x, size=skip.shape[2:], mode='trilinear', align_corners=True)
        x = torch.cat([x, skip], dim=1)
        x = self.res_block(x)
        return x


class model(nn.Module):
    """
    ResUNet for 2D/3D registration with subspace learning.
    
    :param img_sz: Voxel shape.
    :param opt: setting for the network.
    """
    def __init__(self, img_sz, opt=None):
        super(model, self).__init__()
        
        self.img_sz = img_sz
        self.gaussian_smooth = GaussianSmoothing(4, 8, 2, dim=2)
        self.bilinear = Bilinear(zero_boundary=True, using_scale=True)
        
        # 特征通道数设置
        base_channels = 16
        enc_channels = [base_channels, base_channels*2, base_channels*4, 
                       base_channels*8, base_channels*8, base_channels*8]
        
        # ===== 编码器路径 (Encoder Path) =====
        self.encoders = nn.ModuleList()
        
        # 初始卷积（不下采样）
        self.init_conv = ResidualBlock(
            opt["drr_feature_num"]+1,  # 输入通道：proj_num + 1
            enc_channels[0], 
            stride=1
        )
        
        # 编码器块（逐步下采样）
        for i in range(len(enc_channels)-1):
            self.encoders.append(
                EncoderBlock(enc_channels[i], enc_channels[i+1], stride=2)
            )
        
        # ===== 瓶颈层 (Bottleneck) =====
        bottleneck_channels = enc_channels[-1]
        self.bottleneck = ResidualBlock(bottleneck_channels, bottleneck_channels, stride=1)
        
        # ===== 解码器路径 (Decoder Path) =====
        self.decoders = nn.ModuleList()
        dec_channels = enc_channels[::-1]  # 反转通道数列表
        
        for i in range(len(dec_channels)-1):
            self.decoders.append(
                DecoderBlock(
                    dec_channels[i],      # 输入通道
                    dec_channels[i+1],    # 跳跃连接通道
                    dec_channels[i+1]     # 输出通道
                )
            )
        
        # ===== 输出层 =====
        # 方案1: 使用 PCA 子空间（保持原有方式）
        if opt["use_pca"]:
            # 全局平均池化 + 全连接层
            self.use_pca = True
            self.global_pool = nn.AdaptiveAvgPool3d(1)
            self.fc_layers = nn.Sequential(
                nn.Flatten(),
                nn.Linear(base_channels, 256),
                nn.ReLU(inplace=True),
                nn.Linear(256, opt["latent_dim"])
            )
            
            # 加载 PCA 组件
            self.pca_vectors = torch.from_numpy(
                np.load(f"{opt['pca_path']}/pca_vectors.npy").T
            ).float().cuda()
            self.pca_mean = torch.from_numpy(
                np.load(f"{opt['pca_path']}/pca_mean.npy")
            ).float().cuda()
        else:
            # 方案2: 直接输出 displacement field
            self.use_pca = False
            self.output_conv = nn.Conv3d(
                base_channels, 3,  # 输出3通道 (x, y, z displacement)
                kernel_size=3, padding=1
            )
        
        self.id_transform = gen_identity_map(self.img_sz, 1.0)
        self.backward_proj_grids = None
        self.detector = Detector(
            1020.0,
            718,
            718,
            0.388,
            0.388,
            0,
            0,
            torch.eye(3),
            reverse_x_axis=False,
            n_subsample=None,
        )
        # self.renderer = Siddon(voxel_shift, **renderer_kwargs)
    def forward(self, input):
        # Parse input
        moving = input['source']
        # target = input['target']
        target_proj = input["target_proj"]
        
        if 'source_label' in input:
            moving_seg = input['source_label']
            # target_seg = input['target_label']
            moving_cp = (moving+1)*moving_seg-1
            # target_cp = (target+1)*target_seg-1
        else:
            moving_cp = moving
            # target_cp = target
        
        B, _, D, W, H = moving.shape
        target_poses = input['target_poses']
        # source, target = self.detector(target_poses, calibration)
        # img = self.render(self.density, source, target, mask_to_channels, **kwargs)
        # 估计形变场
        coefs, disp_field = self._estimate_flow(moving, target_proj, target_poses)
        
        # 应用形变
        deform_field = disp_field + self.id_transform
        warped_source = self.bilinear(moving_cp, deform_field)
        
        model_output = {
            "warped": warped_source,
            "phi": deform_field,
            "params": disp_field,
            # "target": target_cp,
            "pca_coefs": coefs,
            "target_proj": target_proj,
            "warped_proj": target_proj
        }
        return model_output
    
    def _estimate_flow(self, moving, target_proj, poses):
        """使用 ResUNet 估计形变场"""
        batch_size, proj_num, proj_w, proj_h = target_proj.shape
        w, d, h = moving.shape[2:]
        B, _, D, W, H = moving.shape
        S, O, U, V = poses
        
        # 反向投影
        if self.backward_proj_grids is None:
            with torch.no_grad():
                self.backward_proj_grids = backproj_grids_with_SOUV(
                    moving.shape[2:], 
                    S, O, U, V,
                    proj_w, proj_h,
                    du=0.388, dv=0.388,
                    cu=(proj_w - 1) / 2.0, cv=(proj_h - 1) / 2.0,
                    device=moving.device
                )

        # target_volume = F.grid_sample(
        #     target_proj.reshape(batch_size*proj_num, 1, proj_w, proj_h), 
        #     self.backward_proj_grids.reshape(
        #         batch_size*proj_num, w*d, h, -1
        #     ),
        #     align_corners=True,
        #     padding_mode="zeros"
        # ).reshape(batch_size, proj_num, w, d, h).detach()
        
        # 1) 先把 grid reshape 出来（明确最后一维=2）
        grid = self.backward_proj_grids.reshape(
            batch_size * proj_num, w * d, h, 2
        )

        # 2) 正常 grid_sample（padding_mode 仍然用 zeros）
        target_volume_flat = F.grid_sample(
            target_proj.reshape(batch_size * proj_num, 1, proj_w, proj_h),
            grid,
            align_corners=True,
            padding_mode="zeros"
        )  # -> (B*P, 1, w*d, h)

        # 3) 计算越界 mask：grid 超出 [-1, 1] 就是 padding 区域
        invalid = (
            (grid[..., 0] < -1.0) | (grid[..., 0] > 1.0) |
            (grid[..., 1] < -1.0) | (grid[..., 1] > 1.0)
        )  # (B*P, w*d, h)

        # 4) 扩到和输出同形状 (B*P, 1, w*d, h)，并把 padding 区域改成 -1
        target_volume_flat[invalid.unsqueeze(1)] = -1.0

        # 5) reshape 回你要的形状
        target_volume = target_volume_flat.reshape(
            batch_size, proj_num, w, d, h
        ).detach()
        # 拼接输入
        x = torch.cat([moving, target_volume], dim=1)
        
        # ===== 编码器路径 =====
        x = self.init_conv(x)
        
        # 保存跳跃连接
        skip_connections = [x]
        
        for encoder in self.encoders:
            x = encoder(x)
            skip_connections.append(x)
        
        # ===== 瓶颈层 =====
        x = self.bottleneck(x)
        
        # ===== 解码器路径 =====
        skip_connections = skip_connections[::-1][1:]  # 反转并移除最后一个
        
        for i, decoder in enumerate(self.decoders):
            x = decoder(x, skip_connections[i])
        
        # ===== 输出层 =====
        if self.use_pca:
            # 使用 PCA 子空间
            features = self.global_pool(x)
            coefs = self.fc_layers(features)
            disp_field = F.linear(coefs, self.pca_vectors, self.pca_mean).reshape(
                B, 3, D, W, H
            )
        else:
            # 直接输出 displacement field
            disp_field = self.output_conv(x)
            coefs = None
        
        return coefs, disp_field
    
    def get_extra_to_plot(self):
        return None, None
    
    def get_disp(self):
        return None, ""