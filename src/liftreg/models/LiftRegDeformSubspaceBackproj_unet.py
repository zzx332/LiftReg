import SimpleITK as sitk
import torch
import torch.nn as nn
import torch.nn.functional as F
from ..layers.layers import convBlock, FullyConnectBlock, GaussianSmoothing
from ..utils.net_utils import Bilinear, gen_identity_map
import numpy as np
from diffdrr.detector import Detector
from diffdrr.renderers import Siddon, Trilinear
from diffdrr.pose import RigidTransform

class ResidualBlock(nn.Module):
    """Residual block"""
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, 
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, 
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(out_channels)
        
        # Residual connection projection layer
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
    """Encoder block (downsampling)"""
    def __init__(self, in_channels, out_channels, stride=2):
        super(EncoderBlock, self).__init__()
        self.res_block = ResidualBlock(in_channels, out_channels, stride=stride)
    
    def forward(self, x):
        return self.res_block(x)


class DecoderBlock(nn.Module):
    """Decoder block (upsampling + skip connection)"""
    def __init__(self, in_channels, skip_channels, out_channels):
        super(DecoderBlock, self).__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode="trilinear", align_corners=False)
        self.res_block = ResidualBlock(
            in_channels + skip_channels,
            out_channels,
            stride=1,
        )
        self.res_block.relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x, skip):
        x = self.upsample(x)
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode="trilinear", align_corners=False)
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
        self.memory_logger = None
        # Feature channel settings
        base_channels = 16
        enc_channels = [base_channels, base_channels*2, base_channels*4, 
                       base_channels*8, base_channels*8, base_channels*8]
        
        # ===== Encoder Path (Encoder Path) =====
        self.encoders = nn.ModuleList()
        
        # Initial convolution (no downsampling)
        self.init_conv = ResidualBlock(
            opt["drr_feature_num"]+1,  # Input channels: proj_num + 1
            enc_channels[0], 
            stride=1
        )
        
        # Encoder blocks (gradually downsampling)
        for i in range(len(enc_channels)-1):
            self.encoders.append(
                EncoderBlock(enc_channels[i], enc_channels[i+1], stride=2)
            )
        
        # ===== Bottleneck Layer (Bottleneck) =====
        bottleneck_channels = enc_channels[-1]
        self.bottleneck = ResidualBlock(bottleneck_channels, bottleneck_channels, stride=1)
        
        # ===== Decoder Path (Decoder Path) =====
        self.decoders = nn.ModuleList()
        dec_channels = enc_channels[::-1]  # Reverse channel number list
        
        for i in range(len(dec_channels)-1):
            self.decoders.append(
                DecoderBlock(
                    dec_channels[i],      # Input channels
                    dec_channels[i+1],    # Skip connection channels
                    dec_channels[i+1]     # Output channels
                )
            )
        
        # ===== Output Layer =====
        # Solution 1: Using PCA subspace (keep original way)
        if opt["use_pca"]:
            # Global average pooling + fully connected layers
            self.use_pca = True
            self.global_pool = nn.AdaptiveAvgPool3d(1)
            self.fc_layers = nn.Sequential(
                nn.Flatten(),
                nn.Linear(base_channels, 256),
                nn.ReLU(inplace=True),
                nn.Linear(256, opt["latent_dim"])
            )
            
            # Load PCA components
            self.pca_vectors = torch.from_numpy(
                np.load(f"{opt['pca_path']}/pca_vectors.npy").T
            ).float().cuda()
            self.pca_mean = torch.from_numpy(
                np.load(f"{opt['pca_path']}/pca_mean.npy")
            ).float().cuda()
        else:
            # Solution 2: Directly output displacement field
            self.use_pca = False
            self.output_conv = nn.Conv3d(
                base_channels, 3,  # Output 3 channels (x, y, z displacement)
                kernel_size=3, padding=1
            )
            self.output_conv.weight.data.normal_(mean=0.0, std=1e-5)
            self.output_conv.bias.data.zero_()
        
        self.id_transform = gen_identity_map(self.img_sz, 1.0)
        # self.backward_proj_grids = None
        self.render = Siddon(voxel_shift=0.0)
        reorient = torch.tensor(
            [
                [1, 0, 0, 0],
                [0, 0, 1, 0],
                [0, 1, 0, 0],
                [0, 0, 0, 1],
            ],
            dtype=torch.float32,
        )
        self.detector = Detector(
            1020.0,
            384,
            384,
            0.7255,
            0.7255,
            0,
            0,
            reorient,
            reverse_x_axis=True,
            n_subsample=None,
        ).cuda()
        # self.detector = Detector(
        #     1020.0,
        #     718,
        #     718,
        #     0.388,
        #     0.388,
        #     0,
        #     0,
        #     reorient,
        #     reverse_x_axis=True,
        #     n_subsample=None,
        # ).cuda() 

    def set_memory_logger(self, logger):
        self.memory_logger = logger

    def _log_memory(self, stage):
        if self.memory_logger is not None:
            self.memory_logger(stage)

    def forward(self, input):
        # Parse input
        moving = input['source']
        target_proj = input["target_proj"]
        target_volume = input["target_volume"]
        if 'source_label' in input:
            moving_seg = input['source_label']
            # target_seg = input['target_label']
            moving_cp = (moving+1)*moving_seg-1
            # target_cp = (target+1)*target_seg-1
        else:
            moving_cp = moving
            # target_cp = target
        
        B, _, W, H, D = moving.shape
        density = input['density']
        target_poses = input['target_poses'].to(density.device)
        affine_inverse = input['affine'].inverse().to(density.device)
        # Estimating Deformation Fields
        coefs, disp_field = self._estimate_flow(moving, target_volume)
        self._log_memory("model/after_estimate_flow")
        # Applying Deformation
        disp_norm = disp_field.clone()
        disp_norm[:, 0] = disp_norm[:, 0] * (2.0 / (W - 1))  # x
        disp_norm[:, 1] = disp_norm[:, 1] * (2.0 / (H - 1))  # y
        disp_norm[:, 2] = disp_norm[:, 2] * (2.0 / (D - 1))  # z
        deform_field = disp_norm + self.id_transform
        self._log_memory("model/after_deform_field")
        warped_moving = self.bilinear(density, deform_field)
        self._log_memory("model/after_warp")
        warped_proj = self.renderer(warped_moving[0, 0], target_poses, affine_inverse).view(B,
            -1,
            self.detector.height,
            self.detector.width)
        self._log_memory("model/after_full_projection_render")
 
        model_output = {
            # "warped": warped_source,
            "warped_moving": warped_moving,
            "phi": deform_field,
            "params": disp_field,
            # "target": target_cp,
            "pca_coefs": coefs,
            "target_proj": target_proj,
            "warped_proj": warped_proj
        }
        return model_output
        
    def renderer(self, density, target_poses, affine_inverse):
        target_poses = RigidTransform(target_poses)
        source, target = self.detector(target_poses, calibration=None)
        # Initialize the image with the length of each cast ray
        img_input = (target - source).norm(dim=-1).unsqueeze(1)
        # Convert rays to voxelspace
        source = RigidTransform(affine_inverse)(source)
        target = RigidTransform(affine_inverse)(target)
        return self.render(density, source, target, img_input)

    def normalize_intensity(self, img, linear_clip=False, clip_range=None):
        """
        a numpy image, normalize into intensity [-1,1]
        (img-img.min())/(img.max() - img.min())
        :param img: image
        :param linear_clip:  Linearly normalized image intensities so that the 95-th percentile gets mapped to 0.95; 0 stays 0
        :return:
        """

        if linear_clip:
            if clip_range is not None:
                img[img<clip_range[0]] = clip_range[0]
                img[img>clip_range[1]] = clip_range[1]
                normalized_img = (img-clip_range[0]) / (clip_range[1] - clip_range[0]) 
            else:
                img = img - img.min()
                normalized_img =img / np.percentile(img, 95) * 0.95
        else:
            # If we normalize in HU range of softtissue
            min_intensity = img.min()
            max_intensity = img.max()
            normalized_img = (img-img.min())/(max_intensity - min_intensity)
        normalized_img = normalized_img * 2 - 1
        return normalized_img

    def inverse_normalization(self, img, clip_range=[-1000, 1000]):
        img = (img + 1) / 2
        img = (img * (clip_range[1] - clip_range[0])) + clip_range[0]
        return img

    def transform_hu_to_density(self, volume, bone_attenuation_multiplier):
        # volume can be loaded as int16, need to convert to float32 to use float bone_attenuation_multiplier
        volume = volume.to(torch.float32)
        air = torch.where(volume <= -800)
        soft_tissue = torch.where((-800 < volume) & (volume <= 350))
        bone = torch.where(350 < volume)

        density = torch.empty_like(volume)
        density[air] = volume[soft_tissue].min()
        density[soft_tissue] = volume[soft_tissue]
        density[bone] = volume[bone] * bone_attenuation_multiplier

        # Important: avoid in-place arithmetic, change to out-of-place
        density_min = density.min()
        density = density - density_min
        density_max = density.max()
        density = density / density_max

        return density

    def _estimate_flow(self, moving, target_volume):
        """Estimating Deformation Fields using ResUNet"""
        B, _, W, H, D = moving.shape

        # Concatenate input
        x = torch.cat([moving, target_volume], dim=1)
        
        # ===== Encoder Path =====
        x = self.init_conv(x)
        
        # Save skip connections
        skip_connections = [x]
        
        for encoder in self.encoders:
            x = encoder(x)
            skip_connections.append(x)
        
        # ===== Bottleneck Layer =====
        x = self.bottleneck(x)
        
        # ===== Decoder Path =====
        skip_connections = skip_connections[::-1][1:]  # Reverse and remove the last one
        
        for i, decoder in enumerate(self.decoders):
            x = decoder(x, skip_connections[i])
        
        # ===== Output Layer =====
        if self.use_pca:
            # Using PCA subspace
            features = self.global_pool(x)
            coefs = self.fc_layers(features)
            disp_field = F.linear(coefs, self.pca_vectors, self.pca_mean).reshape(
                B, 3, D, H, W
            )
        else:
            # Directly output displacement field
            disp_field = self.output_conv(x)
            coefs = None
        
        return coefs, disp_field
    
    def get_extra_to_plot(self):
        return None, None
    
    def get_disp(self):
        return None, ""