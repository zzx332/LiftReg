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
        # self.bn1 = nn.InstanceNorm3d(out_channels)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, 
                               stride=1, padding=1, bias=False)
        # self.bn2 = nn.InstanceNorm3d(out_channels)
        self.bn2 = nn.BatchNorm3d(out_channels)
        
        # Residual connection projection layer
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1, 
                         stride=stride, bias=False),
                # nn.InstanceNorm3d(out_channels)
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
        self.use_pca = opt["use_pca"]
        self.use_polyrigid = opt["use_polyrigid"]

        if self.use_pca:
            self.global_pool = nn.AdaptiveAvgPool3d(1)
            self.fc_layers = nn.Sequential(
                nn.Flatten(),
                nn.Linear(base_channels, 256),
                nn.ReLU(inplace=True),
                nn.Linear(256, opt["latent_dim"])
            )
            self.pca_vectors = torch.from_numpy(
                np.load(f"{opt['pca_path']}/pca_vectors.npy").T
            ).float().cuda()
            self.pca_mean = torch.from_numpy(
                np.load(f"{opt['pca_path']}/pca_mean.npy")
            ).float().cuda()
        elif self.use_polyrigid:
            self.num_segments = opt["num_segments"]
            self.global_pool = nn.AdaptiveAvgPool3d(1)
            self.fc_polyrigid = nn.Sequential(
                nn.Flatten(),
                nn.Linear(base_channels, 128),
                nn.ReLU(inplace=True),
                nn.Linear(128, self.num_segments * 3),
            )
            self.fc_polyrigid[-1].weight.data.normal_(mean=0.0, std=0.1)
            self.fc_polyrigid[-1].bias.data.zero_()
        else:
            self.output_conv = nn.Conv3d(
                base_channels, 3,
                kernel_size=3, padding=1
            )
            self.output_conv.weight.data.normal_(mean=0.0, std=1e-5)
            self.output_conv.bias.data.zero_()
        
        self.id_transform = gen_identity_map(self.img_sz, 1.0)
        # self.backward_proj_grids = None
        self.render = Siddon(voxel_shift=0.5)
        self.reorient = torch.tensor(
            [
                [1, 0, 0, 0],
                [0, 0, 1, 0],
                [0, 1, 0, 0],
                [0, 0, 0, 1],
            ],
            dtype=torch.float32,
        )
        self.sdd = 1020.0
        self.infer_detector_width = 384
        self.proj_spacing = 0.7255
        self.detector = Detector(
            self.sdd,
            self.infer_detector_width,
            self.infer_detector_width,
            self.proj_spacing,
            self.proj_spacing,
            0,
            0,
            self.reorient,
            reverse_x_axis=True,
            n_subsample=None,
        ).float().cuda()

    @staticmethod
    def _to_float_scalar(value):
        if torch.is_tensor(value):
            return float(value.detach().cpu().item())
        return float(value)

    def _batch_scalar(self, values, batch_idx, flat_index=0):
        """Read one scalar from a collated per-sample field (0-d or 1-d)."""
        entry = values[batch_idx]
        if torch.is_tensor(entry) and entry.ndim > 0:
            entry = entry.reshape(-1)[flat_index]
        return self._to_float_scalar(entry)

    def _build_detector(self, sdd, width, delx, dely, device):
        width = int(self._to_float_scalar(width))
        return Detector(
            self._to_float_scalar(sdd),
            width,
            width,
            self._to_float_scalar(delx),
            self._to_float_scalar(dely),
            0,
            0,
            self.reorient.to(device=device, dtype=torch.float32),
            reverse_x_axis=True,
            n_subsample=None,
        ).float().to(device)

    def set_memory_logger(self, logger):
        self.memory_logger = logger

    def _log_memory(self, stage):
        if self.memory_logger is not None:
            self.memory_logger(stage)

    def load_input_params(self, input):
        moving = input['source']
        target_proj = input["target_proj"]
        target_volume = input["target_volume"]
        if 'source_label' in input:
            moving_seg = input['source_label']
            moving_cp = (moving + 1) * moving_seg - 1
        else:
            moving_cp = moving
        density = input['density']
        target_poses = input['target_poses'].to(
            device=density.device, dtype=torch.float32)
        affine_inverse = input['affine'].to(
            device=density.device, dtype=torch.float32).inverse()

        return moving, target_proj, target_volume, moving_cp, density, target_poses, affine_inverse

    def renderer(self, density, target_poses, affine_inverse):
        target_poses = RigidTransform(target_poses.float())
        source, target = self.detector(target_poses, calibration=None)
        # Initialize the image with the length of each cast ray
        img_input = (target - source).norm(dim=-1).unsqueeze(1)
        # Convert rays to voxelspace
        affine_inverse = RigidTransform(affine_inverse.float())
        source = affine_inverse(source)
        target = affine_inverse(target)
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
        """Estimate deformation parameters using ResUNet encoder-decoder."""
        B, _, W, H, D = moving.shape

        x = torch.cat([moving, target_volume], dim=1)

        x = self.init_conv(x)
        skip_connections = [x]

        for encoder in self.encoders:
            x = encoder(x)
            skip_connections.append(x)

        x = self.bottleneck(x)

        skip_connections = skip_connections[::-1][1:]
        for i, decoder in enumerate(self.decoders):
            x = decoder(x, skip_connections[i])

        if self.use_pca:
            features = self.global_pool(x)
            coefs = self.fc_layers(features)
            disp_field = F.linear(coefs, self.pca_vectors, self.pca_mean).reshape(
                B, 3, D, H, W
            )
            return coefs, disp_field
        elif self.use_polyrigid:
            features = self.global_pool(x)
            raw = self.fc_polyrigid(features)  # (B, K*3)
            polyrigid_params = raw.view(B, self.num_segments, 3)
            return polyrigid_params, None
        else:
            disp_field = self.output_conv(x)
            return None, disp_field

    def _polyrigid_to_displacement(self, polyrigid_params, weights, affine, spatial_shape):
        """
        Pure-translation polyrigid: weighted sum of per-segment translation vectors.

        Each voxel's displacement = sum_k( w_k * t_k ), where t_k is the predicted
        translation for segment k and w_k is its segmentation weight at that voxel.

        No SE3 exponential map, no affine composition — a single einsum suffices.

        Args:
            polyrigid_params: (B, K, 3) predicted translation per segment (voxel units)
            weights: (B, K, W, H, D) segmentation blending weights
            affine:  unused (kept for interface compatibility)
            spatial_shape: unused (kept for interface compatibility)

        Returns:
            disp_field: (B, 3, W, H, D) dense voxel-unit displacement field
        """
        # weights: (B, K, W, H, D)   polyrigid_params: (B, K, 3)
        # result:  (B, 3, W, H, D)
        disp_field = torch.einsum("bkwhd,bkn->bnwhd", weights, polyrigid_params)
        return disp_field
    
    def get_extra_to_plot(self):
        return None, None
    
    def get_disp(self):
        return None, ""

    def _preprocess_proj(self, proj, mask):
        def robust_normalization(img, lower_percentile=1, upper_percentile=99):
            q = torch.tensor(
                [lower_percentile / 100.0, upper_percentile / 100.0],
                device=img.device, dtype=img.dtype,
            )
            p_min, p_max = torch.quantile(img, q)
            img_clipped = torch.clamp(img, min=p_min, max=p_max)
            return (img_clipped - p_min) / (p_max - p_min + 1e-7)

        mask = mask.to(device=proj.device, dtype=proj.dtype)
        img_mask = proj * mask
        return robust_normalization(img_mask)

    def warp_render(self, warped_moving, target_poses, affine_inverse):
        B, _, _, _, _ = warped_moving.shape
        warped_projs = []
        for b in range(B):
            warped_proj = self.renderer(
                warped_moving[b, 0],
                target_poses[b],
                affine_inverse[b],
            ).view(1, -1, self.detector.height, self.detector.width)
            warped_projs.append(warped_proj)
        return torch.cat(warped_projs, dim=0)

    def forward(self, input):
        moving, target_proj, target_volume, moving_cp, density, target_poses, affine_inverse = self.load_input_params(input)
        B, _, W, H, D = moving.shape
        coefs, disp_field = self._estimate_flow(moving, target_volume)
        self._log_memory("model/after_estimate_flow")

        if self.use_polyrigid:
            seg_weights = input['weights'].to(density.device)
            affine_mat = input['affine'].to(density.device)
            disp_field = self._polyrigid_to_displacement(
                coefs, seg_weights, affine_mat, (W, H, D))
            self._log_memory("model/after_polyrigid_disp")

        disp_norm = disp_field.clone()
        disp_norm[:, 0] = disp_norm[:, 0] * (2.0 / (W - 1))
        disp_norm[:, 1] = disp_norm[:, 1] * (2.0 / (H - 1))
        disp_norm[:, 2] = disp_norm[:, 2] * (2.0 / (D - 1))
        deform_field = disp_norm + self.id_transform
        self._log_memory("model/after_deform_field")

        warped_moving = self.bilinear(density, deform_field)
        warped_proj = self.warp_render(warped_moving, target_poses, affine_inverse)
        self._log_memory("model/after_warp_render")

        model_output = {
            "params": disp_field,
            "pca_coefs": coefs,
            "target_proj": target_proj,
            "warped_proj": warped_proj,
        }
        if 'warped_density_gt' in input:
            model_output['warped_density_gt'] = input['warped_density_gt']
        if 'displacement_gt' in input:
            model_output['displacement_gt'] = input['displacement_gt']
        return model_output
        

class model_tips3d(model):
    def __init__(self, img_sz, opt=None):
        super(model_tips3d, self).__init__(img_sz, opt)
        self.reorient = torch.tensor(
            [
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ],
            dtype=torch.float32,
        )
    
    def load_input_params(self, input):
        self.sdd = input["sdd"]
        self.infer_detector_width = input["infer_detector_width"]
        self.proj_spacing = input["proj_spacing"]
        self.foreground_mask = input["foreground_mask"]
        moving = input['source']
        target_proj = input["target_proj"]
        target_volume = input["target_volume"]
        if 'source_label' in input:
            moving_seg = input['source_label']
            moving_cp = (moving + 1) * moving_seg - 1
        else:
            moving_cp = moving
        density = input['density']
        target_poses = input['target_poses'].to(
            device=density.device, dtype=torch.float32)
        affine_inverse = input['affine'].to(
            device=density.device, dtype=torch.float32).inverse()

        return moving, target_proj, target_volume, moving_cp, density, target_poses, affine_inverse

    def warp_render(self, warped_moving, target_poses, affine_inverse):
        B, _, _, _, _ = warped_moving.shape
        device = warped_moving.device
        warped_projs = []
        for b in range(B):
            width = self._batch_scalar(self.infer_detector_width, b)
            delx = self._batch_scalar(self.proj_spacing, b, flat_index=0)
            dely = self._batch_scalar(self.proj_spacing, b, flat_index=1)
            self.detector = self._build_detector(
                self.sdd[b], width, delx, dely, device)
            warped_proj = self.renderer(
                warped_moving[b, 0],
                target_poses[b],
                affine_inverse[b],
            ).view(1, -1, self.detector.height, self.detector.width)
            warped_proj = self._preprocess_proj(warped_proj, self.foreground_mask[b])
            warped_projs.append(warped_proj)
        return torch.cat(warped_projs, dim=0).flip(dims=[2])
