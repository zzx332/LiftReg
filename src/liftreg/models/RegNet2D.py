import torch
import torch.nn as nn
import torch.nn.functional as F

class SpatialTransformer(nn.Module):
    """
    2D Spatial Transformer Network
    """
    def __init__(self, size):
        super().__init__()
        # Create sampling grid
        vectors = [torch.arange(0, s) for s in size]
        grids = torch.meshgrid(vectors, indexing='ij')
        grid = torch.stack(grids)
        grid = torch.unsqueeze(grid, 0)
        grid = grid.type(torch.FloatTensor)
        self.register_buffer('grid', grid)

    def forward(self, src, flow, mode="bilinear"):
        # new locations
        new_locs = self.grid + flow
        shape = flow.shape[2:]

        # Normalize grid values to [-1, 1] for grid_sample
        for i in range(len(shape)):
            new_locs[:, i, ...] = 2 * (new_locs[:, i, ...] / (shape[i] - 1) - 0.5)

        # Move channels dim to last position
        if len(shape) == 2:
            new_locs = new_locs.permute(0, 2, 3, 1)
            # new_locs = new_locs[..., [1, 0]]  # x, y format for grid_sample
            new_locs = torch.stack((new_locs[..., 1], new_locs[..., 0]), dim=-1)

        return F.grid_sample(src, new_locs, align_corners=True, mode=mode)


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        # self.gn = nn.GroupNorm(8, out_channels)
        self.bn = nn.BatchNorm2d(out_channels)
        # self.in_norm = nn.InstanceNorm2d(out_channels)
        self.lrelu = nn.LeakyReLU(0.2)
        
    def forward(self, x):
        # return self.lrelu(self.gn(self.conv(x)))
        return self.lrelu(self.bn(self.conv(x)))


class RegNet2D(nn.Module):
    """
    2D Registration Network based on UNet architecture.
    
    Args:
        img_size: Spatial size of the input images (H, W).
        enc_channels: Tuple of encoder channel sizes.
        dec_channels: Tuple of decoder channel sizes.
        use_polyrigid: If True, predict per-segment 2D translations and compose
            them into a dense displacement field via segmentation blending weights,
            instead of directly predicting a per-pixel flow field.
        num_segments: Number of rigid segments for polyrigid mode (ignored when
            use_polyrigid is False).
    """
    def __init__(self, img_size=(160, 160), enc_channels=(16, 32, 64, 128),
                 dec_channels=(64, 32, 16, 16), use_polyrigid=False,
                 num_segments=4):
        super(RegNet2D, self).__init__()
        
        self.img_size = img_size
        self.use_polyrigid = use_polyrigid
        self.num_segments = num_segments

        # Encoder
        self.enc = nn.ModuleList()
        in_ch = 2  # source_proj and target_proj (1 channel each)
        for out_ch in enc_channels:
            self.enc.append(ConvBlock(in_ch, out_ch))
            in_ch = out_ch
            
        # Decoder
        self.dec = nn.ModuleList()
        # Decoder receives concatenated skip connections
        in_ch = enc_channels[-1] + enc_channels[-2]
        for idx, out_ch in enumerate(dec_channels):
            self.dec.append(ConvBlock(in_ch, out_ch))
            if idx < len(dec_channels) - 1:
                in_ch = out_ch + enc_channels[-(idx+3)] if idx < len(enc_channels) - 2 else out_ch

        # ===== Output head =====
        if self.use_polyrigid:
            # Polyrigid head: global pool → FC → per-segment 2D translations
            self.global_pool = nn.AdaptiveAvgPool2d(1)
            self.fc_polyrigid = nn.Sequential(
                nn.Flatten(),
                nn.Linear(dec_channels[-1], 128),
                nn.ReLU(inplace=True),
                nn.Linear(128, self.num_segments * 2),  # 2D translation per segment
            )
            # Small-weight init so initial displacement is near zero
            self.fc_polyrigid[-1].weight.data.normal_(mean=0.0, std=0.1)
            self.fc_polyrigid[-1].bias.data.zero_()
        else:
            # Dense flow head
            self.flow_conv1 = nn.Conv2d(dec_channels[-1], 2, kernel_size=3, padding=1)
            # Initialize small weights for flow
            self.flow_conv1.weight.data.normal_(0, 1e-5)
            self.flow_conv1.bias.data.zero_()

        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.transformer = SpatialTransformer(img_size)

    def _estimate_flow(self, x):
        """Run encoder-decoder and return decoded feature map."""
        # Encoder
        skips = [x]
        for i, enc_block in enumerate(self.enc):
            x = enc_block(x)
            skips.append(x)
            if i < len(self.enc) - 1:
                x = F.max_pool2d(x, 2)

        # Decoder
        for i, dec_block in enumerate(self.dec):
            if i < len(self.dec) - 1:
                x = self.up(x)
            # Concatenate with skip connection
            skip_idx = -(i + 2)
            if abs(skip_idx) < len(skips):
                skip = skips[skip_idx]
                x = torch.cat([x, skip], dim=1)
            x = dec_block(x)

        return x

    @staticmethod
    def _polyrigid_to_displacement(polyrigid_params, weights):
        """
        Pure-translation polyrigid: weighted sum of per-segment 2D translation vectors.

        Each pixel's displacement = sum_k( w_k * t_k ), where t_k is the predicted
        translation for segment k and w_k is its segmentation weight at that pixel.

        Args:
            polyrigid_params: (B, K, 2) predicted 2D translation per segment (pixel units)
            weights:          (B, K, H, W) segmentation blending weights

        Returns:
            disp_field: (B, 2, H, W) dense pixel-unit displacement field
        """
        # weights: (B, K, H, W)   polyrigid_params: (B, K, 2)
        # result:  (B, 2, H, W)
        disp_field = torch.einsum("bkhw,bkn->bnhw", weights, polyrigid_params)
        return disp_field

    def forward(self, input_dict):
        # Expecting inputs as [B, P, H, W] where P=1
        source = input_dict['source_proj']
        target = input_dict['target_proj']
        
        # Concat inputs
        x = torch.cat([source, target], dim=1)

        # Encoder-Decoder
        features = self._estimate_flow(x)

        # Flow prediction
        if self.use_polyrigid:
            B = source.shape[0]
            # Predict per-segment 2D translations
            pooled = self.global_pool(features)
            raw = self.fc_polyrigid(pooled)                   # (B, K*2)
            polyrigid_params = raw.view(B, self.num_segments, 2)  # (B, K, 2)

            # Build dense displacement from segmentation weights
            # Expects input_dict['weights'] of shape (B, K, H, W)
            seg_weights = input_dict['weights'].to(source.device)
            flow = self._polyrigid_to_displacement(polyrigid_params, seg_weights)
        else:
            polyrigid_params = None
            flow = self.flow_conv1(features)  # [B, 2, H, W]

        # Warp source
        y_source = self.transformer(source, flow)

        output = {
            'phi': flow,             # Predicted 2D deformation field
            'warped_moving': y_source,  # Warped DRR
            'target_proj': target,      # Ground truth X-ray
        }

        if self.use_polyrigid:
            output['polyrigid_params'] = polyrigid_params  # (B, K, 2)
        
        # Warp segmentation if provided
        if 'source_label' in input_dict:
            y_source_label = self.transformer(input_dict['source_label'].float(), flow, mode='nearest')
            output['warped_label'] = y_source_label

        return output
