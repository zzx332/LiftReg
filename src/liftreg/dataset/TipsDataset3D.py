from __future__ import division, print_function

import os
import numpy as np
import progressbar as pb
import SimpleITK as sitk
import torch
import torchio
import torch.nn.functional as F

from pathlib import Path
from torch.utils.data import Dataset
from torchio import ScalarImage, LabelMap, Subject
from diffdrr.detector import Detector
from diffdrr.renderers import Siddon
from diffdrr.pose import RigidTransform, convert
from ..utils.sdct_projection_utils import backproj_grids_with_SOUV
from polypose.weights import compute_weights
from scipy.spatial.transform import Rotation

class Warp(torch.nn.Module):
    """Base class for all 3D deformation fields."""

    def __init__(self, density, mask, weights, poses_rot, poses_xyz, affine):
        super().__init__()

        # Load the (possibly downsampled) volume and segmentation mask
        self.density = torch.from_numpy(density).to(dtype=torch.float32).permute(2, 1, 0)[None, None]
        self.mask = torch.from_numpy(mask).to(dtype=torch.uint8).permute(2, 1, 0)[None, None]
        *_, self.W, self.H, self.D = self.density.shape

        # Initialize identity points for sampling the displacement field
        X, Y, Z = torch.meshgrid(
            torch.arange(self.D),
            torch.arange(self.H),
            torch.arange(self.W),
            indexing="ij",
        )
        self.pts = torch.stack([X, Y, Z], dim=-1).to(torch.float32)
        
        self.shape = torch.tensor([self.D, self.H, self.W])
        self.weights = torch.from_numpy(weights).to(dtype=torch.float32)
        self.weights = F.interpolate(
            self.weights[None],
            (self.D, self.H, self.W),
            mode="trilinear",
            align_corners=False,
        )[0]
        self.K, *_ = self.weights.shape
        # Initialize the log transforms for the articulated structures in the volume
        self.poses_rot = torch.nn.Parameter(poses_rot)
        self.poses_xyz = torch.nn.Parameter(poses_xyz)
        self.affine = RigidTransform(affine)
        self.affine_inverse = RigidTransform(affine.inverse())

    def normalize(self, x):
        return 2 * x / self.shape - 1
    
    @property
    def pose(self) -> RigidTransform:
        """Compute the average log transform at every point in space and map to the manifold."""
        poses = torch.concat([self.poses_rot, self.poses_xyz], dim=-1)
        logs = torch.einsum("cdhw,cn->dhwn", self.weights, poses).reshape(-1, 6)
        pose = convert(*logs.split([3, 3], dim=1), parameterization="se3_log_map")
        return self.affine.compose(pose).compose(self.affine_inverse)

    def warp(self):
        """Sample the displacement field at the identity points."""
        x = self.pts.reshape(-1, 1, 3)
        x = self.pose(x)
        return x.reshape(1, self.D, self.H, self.W, 3)

    def forward(self):
        warped_coords = self.warp()
        original_coords = self.pts
        displacement = warped_coords[0] - original_coords
        pts = self.normalize(warped_coords)
        warped_density = self._warp_volume(self.density, pts)
        warped_mask = self._warp_mask(self.mask, pts)
        return warped_density, warped_mask, displacement


    def _warp_volume(self, volume, pts):
        dtype = volume.dtype
        if volume.dtype != pts.dtype:
            volume = volume.to(pts.dtype)
        return F.grid_sample(volume, pts, align_corners=False, mode="bilinear", padding_mode="border").squeeze().to(dtype)

    def _warp_mask(self, mask, pts):
        dtype = mask.dtype
        if mask.dtype != pts.dtype:
            mask = mask.to(pts.dtype)
        return F.grid_sample(mask, pts, align_corners=False, mode="nearest").squeeze().to(dtype)

def parse_mhd_header(mhd_path):
    """读取 mhd 文本头，解析 key=value 字段。"""
    meta = {}
    with mhd_path.open("r", encoding="utf-8", errors="ignore") as f:
        for raw in f:
            line = raw.strip()
            if not line or "=" not in line:
                continue
            key, value = line.split("=", 1)
            meta[key.strip()] = value.strip()
    return meta

class TipsDataset3D(Dataset):
    """
    2D DRR + 3D CT registration dataset.

    Data pipeline (nnUNet-style):
      Stage 1 – Preprocessing: runs once per identifier, results saved to
                {data_path}/preprocessed/ as .npz + _meta.pt files.
      Stage 2 – Online augmentation in __getitem__ (train phase only, CPU numpy).
      Stage 3 – Collation & GPU transfer handled by DataLoader / Trainer.
    """

    def __init__(self, data_path, phase=None, transform=None, option=None):
        self.data_path = data_path
        self.org_data_path = "/home/zzx/data/tips_dataset3d"
        self.cache_dir = os.path.join(self.data_path, "preprocessed")
        self.warp_path = os.path.join(self.data_path, "warp_pose")
        self.warped_data_path = os.path.join(self.data_path, "warped_data")
        self.cache_version = 2
        
        os.makedirs(self.cache_dir, exist_ok=True)

        self.phase     = phase
        self.transform = transform

        ind = ['train', 'val', 'test', 'debug'].index(phase)
        max_num = option[('max_num_for_loading', (-1, -1, -1, -1),
                          'max pairs to load per split')]
        self.max_num_for_loading = max_num[ind]

        self.has_label = option[('use_segmentation_map', False,
                                 'load segmentation maps')]
        self.spacing   = option[('spacing_to_refer', (1, 1, 1), '')]
        self.img_after_resize = option[('img_after_resize', (160, 160, 160), '')]
        self.img_after_resize_2d = option[('img_after_resize_2d', (512, 512), '')]
        self.load_projection_interval = option[('load_projection_interval', 1, '')]
        self.apply_hu_clip = option[('apply_hu_clip', False, '')]

        # ---- Augmentation config (train phase only) ----
        self.enable_aug = (
            option[('augmentation_enable', True, 'enable data augmentation')]
            and (phase == 'train')
        )
        self.offline_aug_variants = int(
            option[('offline_aug_variants', 4, 'number of cached affine augmentation variants')]
        )
        if not self.enable_aug:
            self.offline_aug_variants = 0
        self.aug_gamma_prob      = option[('aug_gamma_prob',      0.4, '')]
        self.aug_brightness_prob = option[('aug_brightness_prob', 0.5, '')]
        self.aug_noise_prob      = option[('aug_noise_prob',      0.5, '')]
        self.aug_lr_flip_prob    = option[('aug_lr_flip_prob',    0.5, '')]
        self.aug_affine_prob     = option[('aug_affine_prob',     0.3, '')]
        self.aug_rotate_deg      = option[('aug_rotate_deg',      5.0, '')]
        self.aug_translate_vox   = option[('aug_translate_vox',   4.0, '')]
        self.aug_scale_range     = option[('aug_scale_range',     (0.95, 1.05), '')]
        # self.aug_proj_mask_prob  = option[('aug_proj_mask_prob',  0.3, '')]
        self.detector_spacing = option[('detector_spacing', 0.7255, '')]
        self.detector_width = option[('detector_width', 512, '')]

        self.reorient = torch.tensor(
            [[1, 0, 0, 0],
             [0, 1, 0, 0],
             [0, 0, 1, 0],
             [0, 0, 0, 1]], dtype=torch.float32)

        # meta_list holds lightweight per-sample dicts (poses, affine)
        self.meta_cache = {}
        self.get_identifier_list()
        self.init_img_pool()

    # ------------------------------------------------------------------
    # Identifier list
    # ------------------------------------------------------------------
    def get_identifier_list(self):
        ct_root = Path(self.org_data_path)
        matches = sorted(ct_root.rglob("*DSA.mhd"))
        self.identifier_list = [
            i.stem for i in matches
        ]
        if self.max_num_for_loading > 0:
            self.identifier_list = self.identifier_list[:self.max_num_for_loading]

    # ------------------------------------------------------------------
    # Projector (lazy init, CPU only)
    # ------------------------------------------------------------------
    def _init_projector_if_needed(self):
        if not hasattr(self, '_siddon') or self._siddon is None:
            self._siddon = Siddon(voxel_shift=0.5)
        if not hasattr(self, '_detector') or self._detector is None:
            self._detector = Detector(
                # 1020.0, 718, 718, 0.388, 0.388, 0, 0,
                self.sdd, self.detector_width, self.detector_width, self.detector_spacing, self.detector_spacing, 0, 0,
                self.reorient, reverse_x_axis=True, n_subsample=None,
            )

    def _run_renderer(self, density, target_poses, affine_inverse):
        """Forward DRR render, returns (1, P, H, W) tensor on CPU."""
        poses = RigidTransform(target_poses)
        src, tgt = self._detector(poses, calibration=None)
        img_input = (tgt - src).norm(dim=-1).unsqueeze(1)
        src = RigidTransform(affine_inverse)(src)
        tgt = RigidTransform(affine_inverse)(tgt)
        return self._siddon(density, src, tgt, img_input, align_corners=False).view(
            1, -1, self._detector.height, self._detector.width)

    def _preprocess_xray(self, xray_path, crop, linearize):
        img = sitk.GetArrayFromImage(sitk.ReadImage(xray_path))
        mask = img > 0
        from scipy.ndimage import binary_erosion
        k = 5  # 想收几像素就设几
        mask = binary_erosion(mask, iterations=k)
        """Configurable X-ray preprocessing"""
        # Remove edge artifacts caused by the collimator
        def center_crop_np(img, out_hw):
            out_h, out_w = out_hw
            *lead, h, w = img.shape
            top = max((h - out_h) // 2, 0)
            left = max((w - out_w) // 2, 0)
            return img[..., top:top + out_h, left:left + out_w]
        def robust_normalization(img, lower_percentile=1, upper_percentile=99):
            # 1. 计算百分位数，剔除上下各 1% 的极端值
            p_min, p_max = np.percentile(img, (lower_percentile, upper_percentile))
            
            # 2. 进行截断处理（Clipping）
            img_clipped = np.clip(img, p_min, p_max)
            
            # 3. 重新归一化到 [0, 1]
            img_norm = (img_clipped - p_min) / (p_max - p_min + 1e-7)
            
            return img_norm
        if crop != 0:
            *_, height, width = img.shape
            img = center_crop_np(img, (height - crop, width - crop))

        # 
        if linearize:
            # img += 1
            if mask is not None:
                # 仅对掩码区域变换
                img_log = np.zeros_like(img, dtype=np.float32)
                img_log[mask] = np.log(img[mask].max()) - np.log(img[mask])
                img = img_log
            else:
                img += 1
                img = np.log(img.max()) - np.log(img)
        else:
            img_mask = np.zeros_like(img, dtype=np.float32)
            img_mask[mask] = img[mask]
            img = img_mask
        # Rescale to [0, 1]
        img = robust_normalization(img)
        return img, mask

    def _load_dataset(self, identifier):
        # datapath = Path(rf"/home/zzx/data/deepfluoro/subject{subject_id:02d}")
        datapath = Path(self.org_data_path) / identifier.split('-')[0]
        # Make paths to the relevant images
        volume = datapath / f"{identifier.split('-')[-2]}_volume.nii.gz"
        mask = datapath / "liver.nii.gz"
        xrays = datapath / f"{identifier}.mhd"
        rigid_case_name = identifier.split('-')[0] + "_DSA_" + identifier.split('-')[1] + "-CT_" + identifier.split('-')[0] \
            + "_CT_" + identifier.split('-')[-2]
        meta = parse_mhd_header(xrays)
        if os.path.exists(datapath / f"{rigid_case_name}_meta.pt"):
            param_dict = torch.load(datapath / f"{rigid_case_name}_meta.pt")
        else:
            param_dict = torch.load(datapath / f"{identifier.split('-')[0]}_meta.pt")
            volume = datapath / "data_volume.nii.gz"
        rigid_param = param_dict["rigid_param"]
        extrinsic_cam2world = torch.as_tensor(param_dict["extrinsic_cam2world"], dtype=torch.float32)
        target_proj, foreground_mask = self._preprocess_xray(xrays, crop=0, linearize=True)
        if target_proj.ndim == 2:
            target_proj = target_proj[None, ...] 
            foreground_mask = foreground_mask[None, ...] 
        # target_proj = np.transpose(target_proj, (0, 2, 1))  # 把 proj变成hw顺序            
        target_proj_np = target_proj.astype(np.float32)
        # return ScalarImage(volume), LabelMap(mask), xrays, LabelMap(segs)
        return volume, mask, target_proj_np, foreground_mask, rigid_param, extrinsic_cam2world, meta

    def _cache_is_valid(self, npz_path, meta_path):
        if not (os.path.exists(npz_path) and os.path.exists(meta_path)):
            return False

        try:
            with np.load(npz_path) as npz:
                required_keys = {
                    'cache_version', 'source', 'density', 'target_proj', 'target_volume', 'spacing', 'proj_spacing',
                    'infer_detector_width', 'sdd',
                    'foreground_mask'
                }
                if self.has_label:
                    required_keys.update({'source_seg', 'weights'})
                if self.offline_aug_variants > 0:
                    required_keys.update({
                        'aug_source', 'aug_density', 'aug_affines'
                    })
                    if self.has_label:
                        required_keys.update({'aug_source_seg', 'aug_weights'})

                if not required_keys.issubset(set(npz.files)):
                    return False

                cache_version = int(np.asarray(npz['cache_version']).reshape(-1)[0])
                if cache_version != self.cache_version:
                    return False

                if self.offline_aug_variants > 0:
                    if npz['aug_source'].shape[0] != self.offline_aug_variants:
                        return False
        except Exception:
            return False

        return True

    def _build_offline_augmented_bank_3d(
        self, identifier, source, density, source_seg, weights, affine, foreground_mask
    ):
        if self.offline_aug_variants <= 0 or source_seg is None or weights is None:
            return None

        aug_source = []
        aug_density = []
        aug_source_seg = []
        aug_weights = []
        aug_foreground_mask = []
        aug_affines = []

        for _ in range(self.offline_aug_variants):
            (source_aug, density_aug, source_seg_aug, weights_aug,
             affine_aug, affine_grid) = self._apply_shared_affine_3d(
                source, density, source_seg, weights, affine.clone(), force=True)

            aug_source.append(np.ascontiguousarray(source_aug.astype(np.float32)))
            aug_density.append(np.ascontiguousarray(density_aug.astype(np.float32)))
            aug_source_seg.append(np.ascontiguousarray(source_seg_aug.astype(np.float32)))
            aug_weights.append(np.ascontiguousarray(weights_aug.astype(np.float32)))
            aug_affines.append(affine_aug.numpy().astype(np.float32))
            aug_foreground_mask.append(np.ascontiguousarray(foreground_mask.astype(np.float32)))
        if len(aug_source) == 0:
            return None

        save_dict = {
            'aug_source': np.stack(aug_source, axis=0).astype(np.float32),
            'aug_density': np.stack(aug_density, axis=0).astype(np.float32),
            'aug_source_seg': np.stack(aug_source_seg, axis=0).astype(np.float32),
            'aug_weights': np.stack(aug_weights, axis=0).astype(np.float32),
            'aug_affines': np.stack(aug_affines, axis=0).astype(np.float32),
            'aug_foreground_mask': np.stack(aug_foreground_mask, axis=0).astype(np.float32),
        }

        return save_dict
    # ------------------------------------------------------------------
    # Stage 1 – Preprocessing & caching  (runs once per identifier)
    # ------------------------------------------------------------------
    def _preprocess_and_cache(self, identifier):
        """Preprocess one CT/pose pair and save results to disk."""
        npz_path  = os.path.join(self.cache_dir, f'{identifier}.npz')
        meta_path = os.path.join(self.cache_dir, f'{identifier}_meta.pt')
        if self._cache_is_valid(npz_path, meta_path):
            return  # already cached
   
        # ---- 1. Load & preprocess CT volume ----
        # source_img, source_seg, xrays, _ = self._load_dataset(identifier.split('_')[0])
        volume, mask, target_proj_np, foreground_mask, rigid_param, extrinsic_cam2world, meta = self._load_dataset(identifier)
        self.sdd = float(meta.get("Extra_SourceToDetectorDistance", 1020.0))
        spacing_vals = [float(x) for x in meta.get("ElementSpacing", "0.7255 0.7255").split()]
        self.detector_width = int(meta.get("DimSize", "512 512").split()[0])
        self.detector_spacing = spacing_vals[0]
        self._init_projector_if_needed()     
        # source_img = ScalarImage(
        #     os.path.join(self.data_path,
        #                  identifier.split('_')[0] + '_source.nii.gz'))
        source_img = ScalarImage(volume) # X, Y, Z
        # source_seg = ScalarImage(mask)
        source_seg = LabelMap(mask)
        source_img = self.canonicalize(source_img, rigid_param)
        source_seg = self.canonicalize(source_seg, rigid_param, is_label=True)
        # subject = self.canonicalize(subject)
        subject = Subject(volume=source_img, mask=source_seg)
        _, weights = compute_weights(subject, labels=[[1]])
        new_spacing = [source_img.shape[i+1] * source_img.spacing[i] / self.img_after_resize[i] for i in range(3)]
        tfm = torchio.transforms.Compose([
            torchio.transforms.Resample(new_spacing),
            # torchio.transforms.CropOrPad(self.img_after_resize, padding_mode=-2048),
        ])
        source_img = tfm(source_img)
        if self.has_label and source_seg is not None:
            source_seg = tfm(source_seg)

        source_arr = source_img.data.squeeze(0).numpy().astype(np.float32)
        if self.apply_hu_clip:
            source_arr = self._normalize_intensity(
                source_arr, linear_clip=True, clip_range=[-1000, 1000])
        else:
            source_arr = self._normalize_intensity(source_arr, linear_clip=True)

        density_t   = self.transform_hu_to_density(
            source_img.data, bone_attenuation_multiplier=2.0)
        density_arr = density_t.squeeze(0).numpy().astype(np.float32)

        affine = torch.as_tensor(source_img.affine, dtype=torch.float32)

        # debug
        data_path = "/home/zzx/code"
        sour_proj = self._run_renderer(density_t[0], extrinsic_cam2world, affine.inverse())
        img = sitk.GetImageFromArray(sour_proj[0].flip(dims=[1]))
        img.SetSpacing((self.detector_spacing, self.detector_spacing, 1))
        sitk.WriteImage(img, os.path.join(data_path, "sour_proj.nii.gz"))
        # ---- 3. Back-project 2D → 3D volume ----
        target_poses_SOUV = self._extrinsic_cam2world_to_SOUV(
            extrinsic_cam2world, self.reorient[:3, :3], \
                sdd=self.sdd)
        # proj_for_bp = (torch.from_numpy(target_proj_np)
        #                .permute(0, 2, 1).flip(dims=[2]))
        proj_for_bp = torch.from_numpy(target_proj_np)
        target_volume = self.backproject_volume(
            target_poses_SOUV, proj_for_bp, source_arr.shape,
            new_spacing, device=torch.device('cpu')).permute(2, 1, 0) # (X, Y, Z)
        target_volume_np = target_volume.detach().cpu().numpy().astype(np.float32) # (X, Y, Z)
        target_proj_np = self._resize_projection_stack(target_proj_np[0], mode="bilinear")
        foreground_mask = self._resize_projection_stack(foreground_mask[0], mode="nearest")
        proj_spacing = [self.detector_spacing * self.detector_width / self.img_after_resize_2d[i] for i in range(2)]
        # ---- 4. Save to disk ----
        save_dict = dict(
            cache_version=np.array([self.cache_version], dtype=np.int16),
            source=source_arr, # (X, Y, Z)
            density=density_arr, # (X, Y, Z)
            target_proj=target_proj_np, # (P, H, W)
            target_volume=target_volume_np,# (X, Y, Z)
            foreground_mask=foreground_mask, # (1, H, W)
            spacing=np.array(new_spacing, dtype=np.float32),
            proj_spacing=np.array(proj_spacing, dtype=np.float32),
            infer_detector_width=self.img_after_resize_2d[0],
            sdd=self.sdd,
        )
        if self.has_label and source_seg is not None:
            save_dict['source_seg'] = (
                source_seg.data.squeeze(0).numpy().astype(np.float32))
            save_dict['weights'] = weights.detach().cpu().numpy().astype(np.float32)

        aug_dict = self._build_offline_augmented_bank_3d(
            identifier,
            source_arr,
            density_arr,
            save_dict.get('source_seg', None),
            save_dict.get('weights', None),
            affine,
            foreground_mask,
        )
        if aug_dict is not None:
            save_dict.update(aug_dict)
        np.savez_compressed(npz_path, **save_dict)

        torch.save(dict(
            affine=affine,
            target_poses=extrinsic_cam2world,
            target_poses_SOUV=target_poses_SOUV,
        ), meta_path)

    # ------------------------------------------------------------------
    # Stage 1 driver: preprocess missing samples, load meta into RAM
    # ------------------------------------------------------------------
    def init_img_pool(self):
        print(f'[{self.phase}] Checking / building cache in {self.cache_dir}')
        pbar = pb.ProgressBar(
            widgets=[pb.Percentage(), pb.Bar(), pb.ETA()],
            maxval=len(self.identifier_list)).start()

        for i, identifier in enumerate(self.identifier_list):
            self._preprocess_and_cache(identifier)
            meta = torch.load(
                os.path.join(self.cache_dir, f'{identifier}_meta.pt'),
                weights_only=False)
            self.meta_cache[identifier] = {"meta": meta}
            pbar.update(i + 1)

        pbar.finish()
        print(f'[{self.phase}] {len(self.identifier_list)} samples ready.')

    # ------------------------------------------------------------------
    # Stage 2 – Online augmentation helpers
    # ------------------------------------------------------------------
    def _augment_3d_intensity(self, arr):
        """
        In-place-safe intensity augmentation for a (D, H, W) float32 ndarray.
        Values expected in [-1, 1]; output clamped to [-1, 1].
        """
        arr = arr.copy()

        if np.random.rand() < self.aug_brightness_prob:
            scale = float(np.random.uniform(0.85, 1.15))
            bias  = float(np.random.uniform(-0.10, 0.10))
            arr   = arr * scale + bias

        if np.random.rand() < self.aug_gamma_prob:
            gamma = float(np.random.uniform(0.75, 1.30))
            arr01 = np.clip((arr + 1.0) * 0.5, 0.0, 1.0)
            arr   = arr01 ** gamma * 2.0 - 1.0

        if np.random.rand() < self.aug_noise_prob:
            sigma = float(np.random.uniform(0.0, 0.04))
            arr   = arr + np.random.randn(*arr.shape).astype(np.float32) * sigma

        return np.clip(arr, -1.0, 1.0).astype(np.float32)

    def _augment_2d_projs_shared(self, *projs):
        """
        X-ray physics-aware augmentation for one or more (P, H, W) float32 stacks.

        One random draw per batch (bernoulli + branch + hyperparameters) is
        shared across all inputs so source_proj and target_proj see the same
        corruption type and strength; each image keeps its own min/max rescale.

        Three physics-motivated transforms (mutually exclusive, one chosen per call):
          1. Poisson noise  – models shot noise (same lam for all stacks).
          2. Gaussian blur  – same sigma for all stacks.
          3. Log-transform  – same alpha for all stacks.

        Applied with probability aug_noise_prob.
        """
        if not projs:
            return tuple(), False

        aug_copies = tuple(p.astype(np.float32).copy() for p in projs)
        aug_tag = False
        if np.random.rand() >= self.aug_noise_prob:
            return aug_copies + (aug_tag,)

        choice = np.random.randint(0, 3)
        metas = []
        p01_list = []
        for p in aug_copies:
            p_min = float(p.min())
            p_max = float(p.max())
            p_range = max(p_max - p_min, 1e-6)
            p01 = np.clip((p - p_min) / p_range, 0.0, 1.0).astype(np.float32)
            metas.append((p_min, p_max, p_range))
            p01_list.append(p01)

        if choice == 0:
            # lam = float(np.random.uniform(300.0, 1000.0))
            lam = float(np.random.uniform(2000.0, 5000.0))
            p01_list = [
                np.clip(np.random.poisson(p01 * lam).astype(np.float32) / lam, 0.0, 1.0)
                for p01 in p01_list
            ]
        elif choice == 1:
            from scipy.ndimage import gaussian_filter

            sigma_blur = float(np.random.uniform(0.5, 2.0))
            new_list = []
            for p01 in p01_list:
                out = np.empty_like(p01)
                for i in range(p01.shape[0]):
                    out[i] = gaussian_filter(p01[i], sigma=sigma_blur).astype(np.float32)
                new_list.append(out)
            p01_list = new_list
        else:
            alpha = float(np.random.uniform(2.0, 10.0))
            log_denom = float(np.log1p(alpha))
            p01_list = [
                (np.log1p(alpha * p01) / log_denom).astype(np.float32) for p01 in p01_list
            ]

        out = []
        for p01, (p_min, p_max, p_range) in zip(p01_list, metas):
            out.append(np.clip(p01 * p_range + p_min, p_min, p_max).astype(np.float32))
        aug_tag = True
        return tuple(out) + (aug_tag,)

    def _resolve_axis_ranges(self, value, symmetric=False):
        arr = np.asarray(value, dtype=np.float32)
        if arr.ndim == 0:
            bound = float(arr)
            pair = (-bound, bound) if symmetric else (bound, bound)
            return [pair, pair, pair]
        if arr.size == 2:
            pair = (float(arr[0]), float(arr[1]))
            return [pair, pair, pair]
        if arr.size == 3:
            return [
                (-float(v), float(v)) if symmetric else (float(v), float(v))
                for v in arr.tolist()
            ]
        if arr.size == 6:
            return [
                (float(arr[0]), float(arr[1])),
                (float(arr[2]), float(arr[3])),
                (float(arr[4]), float(arr[5])),
            ]
        raise ValueError(f'Unsupported axis range config: {value}')

    def _resolve_scale_range(self, value):
        arr = np.asarray(value, dtype=np.float32)
        if arr.ndim == 0:
            delta = float(arr)
            return 1.0 - delta, 1.0 + delta
        if arr.size == 2:
            return float(arr[0]), float(arr[1])
        raise ValueError(f'Unsupported scale range config: {value}')

    def _normalized_to_voxel_h(self, vol_shape):
        D, H, W = [int(x) for x in vol_shape]
        mat = torch.eye(4, dtype=torch.float32)
        if W > 1:
            mat[0, 0] = (W - 1) / 2.0
            mat[0, 3] = (W - 1) / 2.0
        else:
            mat[0, 0] = 0.0
            mat[0, 3] = 0.0
        if H > 1:
            mat[1, 1] = (H - 1) / 2.0
            mat[1, 3] = (H - 1) / 2.0
        else:
            mat[1, 1] = 0.0
            mat[1, 3] = 0.0
        if D > 1:
            mat[2, 2] = (D - 1) / 2.0
            mat[2, 3] = (D - 1) / 2.0
        else:
            mat[2, 2] = 0.0
            mat[2, 3] = 0.0
        return mat

    def _voxel_to_normalized_h(self, vol_shape):
        D, H, W = [int(x) for x in vol_shape]
        mat = torch.eye(4, dtype=torch.float32)
        if W > 1:
            mat[0, 0] = 2.0 / (W - 1)
            mat[0, 3] = -1.0
        else:
            mat[0, 0] = 0.0
            mat[0, 3] = 0.0
        if H > 1:
            mat[1, 1] = 2.0 / (H - 1)
            mat[1, 3] = -1.0
        else:
            mat[1, 1] = 0.0
            mat[1, 3] = 0.0
        if D > 1:
            mat[2, 2] = 2.0 / (D - 1)
            mat[2, 3] = -1.0
        else:
            mat[2, 2] = 0.0
            mat[2, 3] = 0.0
        return mat

    def _build_shared_affine(self, vol_shape, force=False):
        if (not force) and np.random.rand() >= self.aug_affine_prob:
            return None, None

        rotate_ranges = self._resolve_axis_ranges(self.aug_rotate_deg, symmetric=True)
        translate_ranges = self._resolve_axis_ranges(self.aug_translate_vox, symmetric=True)
        scale_min, scale_max = self._resolve_scale_range(self.aug_scale_range)

        rx, ry, rz = [
            np.deg2rad(np.random.uniform(lo, hi)) for lo, hi in rotate_ranges
        ]
        tx, ty, tz = [
            np.random.uniform(lo, hi) for lo, hi in translate_ranges
        ]
        scale = float(np.random.uniform(scale_min, scale_max))

        cx = (vol_shape[2] - 1) / 2.0
        cy = (vol_shape[1] - 1) / 2.0
        cz = (vol_shape[0] - 1) / 2.0

        translate_to_center = np.eye(4, dtype=np.float32)
        translate_to_center[:3, 3] = np.array([-cx, -cy, -cz], dtype=np.float32)
        translate_back = np.eye(4, dtype=np.float32)
        translate_back[:3, 3] = np.array([cx, cy, cz], dtype=np.float32)
        shift = np.eye(4, dtype=np.float32)
        shift[:3, 3] = np.array([tx, ty, tz], dtype=np.float32)
        scale_mat = np.diag([scale, scale, scale, 1.0]).astype(np.float32)

        cosx, sinx = np.cos(rx), np.sin(rx)
        cosy, siny = np.cos(ry), np.sin(ry)
        cosz, sinz = np.cos(rz), np.sin(rz)

        rot_x = np.array([
            [1.0, 0.0,  0.0,   0.0],
            [0.0, cosx, -sinx, 0.0],
            [0.0, sinx, cosx,  0.0],
            [0.0, 0.0,  0.0,   1.0],
        ], dtype=np.float32)
        rot_y = np.array([
            [cosy,  0.0, siny, 0.0],
            [0.0,   1.0, 0.0,  0.0],
            [-siny, 0.0, cosy, 0.0],
            [0.0,   0.0, 0.0,  1.0],
        ], dtype=np.float32)
        rot_z = np.array([
            [cosz, -sinz, 0.0, 0.0],
            [sinz, cosz,  0.0, 0.0],
            [0.0,  0.0,   1.0, 0.0],
            [0.0,  0.0,   0.0, 1.0],
        ], dtype=np.float32)

        forward = shift @ translate_back @ rot_z @ rot_y @ rot_x @ scale_mat @ translate_to_center
        voxel_to_input = torch.from_numpy(np.linalg.inv(forward).astype(np.float32))
        theta_h = (
            self._voxel_to_normalized_h(vol_shape)
            @ voxel_to_input
            @ self._normalized_to_voxel_h(vol_shape)
        )
        theta = theta_h[:3, :].unsqueeze(0)
        grid = F.affine_grid(
            theta,
            size=(1, 1, int(vol_shape[0]), int(vol_shape[1]), int(vol_shape[2])),
            align_corners=False,
        )
        return grid, voxel_to_input

    def _apply_affine_grid(self, arr, grid, mode='bilinear', padding_mode='border'):
        if arr.ndim == 3:
            tensor = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)
        elif arr.ndim == 4:
            tensor = torch.from_numpy(arr).unsqueeze(0)
        else:
            raise ValueError(f'Expected 3D or 4D array, got shape {arr.shape}')
        sampled = F.grid_sample(
            tensor.to(dtype=torch.float32),
            grid,
            mode=mode,
            padding_mode=padding_mode,
            align_corners=False,
        )
        sampled = sampled.squeeze(0).cpu().numpy().astype(np.float32)
        return sampled[0] if arr.ndim == 3 else sampled

    def _apply_shared_affine_3d(self, source, density, source_seg, weights, affine, force=False):
        grid, voxel_to_input = self._build_shared_affine(source.shape, force=force)
        if grid is None:
            return source, density, source_seg, weights, affine, None

        stacked = np.stack([source, density], axis=0)
        stacked = self._apply_affine_grid(
            stacked, grid, mode='bilinear', padding_mode='border')
        source_aug, density_aug = stacked[0], stacked[1]

        source_seg_aug = source_seg
        if source_seg is not None:
            source_seg_aug = self._apply_affine_grid(
                source_seg, grid, mode='nearest', padding_mode='zeros')

        weights_aug = weights
        if weights is not None:
            weights_aug = self._apply_affine_grid(
                weights, grid, mode='nearest', padding_mode='zeros')

        new_affine = affine @ voxel_to_input
        return source_aug, density_aug, source_seg_aug, weights_aug, new_affine, grid

    def _apply_lr_flip(self, source, density, source_seg, weights,
                       target_proj, target_volume, affine):
        """
        Synchronized left-right (W-axis) flip across all modalities.

        Affine update: the voxel-to-physical affine maps (w, h, d) → (x, y, z).
        Flipping W reverses w → W-1-w, so the new affine is:
            new_affine = old_affine @ F
        where F = diag(-1, 1, 1, 1) with F[0, 3] = W-1 (in homogeneous coords).
        This assumes the affine's 0-th column corresponds to the W (x-physical)
        direction, which holds for torchio's canonicalized isotropic volumes.
        """
        W = source.shape[-1]

        source        = np.ascontiguousarray(source[..., ::-1])
        density       = np.ascontiguousarray(density[..., ::-1])
        target_volume = np.ascontiguousarray(target_volume[..., ::-1])
        weights       = np.ascontiguousarray(weights[..., ::-1])
        # Horizontal flip of projection mirrors the LR-flipped 3D volume
        target_proj   = np.ascontiguousarray(target_proj[..., ::-1])
        if source_seg is not None:
            source_seg = np.ascontiguousarray(source_seg[..., ::-1])

        flip_mat = torch.eye(4, dtype=torch.float32)
        flip_mat[0, 0] = -1.0
        flip_mat[0, 3] = float(W - 1)
        new_affine = affine @ flip_mat

        return source, density, source_seg, weights, target_proj, target_volume, new_affine

    def _resize_projection_stack(self, proj, mode):
        tensor = torch.from_numpy(np.ascontiguousarray(proj.astype(np.float32))).unsqueeze(0).unsqueeze(0)
        kwargs = {"align_corners": False} if mode == "bilinear" else {}
        # kwargs = {"align_corners": True} if mode == "bilinear" else {}
        resized = F.interpolate(
            tensor,
            size=self.img_after_resize_2d,
            mode=mode,
            **kwargs,
        )
        return resized.squeeze(0).numpy().astype(np.float32)

    # ------------------------------------------------------------------
    # Stage 2 + 3 – __getitem__
    # ------------------------------------------------------------------
    def __getitem__(self, idx):
        idx        = idx % len(self.identifier_list)
        identifier = self.identifier_list[idx]
        meta       = self.meta_cache[identifier]["meta"]
        affine     = meta['affine'].clone()
        # ---- Load preprocessed arrays from disk (fast np.load) ----
        npz_path = os.path.join(self.cache_dir, f'{identifier}.npz')
        with np.load(npz_path) as npz:
            source        = np.ascontiguousarray(npz['source'].astype(np.float32))         # (D, H, W)
            density       = np.ascontiguousarray(npz['density'].astype(np.float32))        # (D, H, W)
            target_proj   = np.ascontiguousarray(npz['target_proj'].astype(np.float32))    # (P, H, W)
            target_volume = np.ascontiguousarray(npz['target_volume'].astype(np.float32))  # (D, H, W)
            spacing       = np.ascontiguousarray(npz['spacing'].astype(np.float32))
            proj_spacing  = np.ascontiguousarray(npz['proj_spacing'].astype(np.float32))
            sdd = np.float32(npz['sdd'])
            infer_detector_width = np.ascontiguousarray(npz['infer_detector_width'].astype(np.float32))
            source_seg    = (np.ascontiguousarray(npz['source_seg'].astype(np.float32))
                             if self.has_label else None)
            weights       = (np.ascontiguousarray(npz['weights'].astype(np.float32))
                             if self.has_label else None)
            foreground_mask = np.ascontiguousarray(npz['foreground_mask'].astype(np.float32))

            if self.enable_aug and self.offline_aug_variants > 0 and 'aug_source' in npz:
                if np.random.rand() < self.aug_affine_prob:
                    aug_idx = np.random.randint(npz['aug_source'].shape[0])
                    source = np.ascontiguousarray(npz['aug_source'][aug_idx].astype(np.float32))
                    density = np.ascontiguousarray(npz['aug_density'][aug_idx].astype(np.float32))
                    affine = torch.from_numpy(npz['aug_affines'][aug_idx].astype(np.float32))
                    if self.has_label and 'aug_source_seg' in npz:
                        source_seg = np.ascontiguousarray(npz['aug_source_seg'][aug_idx].astype(np.float32))
                    if self.has_label and 'aug_weights' in npz:
                        weights = np.ascontiguousarray(npz['aug_weights'][aug_idx].astype(np.float32))
        # ---- Online augmentation (train phase only) ----
        if self.enable_aug:
            source_aug = source
            target_volume_aug = target_volume
            source_aug = self._augment_3d_intensity(source_aug)
            density = self._augment_3d_intensity(density)
            target_proj_aug, aug_tag = self._augment_2d_projs_shared(target_proj)
        else:
            source_aug = source
            target_volume_aug = target_volume
            target_proj_aug = target_proj

        # Ensure numpy buffers are contiguous/writable for DataLoader collation.
        source_aug = np.ascontiguousarray(source_aug.astype(np.float32))
        density = np.ascontiguousarray(density.astype(np.float32))
        target_volume_aug = np.ascontiguousarray(target_volume_aug.astype(np.float32))
        target_proj_aug = np.ascontiguousarray(target_proj_aug.astype(np.float32))
        if source_seg is not None:
            source_seg = np.ascontiguousarray(source_seg.astype(np.float32))
        if weights is not None:
            weights = np.ascontiguousarray(weights.astype(np.float32))
        # ---- Build sample dict (add channel dim to 3D volumes) ----
        sample = {
            'source':            np.expand_dims(source_aug,        0),  # (1,D,H,W)
            'density':           np.expand_dims(density,       0),  # (1,D,H,W)
            'target_volume':     np.expand_dims(target_volume_aug, 0),  # (1,D,H,W)
            'target_proj':       target_proj_aug,                        # (P,H,W)
            'foreground_mask':   foreground_mask,     # (1,H,W)
            'spacing':           spacing,
            'affine':            affine,
            'target_poses':      meta['target_poses'],
            'target_poses_SOUV': meta['target_poses_SOUV'],
            'sdd': sdd,
            'infer_detector_width': infer_detector_width,
            'proj_spacing':      proj_spacing,
        }
        if self.has_label and source_seg is not None:
            sample['source_label'] = np.expand_dims(source_seg, 0)  # (1,D,H,W)
        if self.has_label and weights is not None:
            weights_t = torch.from_numpy(weights).unsqueeze(0).float()
            weights_interp = F.interpolate(
                weights_t, size=source.shape, mode='trilinear', align_corners=False
            ).squeeze(0).numpy().astype(np.float32)
            sample['weights'] = weights_interp  # (K,D,H,W)

        if self.transform:
            sample['source'] = self.transform(sample['source'])

        return sample, identifier

    def __len__(self):
        return len(self.identifier_list)

    # ------------------------------------------------------------------
    # Utility methods (physics / geometry helpers)
    # ------------------------------------------------------------------
    def transform_hu_to_density(self, volume, bone_attenuation_multiplier):
        volume      = volume.to(torch.float32)
        air         = torch.where(volume <= -800)
        soft_tissue = torch.where((-800 < volume) & (volume <= 350))
        bone        = torch.where(350 < volume)
        density     = torch.empty_like(volume)
        density[air]         = volume[soft_tissue].min()
        density[soft_tissue] = volume[soft_tissue]
        density[bone]        = volume[bone] * bone_attenuation_multiplier
        density -= density.min()
        density /= density.max()
        return density

    def backproject_volume(self, target_poses_SOUV, target_proj,
                           source_arr_shape, spacing, device=torch.device('cpu')):
        S, O, U, V = [x.to(device=device, dtype=torch.float32)
                      for x in target_poses_SOUV]
        grids = backproj_grids_with_SOUV(
            source_arr_shape, spacing,
            S.unsqueeze(0), O.unsqueeze(0), U.unsqueeze(0), V.unsqueeze(0),
            target_proj.shape[2], target_proj.shape[1],
            # du=0.388, dv=0.388,
            du=self.detector_spacing, dv=self.detector_spacing,
            cu=(target_proj.shape[2] - 1) / 2.0,
            cv=(target_proj.shape[1] - 1) / 2.0,
            device=device,
        ) # (Z, Y, X, 2)
        grid = grids.reshape(
            1, source_arr_shape[0] * source_arr_shape[1], source_arr_shape[2], 2)
        flat = F.grid_sample(
            target_proj.reshape(1, 1, target_proj.shape[1], target_proj.shape[2]),
            # grid, align_corners=True, padding_mode='zeros')
            grid, align_corners=False, padding_mode='zeros')
        invalid = ((grid[..., 0] < -1) | (grid[..., 0] > 1) |
                   (grid[..., 1] < -1) | (grid[..., 1] > 1))
        # flat = flat.masked_fill(invalid.unsqueeze(1), -1.0)
        flat = flat.masked_fill(invalid.unsqueeze(1), 0)
        vol  = flat.reshape(*source_arr_shape)
        return self._normalize_intensity_bp(vol)

    def rigid_from_6params(self, rx_deg, ry_deg, rz_deg, tx, ty, tz):
        Rm = Rotation.from_euler("xyz", [rx_deg, ry_deg, rz_deg], degrees=True).as_matrix()
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = Rm
        T[:3, 3] = np.array([tx, ty, tz], dtype=np.float64)
        return T

    def canonicalize(self, volume, rigid_param, is_label=False):
        """Move the Subject's isocenter to the origin in world coordinates"""
        isocenter = volume.get_center()
        Tinv = np.array([
            [1, 0, 0, -isocenter[0]],
            [0, 1, 0, -isocenter[1]],
            [0, 0, 1, -isocenter[2]],
            [0, 0, 0,  1           ]], dtype=np.float64)
        T_user_lps = self.rigid_from_6params(*rigid_param)
        # T_user = np.linalg.inv(T_user)
        # LPS → RAS 转换: 翻转 x, y 轴
        S = np.diag([-1.0, -1.0, 1.0, 1.0])
        T_user_ras = S @ T_user_lps @ S
        affine_new = T_user_ras @ Tinv @ volume.affine
        if is_label:
            return LabelMap(tensor=volume.data, affine=affine_new)
        else:
            return ScalarImage(tensor=volume.data, affine=affine_new)

    def _extrinsic_cam2world_to_SOUV(self, extrinsic, reorient, sdd, device=None):
        R = extrinsic[:3, :3]
        t = extrinsic[:3, 3]
        if device is None:
            device = R.device if torch.is_tensor(R) else 'cpu'
        R = torch.as_tensor(R, dtype=torch.float32, device=device)
        t = torch.as_tensor(t, dtype=torch.float32, device=device)
        Sc = torch.tensor([0., 0., 0.],          device=device)
        Oc = torch.tensor([0., 0., float(sdd)],  device=device)
        Uc = torch.tensor([1., 0., 0.],          device=device)
        Vc = torch.tensor([0., 1., 0.],          device=device)
        S = (R @ reorient @ Sc) + t
        O = (R @ reorient @ Oc) + t
        U =  R @ reorient @ Uc
        V =  R @ reorient @ Vc
        return S.unsqueeze(0), O.unsqueeze(0), U.unsqueeze(0), V.unsqueeze(0)

    def _normalize_intensity(self, img, linear_clip=False, clip_range=None):
        """Normalize image intensities to [-1, 1]. Handles numpy and torch."""
        if linear_clip:
            if clip_range is not None:
                if isinstance(img, np.ndarray):
                    img = img.copy()
                    img[img < clip_range[0]] = clip_range[0]
                    img[img > clip_range[1]] = clip_range[1]
                else:
                    img = img.clone()
                    img[img < clip_range[0]] = clip_range[0]
                    img[img > clip_range[1]] = clip_range[1]
                normalized = (img - clip_range[0]) / (clip_range[1] - clip_range[0])
            else:
                img = img - img.min()
                arr_np = img.numpy() if torch.is_tensor(img) else img
                normalized = img / float(np.percentile(arr_np, 95)) * 0.95
        else:
            mn = img.min()
            mx = img.max()
            normalized = (img - mn) / (mx - mn)
        return normalized * 2 - 1

    def _normalize_intensity_bp(self, img, linear_clip=False, clip_range=None):
        """Normalize image intensities to [-1, 1]. Handles numpy and torch."""
        if linear_clip:
            if clip_range is not None:
                if isinstance(img, np.ndarray):
                    img = img.copy()
                    img[img < clip_range[0]] = clip_range[0]
                    img[img > clip_range[1]] = clip_range[1]
                else:
                    img = img.clone()
                    img[img < clip_range[0]] = clip_range[0]
                    img[img > clip_range[1]] = clip_range[1]
                normalized = (img - clip_range[0]) / (clip_range[1] - clip_range[0])
            else:
                img = img - img.min()
                arr_np = img.numpy() if torch.is_tensor(img) else img
                normalized = img / float(np.percentile(arr_np, 95)) * 0.95
        else:
            mn = img.min()
            mx = img.max()
            normalized = (img - mn) / (mx - mn)
        return torch.where(normalized > 0, normalized + 0.2 - 1, normalized - 1)

    def _resample_image(self, image, new_size, new_spacing, is_label=False):
        resampler = sitk.ResampleImageFilter()
        resampler.SetSize(new_size)
        resampler.SetOutputSpacing(new_spacing)
        direction  = np.array(image.GetDirection()).reshape(3, 3)
        half_size  = np.array(new_spacing) * (np.array(new_size) - 1) / 2.0
        new_origin = -(direction.dot(half_size))
        resampler.SetOutputOrigin(new_origin.tolist())
        resampler.SetOutputDirection(image.GetDirection())
        if is_label:
            resampler.SetInterpolator(sitk.sitkNearestNeighbor)
            resampler.SetDefaultPixelValue(0)
        else:
            resampler.SetInterpolator(sitk.sitkLinear)
            resampler.SetDefaultPixelValue(-2048)
        org_in    = np.array(image.GetOrigin(), dtype=float)
        dir_in    = np.array(image.GetDirection()).reshape(3, 3)
        sp_in     = np.array(image.GetSpacing(), dtype=float)
        sz_in     = np.array(image.GetSize(),    dtype=float)
        center_in = org_in + dir_in.dot(sp_in * (sz_in / 2.0))
        T = sitk.TranslationTransform(3)
        T.SetOffset(center_in.tolist())
        resampler.SetTransform(T)
        return resampler.Execute(image)

    def __split_dict(self, dict_to_split, split_num):
        idx_list  = list(range(len(dict_to_split)))
        idx_split = np.array_split(np.array(idx_list), split_num)
        return [dict_to_split[s[0]:s[-1] + 1] for s in idx_split]
