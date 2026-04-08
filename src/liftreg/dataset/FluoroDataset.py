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

class FluoroDataset(Dataset):
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
        self.org_data_path = "/home/zzx/data/deepfluoro"
        self.drr_path  = os.path.join(self.data_path, "drr")
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
        self.supervise = option[('supervise', False,
                                 'use warped density GT for supervision')]
        self.spacing   = option[('spacing_to_refer', (1, 1, 1), '')]
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
        self.detector_width = option[('detector_width', 384, '')]

        self.reorient = torch.tensor(
            [[1, 0, 0, 0],
             [0, 0, 1, 0],
             [0, 1, 0, 0],
             [0, 0, 0, 1]], dtype=torch.float32)

        # meta_list holds lightweight per-sample dicts (poses, affine)
        self.meta_cache = {}
        self.get_identifier_list()
        self.init_img_pool()

    # ------------------------------------------------------------------
    # Identifier list
    # ------------------------------------------------------------------
    def get_identifier_list(self):
        self.identifier_list = [
            i[:-7] for i in os.listdir(self.drr_path) if i.endswith('.nii.gz')
        ]
        if self.max_num_for_loading > 0:
            self.identifier_list = self.identifier_list[:self.max_num_for_loading]

    # ------------------------------------------------------------------
    # Projector (lazy init, CPU only)
    # ------------------------------------------------------------------
    def _init_projector_if_needed(self):
        if not hasattr(self, '_siddon') or self._siddon is None:
            self._siddon = Siddon(voxel_shift=0.0)
        if not hasattr(self, '_detector') or self._detector is None:
            self._detector = Detector(
                # 1020.0, 718, 718, 0.388, 0.388, 0, 0,
                1020.0, self.detector_width, self.detector_width, self.detector_spacing, self.detector_spacing, 0, 0,
                self.reorient, reverse_x_axis=True, n_subsample=None,
            )

    def _run_renderer(self, density, target_poses, affine_inverse):
        """Forward DRR render, returns (1, P, H, W) tensor on CPU."""
        poses = RigidTransform(target_poses)
        src, tgt = self._detector(poses, calibration=None)
        img_input = (tgt - src).norm(dim=-1).unsqueeze(1)
        src = RigidTransform(affine_inverse)(src)
        tgt = RigidTransform(affine_inverse)(tgt)
        return self._siddon(density, src, tgt, img_input).view(
            1, -1, self._detector.height, self._detector.width)

    def _load_dataset(self, subject_id, dataset="deepfluoro"):
        # datapath = Path(rf"/home/zzx/data/deepfluoro/subject{subject_id:02d}")
        datapath = Path(os.path.join(self.org_data_path, subject_id))

        # Make paths to the relevant images
        volume = datapath / "volume.nii.gz"
        mask = datapath / "mask.nii.gz"
        xrays = datapath / "xrays"
        segs = datapath / "segmentations"

        # return ScalarImage(volume), LabelMap(mask), xrays, LabelMap(segs)
        return volume, mask, xrays, segs

    def _cache_is_valid(self, npz_path, meta_path):
        if not (os.path.exists(npz_path) and os.path.exists(meta_path)):
            return False

        try:
            with np.load(npz_path) as npz:
                required_keys = {
                    'cache_version', 'source', 'density', 'target_proj', 'target_volume', 'spacing'
                }
                if self.has_label:
                    required_keys.update({'source_seg', 'weights'})
                if self.supervise:
                    required_keys.add('warped_density_gt')
                if self.offline_aug_variants > 0:
                    required_keys.update({
                        'aug_source', 'aug_density', 'aug_target_proj',
                        'aug_target_volume', 'aug_affines'
                    })
                    if self.has_label:
                        required_keys.update({'aug_source_seg', 'aug_weights'})
                    if self.supervise:
                        required_keys.update({'aug_warped_density_gt', 'aug_displacement_gt'})

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
        self, identifier, source, density, source_seg, weights, affine, target_poses, target_poses_SOUV
    ):
        if self.offline_aug_variants <= 0 or source_seg is None or weights is None:
            return None

        rot_path = os.path.join(self.warp_path, f'{identifier.replace("drr", "poses_rot_")}.pt')
        xyz_path = os.path.join(self.warp_path, f'{identifier.replace("drr", "poses_xyz_")}.pt')
        poses_rot = torch.load(rot_path, map_location="cpu", weights_only=True)
        poses_xyz = torch.load(xyz_path, map_location="cpu", weights_only=True)

        aug_source = []
        aug_density = []
        aug_source_seg = []
        aug_weights = []
        aug_target_proj = []
        aug_target_volume = []
        aug_affines = []
        aug_warped_density_gt = [] if self.supervise else None
        aug_displacement_gt = [] if self.supervise else None

        for _ in range(self.offline_aug_variants):
            (source_aug, density_aug, source_seg_aug, weights_aug,
             affine_aug, affine_grid) = self._apply_shared_affine_3d(
                source, density, source_seg, weights, affine.clone(), force=True)

            warp = Warp(density_aug, source_seg_aug, weights_aug, poses_rot, poses_xyz, affine_aug)
            warped_density, _, displacement = warp()

            target_proj_aug = self._run_renderer(
                warped_density, target_poses, affine_aug.inverse())[0].detach().cpu().numpy()
            proj_for_bp = torch.from_numpy(target_proj_aug).flip(dims=[1, 2])
            target_volume_aug = self.backproject_volume(
                target_poses_SOUV, proj_for_bp, source_aug.shape,
                device=torch.device('cpu')).cpu().numpy()
            target_volume_aug = np.transpose(target_volume_aug, (2, 1, 0))
            target_volume_aug = self._apply_affine_grid(
                target_volume_aug, affine_grid, mode='bilinear', padding_mode='border')

            aug_source.append(np.ascontiguousarray(source_aug.astype(np.float32)))
            aug_density.append(np.ascontiguousarray(density_aug.astype(np.float32)))
            aug_source_seg.append(np.ascontiguousarray(source_seg_aug.astype(np.float32)))
            aug_weights.append(np.ascontiguousarray(weights_aug.astype(np.float32)))
            aug_target_proj.append(np.ascontiguousarray(target_proj_aug.astype(np.float32)))
            aug_target_volume.append(np.ascontiguousarray(target_volume_aug.astype(np.float32)))
            aug_affines.append(affine_aug.numpy().astype(np.float32))
            if self.supervise:
                aug_warped_density_gt.append(
                    np.ascontiguousarray(warped_density.detach().cpu().numpy().astype(np.float32))
                )
                aug_displacement_gt.append(
                    np.transpose(
                        displacement.detach().cpu().numpy().astype(np.float32),
                        (3, 0, 1, 2)
                    )
                )

        if len(aug_source) == 0:
            return None

        save_dict = {
            'aug_source': np.stack(aug_source, axis=0).astype(np.float32),
            'aug_density': np.stack(aug_density, axis=0).astype(np.float32),
            'aug_source_seg': np.stack(aug_source_seg, axis=0).astype(np.float32),
            'aug_weights': np.stack(aug_weights, axis=0).astype(np.float32),
            'aug_target_proj': np.stack(aug_target_proj, axis=0).astype(np.float32),
            'aug_target_volume': np.stack(aug_target_volume, axis=0).astype(np.float32),
            'aug_affines': np.stack(aug_affines, axis=0).astype(np.float32),
        }
        if self.supervise:
            save_dict['aug_warped_density_gt'] = np.stack(aug_warped_density_gt, axis=0).astype(np.float32)
            save_dict['aug_displacement_gt'] = np.stack(aug_displacement_gt, axis=0).astype(np.float32)
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

        self._init_projector_if_needed()
        # ---- 1. Load & preprocess CT volume ----
        # source_img, source_seg, xrays, _ = self._load_dataset(identifier.split('_')[0])
        volume, mask, xrays, segs = self._load_dataset(identifier.split('_')[0])
        # source_img = ScalarImage(
        #     os.path.join(self.data_path,
        #                  identifier.split('_')[0] + '_source.nii.gz'))
        source_img = ScalarImage(volume) # X, Y, Z
        # source_seg = ScalarImage(mask)
        source_seg = LabelMap(mask)
        source_img = self.canonicalize(source_img)
        source_seg = self.canonicalize(source_seg, is_label=True)
        # subject = self.canonicalize(subject)
        subject = Subject(volume=source_img, mask=source_seg)
        _, weights = compute_weights(subject, labels=[[1, 2, 3, 4, 7], [5], [6]])

        tfm = torchio.transforms.Compose([
            torchio.transforms.Resample((2.2, 2.2, 2.2)),
            torchio.transforms.CropOrPad((160, 160, 160), padding_mode=-2048),
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

        # ---- 2. Load pose & render DRR ----
        target_poses, *_ = torch.load(
            os.path.join(self.drr_path,
                         identifier.split('_')[0] + '_pose.pt'),
            weights_only=False)['pose'][::self.load_projection_interval]
        # render要求的输入图像是xyz顺序
        target_proj_t = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(self.drr_path,
                         identifier + '.nii.gz')))
        target_proj_t = np.transpose(target_proj_t, (0, 2, 1))  # 把 proj变成hw顺序            
        target_proj_np = target_proj_t.astype(np.float32)

        # ---- 3. Back-project 2D → 3D volume ----
        target_poses_SOUV = self._extrinsic_cam2world_to_SOUV(
            target_poses, self.reorient[:3, :3], sdd=1020.0)
        # proj_for_bp = (torch.from_numpy(target_proj_np)
        #                .permute(0, 2, 1).flip(dims=[2]))
        proj_for_bp = torch.from_numpy(target_proj_np).flip(dims=[1, 2])
        target_volume = self.backproject_volume(
            target_poses_SOUV, proj_for_bp, source_arr.shape,
            device=torch.device('cpu')).permute(2, 1, 0) # (X, Y, Z)
        target_volume_np = target_volume.detach().cpu().numpy().astype(np.float32) # (X, Y, Z)

        # ---- 3b. Load warped density GT (if available) ----
        warped_gt_name = identifier.replace("drr", "density_") + ".nii.gz"
        warped_gt_file = os.path.join(self.warped_data_path, warped_gt_name)

        # ---- 4. Save to disk ----
        save_dict = dict(
            cache_version=np.array([self.cache_version], dtype=np.int16),
            source=source_arr,
            density=density_arr,
            target_proj=target_proj_np, # (P, H, W)
            target_volume=target_volume_np,
            spacing=np.array(self.spacing, dtype=np.float32),
        )
        if self.has_label and source_seg is not None:
            save_dict['source_seg'] = (
                source_seg.data.squeeze(0).numpy().astype(np.float32))
            save_dict['weights'] = weights.detach().cpu().numpy().astype(np.float32)
        if os.path.exists(warped_gt_file):
            warped_gt_img = ScalarImage(warped_gt_file)
            warped_gt_img = self.canonicalize(warped_gt_img)
            tfm_wgt = torchio.transforms.Compose([
                torchio.transforms.Resample((2.2, 2.2, 2.2)),
                torchio.transforms.CropOrPad((160, 160, 160), padding_mode=0),
            ])
            warped_gt_img = tfm_wgt(warped_gt_img)
            save_dict['warped_density_gt'] = (
                warped_gt_img.data.squeeze(0).numpy().astype(np.float32))
        elif self.supervise:
            save_dict['warped_density_gt'] = np.zeros_like(source_arr, dtype=np.float32)

        aug_dict = self._build_offline_augmented_bank_3d(
            identifier,
            source_arr,
            density_arr,
            save_dict.get('source_seg', None),
            save_dict.get('weights', None),
            affine,
            target_poses,
            target_poses_SOUV,
        )
        if aug_dict is not None:
            save_dict.update(aug_dict)
        np.savez_compressed(npz_path, **save_dict)

        torch.save(dict(
            affine=affine,
            target_poses=target_poses,
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
            align_corners=True,
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
            align_corners=True,
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
            source_seg    = (np.ascontiguousarray(npz['source_seg'].astype(np.float32))
                             if self.has_label else None)
            weights       = (np.ascontiguousarray(npz['weights'].astype(np.float32))
                             if self.has_label else None)

            # ---- Load warped density GT from cache (for 3D volume similarity loss) ----
            if self.supervise:
                has_warped_gt = 'warped_density_gt' in npz
                warped_density_gt = (
                    np.ascontiguousarray(npz['warped_density_gt'].astype(np.float32))
                    if has_warped_gt else np.zeros_like(source))
                displacement_gt = np.zeros((3, *source.shape), dtype=np.float32)

            if self.enable_aug and self.offline_aug_variants > 0 and 'aug_source' in npz:
                if np.random.rand() < self.aug_affine_prob:
                    aug_idx = np.random.randint(npz['aug_source'].shape[0])
                    source = np.ascontiguousarray(npz['aug_source'][aug_idx].astype(np.float32))
                    density = np.ascontiguousarray(npz['aug_density'][aug_idx].astype(np.float32))
                    target_proj = np.ascontiguousarray(npz['aug_target_proj'][aug_idx].astype(np.float32))
                    target_volume = np.ascontiguousarray(npz['aug_target_volume'][aug_idx].astype(np.float32))
                    affine = torch.from_numpy(npz['aug_affines'][aug_idx].astype(np.float32))
                    if self.has_label and 'aug_source_seg' in npz:
                        source_seg = np.ascontiguousarray(npz['aug_source_seg'][aug_idx].astype(np.float32))
                    if self.has_label and 'aug_weights' in npz:
                        weights = np.ascontiguousarray(npz['aug_weights'][aug_idx].astype(np.float32))
                    if self.supervise and 'aug_warped_density_gt' in npz:
                        warped_density_gt = np.ascontiguousarray(
                            npz['aug_warped_density_gt'][aug_idx].astype(np.float32))
                    if self.supervise and 'aug_displacement_gt' in npz:
                        displacement_gt = np.ascontiguousarray(
                            npz['aug_displacement_gt'][aug_idx].astype(np.float32))
        # ---- Online augmentation (train phase only) ----
        if self.enable_aug:
            source_aug = source
            target_volume_aug = target_volume
            target_proj_aug = target_proj
            source_aug = self._augment_3d_intensity(source_aug)
            density = self._augment_3d_intensity(density)
            target_proj_aug, aug_tag = self._augment_2d_projs_shared(target_proj_aug)
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
            'spacing':           spacing,
            'affine':            affine,
            'target_poses':      meta['target_poses'],
            'target_poses_SOUV': meta['target_poses_SOUV'],
        }
        if self.has_label and source_seg is not None:
            sample['source_label'] = np.expand_dims(source_seg, 0)  # (1,D,H,W)
        if self.has_label and weights is not None:
            weights_t = torch.from_numpy(weights).unsqueeze(0).float()
            weights_interp = F.interpolate(
                weights_t, size=source.shape, mode='trilinear', align_corners=False
            ).squeeze(0).numpy().astype(np.float32)
            sample['weights'] = weights_interp  # (K,D,H,W)
        if self.supervise:
            warped_density_gt = np.ascontiguousarray(warped_density_gt.astype(np.float32))
            displacement_gt = np.ascontiguousarray(displacement_gt.astype(np.float32))
            sample['warped_density_gt'] = np.expand_dims(warped_density_gt, 0)
            sample['displacement_gt'] = displacement_gt

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
                           source_arr_shape, device=torch.device('cpu')):
        S, O, U, V = [x.to(device=device, dtype=torch.float32)
                      for x in target_poses_SOUV]
        grids = backproj_grids_with_SOUV(
            source_arr_shape,
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
            grid, align_corners=True, padding_mode='zeros')
        invalid = ((grid[..., 0] < -1) | (grid[..., 0] > 1) |
                   (grid[..., 1] < -1) | (grid[..., 1] > 1))
        flat = flat.masked_fill(invalid.unsqueeze(1), -1.0)
        vol  = flat.reshape(*source_arr_shape)
        return self._normalize_intensity(vol)

    def canonicalize(self, volume, is_label=False):
        """Move the Subject's isocenter to the origin in world coordinates"""
        isocenter = volume.get_center()
        Tinv = np.array([
            [1, 0, 0, -isocenter[0]],
            [0, 1, 0, -isocenter[1]],
            [0, 0, 1, -isocenter[2]],
            [0, 0, 0,  1           ]], dtype=np.float64)
        # return ScalarImage(tensor=volume.data, affine=)
        if is_label:
            return LabelMap(tensor=volume.data, affine=Tinv.dot(volume.affine))
        else:
            return ScalarImage(tensor=volume.data, affine=Tinv.dot(volume.affine))

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


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        return torch.from_numpy(sample)
