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
from diffdrr.pose import RigidTransform
from ..utils.sdct_projection_utils import backproj_grids_with_SOUV
from polypose.weights import compute_weights

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
        self.data_path = "/home/zzx/data/deepfluoro"
        self.drr_path  = os.path.join(data_path, "drr")
        self.cache_dir = os.path.join(data_path, "preprocessed")
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
        self.load_projection_interval = option[('load_projection_interval', 1, '')]
        self.apply_hu_clip = option[('apply_hu_clip', False, '')]

        # ---- Augmentation config (train phase only) ----
        self.enable_aug = (
            option[('augmentation_enable', True, 'enable data augmentation')]
            and (phase == 'train')
        )
        self.aug_gamma_prob      = option[('aug_gamma_prob',      0.4, '')]
        self.aug_brightness_prob = option[('aug_brightness_prob', 0.5, '')]
        self.aug_noise_prob      = option[('aug_noise_prob',      0.5, '')]
        self.aug_lr_flip_prob    = option[('aug_lr_flip_prob',    0.5, '')]
        # self.aug_proj_mask_prob  = option[('aug_proj_mask_prob',  0.3, '')]

        self.reorient = torch.tensor(
            [[1, 0, 0, 0],
             [0, 0, 1, 0],
             [0, 1, 0, 0],
             [0, 0, 0, 1]], dtype=torch.float32)

        # meta_list holds lightweight per-sample dicts (poses, affine)
        self.meta_list = []

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
                1020.0, 718, 718, 0.388, 0.388, 0, 0,
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
        datapath = Path(os.path.join(self.data_path, subject_id))

        # Make paths to the relevant images
        volume = datapath / "volume.nii.gz"
        mask = datapath / "mask.nii.gz"
        xrays = datapath / "xrays"
        segs = datapath / "segmentations"

        # return ScalarImage(volume), LabelMap(mask), xrays, LabelMap(segs)
        return volume, mask, xrays, segs
    # ------------------------------------------------------------------
    # Stage 1 – Preprocessing & caching  (runs once per identifier)
    # ------------------------------------------------------------------
    def _preprocess_and_cache(self, identifier):
        """Preprocess one CT/pose pair and save results to disk."""
        npz_path  = os.path.join(self.cache_dir, f'{identifier}.npz')
        meta_path = os.path.join(self.cache_dir, f'{identifier}_meta.pt')
        if os.path.exists(npz_path) and os.path.exists(meta_path):
            return  # already cached

        self._init_projector_if_needed()
        # ---- 1. Load & preprocess CT volume ----
        # source_img, source_seg, xrays, _ = self._load_dataset(identifier.split('_')[0])
        volume, mask, xrays, segs = self._load_dataset(identifier.split('_')[0])
        # source_img = ScalarImage(
        #     os.path.join(self.data_path,
        #                  identifier.split('_')[0] + '_source.nii.gz'))
        source_img = ScalarImage(volume) # X, Y, Z
        source_seg = ScalarImage(mask)
        source_img = self.canonicalize(source_img)
        source_seg = self.canonicalize(source_seg)
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
        # target_proj_t = self._run_renderer(
        #     density_t.squeeze(0), target_poses, affine.inverse())
        target_proj_t = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(self.drr_path,
                         identifier + '.nii.gz')))
        target_proj_t = np.transpose(target_proj_t, (0, 2, 1))  # 把 proj变成hw顺序            
        # (1, P, H, W) → squeeze batch → (P, H, W)
        # target_proj_np = (target_proj_t.squeeze(0)
                        #   .detach().cpu().numpy().astype(np.float32))
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

        # ---- 4. Save to disk ----
        save_dict = dict(
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
            self.meta_list.append(meta)
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

    def _augment_2d_proj(self, proj):
        """
        X-ray physics-aware augmentation for a (P, H, W) float32 array in [-1, 1].

        Three physics-motivated transforms (mutually exclusive, one chosen per call):
          1. Poisson noise  – models photon-counting shot noise in X-ray detectors.
          2. Gaussian blur  – models focal-spot blur / detector PSF degradation.
          3. Log-transform  – models the Beer-Lambert law nonlinearity that maps
                              raw detector intensity to equivalent path length;
                              a random strength parameter controls its severity.

        Each transform is applied with probability aug_noise_prob.
        Brightness/contrast jitter is always applied independently before them.
        """
        proj = proj.copy()
        aug_tag = False 
        if np.random.rand() < self.aug_noise_prob:
            choice = np.random.randint(0, 3)

            # Rescale [-1, 1] → [0, 1] for all three transforms
            # p01 = np.clip((proj + 1.0) * 0.5, 0.0, 1.0).astype(np.float32)
            p_min = float(proj.min())
            p_max = float(proj.max())
            p_range = max(p_max - p_min, 1e-6)
            p01 = np.clip((proj - p_min) / p_range, 0.0, 1.0).astype(np.float32)

            if choice == 0:
                # --- Poisson noise ---
                # Simulate photon counts: scale p01 to a mean count level,
                # draw Poisson samples, then rescale back.
                # lam controls SNR: higher lam = less noise.
                lam = float(np.random.uniform(300.0, 1000.0))
                counts = np.random.poisson(p01 * lam).astype(np.float32)
                p01 = np.clip(counts / lam, 0.0, 1.0)

            elif choice == 1:
                # --- Gaussian blur (detector PSF / focal-spot blur) ---
                # Apply a separable Gaussian filter independently to each
                # projection frame using a small kernel.
                from scipy.ndimage import gaussian_filter
                sigma_blur = float(np.random.uniform(0.5, 2.0))
                for i in range(p01.shape[0]):
                    p01[i] = gaussian_filter(p01[i], sigma=sigma_blur).astype(np.float32)

            else:
                # --- Log-transform (Beer-Lambert nonlinearity) ---
                # I_out = log(1 + alpha * I) / log(1 + alpha)
                # alpha controls transform strength: large alpha compresses highlights.
                alpha = float(np.random.uniform(2.0, 10.0))
                p01 = (np.log1p(alpha * p01) / np.log1p(alpha)).astype(np.float32)

            # proj = np.clip(p01 * 2.0 - 1.0, -1.0, 1.0)
            proj = np.clip(p01 * p_range + p_min, p_min, p_max)
            aug_tag = True
        # ---- Rectangular occlusion (simulate surgical instruments / table edge) ----
        # if np.random.rand() < self.aug_proj_mask_prob:
        #     _, H, W = proj.shape
        #     rh = int(H * np.random.uniform(0.05, 0.20))
        #     rw = int(W * np.random.uniform(0.05, 0.20))
        #     y0 = np.random.randint(0, max(1, H - rh))
        #     x0 = np.random.randint(0, max(1, W - rw))
        #     proj[:, y0:y0 + rh, x0:x0 + rw] = float(proj.min())

        return proj.astype(np.float32), aug_tag  # 已由上面的 clip 保证在原始值域内

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
        meta       = self.meta_list[idx]

        # ---- Load preprocessed arrays from disk (fast np.load) ----
        npz           = np.load(os.path.join(self.cache_dir, f'{identifier}.npz'))
        source        = npz['source'].astype(np.float32)         # (D, H, W)
        density       = npz['density'].astype(np.float32)        # (D, H, W)
        target_proj   = npz['target_proj'].astype(np.float32)    # (P, H, W)
        target_volume = npz['target_volume'].astype(np.float32)  # (D, H, W)
        spacing       = npz['spacing']
        source_seg    = (npz['source_seg'].astype(np.float32)
                         if self.has_label else None)
        weights       = (npz['weights'].astype(np.float32)
                         if self.has_label else None)
        affine        = meta['affine'].clone()
        affine        = meta['affine'].clone()

        # ---- Online augmentation (train phase only) ----
        if self.enable_aug:
            # if np.random.rand() < self.aug_lr_flip_prob:
            #     (source, density, source_seg, weights,
            #      target_proj, target_volume, affine) = self._apply_lr_flip(
            #         source, density, source_seg, weights,
            #         target_proj, target_volume, affine)

            source_aug        = self._augment_3d_intensity(source)
            # density       = self._augment_3d_intensity(density)
            # target_volume = self._augment_3d_intensity(target_volume)
            target_proj_aug, aug_tag = self._augment_2d_proj(target_proj)
            if aug_tag:
                proj_for_bp = (torch.from_numpy(target_proj_aug).flip(dims=[1, 2]))
                target_volume_aug = self.backproject_volume(
                    meta['target_poses_SOUV'], proj_for_bp, source.shape,
                    device=torch.device('cpu'))
                target_volume_aug = np.transpose(target_volume_aug, (2, 1, 0))
            else:
                target_volume_aug = target_volume
        else:
            source_aug = source
            target_volume_aug = target_volume
            target_proj_aug = target_proj
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
            sample['weights'] = np.expand_dims(weights, 0)  # (1,D,H,W)

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
            du=0.388, dv=0.388,
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

    def canonicalize(self, volume):
        isocenter = volume.get_center()
        Tinv = np.array([
            [1, 0, 0, -isocenter[0]],
            [0, 1, 0, -isocenter[1]],
            [0, 0, 1, -isocenter[2]],
            [0, 0, 0,  1           ]], dtype=np.float64)
        return ScalarImage(tensor=volume.data, affine=Tinv.dot(volume.affine))
        # Move the Subject's isocenter to the origin in world coordinates
        # for image in subject.get_images(intensity_only=False):
        #     isocenter = image.get_center()
        #     Tinv = np.array(
        #         [
        #             [1.0, 0.0, 0.0, -isocenter[0]],
        #             [0.0, 1.0, 0.0, -isocenter[1]],
        #             [0.0, 0.0, 1.0, -isocenter[2]],
        #             [0.0, 0.0, 0.0, 1.0],
        #         ]
        #     )
        #     image.affine = Tinv.dot(image.affine)
        # return subject

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
