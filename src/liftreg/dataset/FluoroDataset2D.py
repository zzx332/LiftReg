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
from liftreg.dataset.FluoroDataset import Warp
from liftreg.synthetic_data.utils import KWARGS

class FluoroDataset2D(Dataset):
    """
    2D DRR + 2D X-ray registration dataset.
    
    Data pipeline:
      Stage 1 – Preprocessing: runs once per identifier, results saved to
                {data_path}/preprocessed_2d/ as .npz + _meta.pt files.
                During this stage, the 3D CT is projected to 2D DRR using the target pose.
      Stage 2 – Online augmentation in __getitem__ (train phase only, CPU numpy).
      Stage 3 – Return [P, H, W] pairs of DRR and X-ray.
    """

    def __init__(self, data_path, phase=None, transform=None, option=None):
        self.data_path = data_path
        self.org_data_path = "/home/zzx/data/deepfluoro"
        self.drr_path  = os.path.join(data_path, "drr")
        self.warp_path = os.path.join(data_path, "warp_pose")
        self.warp_pose_cache = {}
        self.cache_dir = os.path.join(data_path, "preprocessed_2d") # New cache dir for 2D
        self.cache_3d_dir = os.path.join(data_path, "preprocessed") # New cache dir for 2D
        os.makedirs(self.cache_dir, exist_ok=True)

        self.phase     = phase
        self.transform = transform

        ind = ['train', 'val', 'test', 'debug'].index(phase)
        max_num = option[('max_num_for_loading', (-1, -1, -1, -1),
                          'max pairs to load per split')]
        if isinstance(max_num, tuple) or isinstance(max_num, list):
            self.max_num_for_loading = max_num[ind]
        else:
            self.max_num_for_loading = max_num

        self.has_label = option[('use_segmentation_map', False,
                                 'load segmentation maps')]
        self.spacing   = option[('spacing_to_refer', (1, 1, 1), '')]
        self.img_after_resize = tuple(option[('img_after_resize', (160, 160), '')])
        self.load_projection_interval = option[('load_projection_interval', 1, '')]
        self.apply_hu_clip = option[('apply_hu_clip', False, '')]

        # ---- Augmentation config (train phase only) ----
        self.enable_aug = (
            option[('augmentation_enable', True, 'enable data augmentation')]
            and (phase == 'train')
        )
        self.aug_noise_prob = option[('aug_noise_prob', 0.5, '')]
        self.aug_affine_prob     = option[('aug_affine_prob',     0.3, '')]
        self.aug_rotate_deg      = option[('aug_rotate_deg',      5.0, '')]
        self.aug_translate_vox   = option[('aug_translate_vox',   4.0, '')]
        self.aug_scale_range     = option[('aug_scale_range',     (0.95, 1.05), '')]
        self.reorient = torch.tensor(
            [[1, 0, 0, 0],
             [0, 0, 1, 0],
             [0, 1, 0, 0],
             [0, 0, 0, 1]], dtype=torch.float32)
        self.detector_spacing = option[('detector_spacing', 0.7255, '')]
        self.detector_width = option[('detector_width', 384, '')]

        self.meta_list = []
        self.get_identifier_list()
        self.init_img_pool()

    def get_identifier_list(self):
        self.identifier_list = [
            i[:-7] for i in os.listdir(self.drr_path) if i.endswith('.nii.gz')
        ]
        if self.max_num_for_loading > 0:
            self.identifier_list = self.identifier_list[:self.max_num_for_loading]

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
        poses = RigidTransform(target_poses)
        src, tgt = self._detector(poses, calibration=None)
        img_input = (tgt - src).norm(dim=-1).unsqueeze(1)
        src = RigidTransform(affine_inverse)(src)
        tgt = RigidTransform(affine_inverse)(tgt)
        return self._siddon(density, src, tgt, img_input).view(
            1, -1, self._detector.height, self._detector.width)

    def _load_dataset(self, subject_id):
        datapath = Path(os.path.join(self.org_data_path, subject_id))
        volume = datapath / "volume.nii.gz"
        mask = datapath / "mask.nii.gz"
        xrays = datapath / "xrays"
        segs = datapath / "segmentations"
        return volume, mask, xrays, segs

    def _preprocess_and_cache(self, identifier):
        npz_path  = os.path.join(self.cache_dir, f'{identifier}.npz')
        meta_path = os.path.join(self.cache_dir, f'{identifier}_meta.pt')
        if os.path.exists(npz_path) and os.path.exists(meta_path):
            return

        self._init_projector_if_needed()
        volume, mask, xrays, segs = self._load_dataset(identifier.split('_')[0])
        
        source_img = ScalarImage(volume) 
        source_seg = ScalarImage(mask)
        source_img = self.canonicalize(source_img)
        source_seg = self.canonicalize(source_seg)

        tfm = torchio.transforms.Compose([
            torchio.transforms.Resample((2.2, 2.2, 2.2)),
            torchio.transforms.CropOrPad((160, 160, 160), padding_mode=-2048),
        ])
        source_img = tfm(source_img)
        if self.has_label and source_seg is not None:
            source_seg = tfm(source_seg)

        density_t = self.transform_hu_to_density(
            source_img.data, bone_attenuation_multiplier=2.0)
        
        affine = torch.as_tensor(source_img.affine, dtype=torch.float32)

        target_poses, *_ = torch.load(
            os.path.join(self.drr_path,
                         identifier.split('_')[0] + '_pose.pt'),
            weights_only=False)['pose'][::self.load_projection_interval]
            
        # 1. Real X-ray (target_proj)
        target_proj_t = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(self.drr_path,
                         identifier + '.nii.gz')))
        target_proj_t = np.transpose(target_proj_t, (0, 2, 1))           
        target_proj_np = target_proj_t.astype(np.float32)

        # 2. Simulated DRR (source_proj) using target_poses
        source_proj_t = self._run_renderer(density_t.squeeze(0), target_poses, affine.inverse())
        source_proj_np = source_proj_t.squeeze(0).detach().cpu().numpy().astype(np.float32)
        
        # 3. Projected Label/Mask (source_seg_proj)
        if self.has_label and source_seg is not None:
            seg_density = source_seg.data.squeeze(0).to(torch.float32)
            source_seg_proj_t = self._run_renderer(seg_density, target_poses, affine.inverse())
            source_seg_proj_np = source_seg_proj_t.squeeze(0).detach().cpu().numpy().astype(np.float32)
            # threshold mask to 0 and 1
            source_seg_proj_np = (source_seg_proj_np > 0.1).astype(np.float32)
        else:
            source_seg_proj_np = None

        save_dict = dict(
            source_proj=source_proj_np,
            target_proj=target_proj_np,
            spacing=np.array(self.spacing, dtype=np.float32),
        )
        if self.has_label and source_seg_proj_np is not None:
            save_dict['source_seg_proj'] = source_seg_proj_np
            
        np.savez_compressed(npz_path, **save_dict)

        torch.save(dict(
            affine=affine,
            target_poses=target_poses,
        ), meta_path)

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
            rot_path = os.path.join(self.warp_path, f'{identifier.replace("drr", "poses_rot_")}.pt')
            xyz_path = os.path.join(self.warp_path, f'{identifier.replace("drr", "poses_xyz_")}.pt')
            self.warp_pose_cache[identifier] = {
                "poses_rot": torch.load(rot_path, map_location="cpu", weights_only=True),
                "poses_xyz": torch.load(xyz_path, map_location="cpu", weights_only=True),
            }
            pbar.update(i + 1)
        pbar.finish()

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

    def _build_shared_affine(self, vol_shape):
        if np.random.rand() >= self.aug_affine_prob:
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

    def _apply_shared_affine_3d(self, source, density, source_seg, weights, affine):
        grid, voxel_to_input = self._build_shared_affine(source.shape)
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

    def __getitem__(self, idx):
        idx        = idx % len(self.identifier_list)
        identifier = self.identifier_list[idx]
        meta       = self.meta_list[idx]
        cache = self.warp_pose_cache[identifier]
        poses_rot = cache["poses_rot"]
        poses_xyz = cache["poses_xyz"]

        npz           = np.load(os.path.join(self.cache_dir, f'{identifier}.npz'))
        npz_3d        = np.load(os.path.join(self.cache_3d_dir, f'{identifier}.npz'))
        source        = np.ascontiguousarray(npz_3d['source'].astype(np.float32))         # (D, H, W)
        density       = np.ascontiguousarray(npz_3d['density'].astype(np.float32))        # (D, H, W)
        source_seg    = (np.ascontiguousarray(npz_3d['source_seg'].astype(np.float32))
                         if self.has_label else None)
        weights       = (np.ascontiguousarray(npz_3d['weights'].astype(np.float32))
                         if self.has_label else None)
        affine        = meta['affine'].clone()
        source_proj   = npz['source_proj'].astype(np.float32)    # (P, H, W)
        target_proj   = npz['target_proj'].astype(np.float32)    # (P, H, W)
        spacing       = npz['spacing']
        
        if self.has_label and 'source_seg_proj' in npz:
            source_seg_proj = npz['source_seg_proj'].astype(np.float32)
        else:
            source_seg_proj = None
        if self.enable_aug:
            affine_grid = None
            (source_aug, density, source_seg, weights,
             affine, affine_grid) = self._apply_shared_affine_3d(
                source, density, source_seg, weights, affine)
            if affine_grid is not None:
                warp = Warp(density, source_seg, weights, poses_rot, poses_xyz, affine)
                warped_density, warped_mask, displacement = warp()
                self._init_projector_if_needed()
                density_t = torch.from_numpy(density).to(dtype=torch.float32)
                source_proj = self._run_renderer(
                    density_t, meta['target_poses'], affine.inverse())[0].detach().cpu().numpy()
                target_proj = self._run_renderer(
                    warped_density, meta['target_poses'], affine.inverse())[0].detach().cpu().numpy()
        # Data norm scaling to match expectations (0-1 usually better for 2D nets)
        p_min = float(target_proj.min())
        p_max = float(target_proj.max())
        target_proj = (target_proj - p_min) / (p_max - p_min + 1e-6)
        
        s_min = float(source_proj.min())
        s_max = float(source_proj.max())
        source_proj = (source_proj - s_min) / (s_max - s_min + 1e-6)

        source_proj = F.interpolate(
            torch.from_numpy(source_proj).unsqueeze(0),
            size=self.img_after_resize,
            mode='bilinear',
            align_corners=False,
        ).squeeze(0).numpy().astype(np.float32)

        target_proj = F.interpolate(
            torch.from_numpy(target_proj).unsqueeze(0),
            size=self.img_after_resize,
            mode='bilinear',
            align_corners=False,
        ).squeeze(0).numpy().astype(np.float32)

        if source_seg_proj is not None:
            source_seg_proj = F.interpolate(
                torch.from_numpy(source_seg_proj).unsqueeze(0),
                size=self.img_after_resize,
                mode='nearest',
            ).squeeze(0).numpy().astype(np.float32)

        sample = {
            'source_proj':       source_proj,          # (P,H,W) -> treating DRR as source
            'target_proj':       target_proj,          # (P,H,W) -> treating X-ray as target
            'spacing':           spacing,
            'affine':            affine,
            'target_poses':      meta['target_poses'],
        }
        if source_seg_proj is not None:
            sample['source_label'] = source_seg_proj
            
        if self.transform:
            sample['source_proj'] = self.transform(sample['source_proj'])

        return sample, identifier

    def __len__(self):
        return len(self.identifier_list)

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

    def canonicalize(self, volume):
        isocenter = volume.get_center()
        Tinv = np.array([
            [1, 0, 0, -isocenter[0]],
            [0, 1, 0, -isocenter[1]],
            [0, 0, 1, -isocenter[2]],
            [0, 0, 0,  1           ]], dtype=np.float64)
        return ScalarImage(tensor=volume.data, affine=Tinv.dot(volume.affine))
