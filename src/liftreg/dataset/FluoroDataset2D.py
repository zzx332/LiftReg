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
        self.data_path = "/home/zzx/data/deepfluoro"
        self.drr_path  = os.path.join(data_path, "drr")
        self.cache_dir = os.path.join(data_path, "preprocessed_2d") # New cache dir for 2D
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

        self.reorient = torch.tensor(
            [[1, 0, 0, 0],
             [0, 0, 1, 0],
             [0, 1, 0, 0],
             [0, 0, 0, 1]], dtype=torch.float32)

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
                1020.0, 384, 384, 0.7255, 0.7255, 0, 0,
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
        datapath = Path(os.path.join(self.data_path, subject_id))
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
            pbar.update(i + 1)
        pbar.finish()

    def __getitem__(self, idx):
        idx        = idx % len(self.identifier_list)
        identifier = self.identifier_list[idx]
        meta       = self.meta_list[idx]

        npz           = np.load(os.path.join(self.cache_dir, f'{identifier}.npz'))
        source_proj   = npz['source_proj'].astype(np.float32)    # (P, H, W)
        target_proj   = npz['target_proj'].astype(np.float32)    # (P, H, W)
        spacing       = npz['spacing']
        
        if self.has_label and 'source_seg_proj' in npz:
            source_seg_proj = npz['source_seg_proj'].astype(np.float32)
        else:
            source_seg_proj = None

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
            'affine':            meta['affine'].clone(),
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
