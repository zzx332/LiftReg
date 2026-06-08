from __future__ import division, print_function

import json
import os
from pathlib import Path

import numpy as np
import progressbar as pb
import SimpleITK as sitk
import torch
import torch.nn.functional as F
import torchio
from diffdrr.detector import Detector
from diffdrr.pose import RigidTransform
from diffdrr.renderers import Siddon
from polypose.weights import compute_weights
from torch.utils.data import Dataset
from torchio import LabelMap, ScalarImage, Subject
from xvr.dicom import read_xray
from liftreg.dataset.FluoroDataset import Warp
from liftreg.utils.compute_weights_2d import compute_weights_2d
from torchvision.transforms.functional import center_crop

_DRR_SUFFIX = "-DRR.mhd"
_PHASE_LIST_FILES = {
    "train": "train_list.json",
    "val": "val_list.json",
    "test": "test_list.json",
    "debug": "debug_list.json",
}
_PHASE_MANIFEST_KEYS = {
    "train": "train_identifiers",
    "val": "val_identifiers",
    "test": "test_identifiers",
    "debug": "debug_identifiers",
}


def identifier_from_drr_filename(filename: str) -> str:
    if not filename.endswith(_DRR_SUFFIX):
        raise ValueError(f"Not a DRR mhd filename: {filename}")
    return filename[: -len(_DRR_SUFFIX)]


class TipsDataset2D(Dataset):
    """
    2D DRR + 2D X-ray registration dataset.

    Data pipeline:
      Stage 1 – Preprocessing: runs once per identifier, results saved to
                {data_path}/preprocessed_2d/ as .npz + _meta.pt files.
                During this stage, the 3D CT is projected to 2D DRR using the target pose.
      Stage 2 – Optional offline augmentation bank generation during preprocessing.
      Stage 3 – Return cached [P, H, W] pairs of DRR and X-ray.
    """

    def __init__(self, data_path, phase=None, transform=None, option=None):
        self.data_path = data_path
        # self.org_data_path = "/home/zzx/data/deepfluoro"
        # self.xray_path = os.path.join(data_path, "xrays")
        self.warp_path = os.path.join(data_path, "warp_pose")
        self.cache_dir = os.path.join(data_path, "preprocessed_xray2d")
        os.makedirs(self.cache_dir, exist_ok=True)

        self.phase = phase
        self.transform = transform
        self.cache_version = 2

        ind = ["train", "val", "test", "debug"].index(phase)
        max_num = option[("max_num_for_loading", (-1, -1, -1, -1), "max pairs to load per split")]
        if isinstance(max_num, (tuple, list)):
            self.max_num_for_loading = max_num[ind]
        else:
            self.max_num_for_loading = max_num

        self.has_label = option[("use_segmentation_map", False, "load segmentation maps")]
        self.provide_weights = option[("provide_weights", False, "online compute polyrigid weights")]
        self.num_segments = int(option[("num_segments", 3, "K for polyrigid")])
        self.label_groups = option[("label_groups", None, "list[int] | list[list[int]]; None=auto sorted unique")]
        self.weightfn = option[("weightfn", "gravity", "gravity|invdf|exp")]
        if self.provide_weights and not self.has_label:
            raise ValueError(
                "provide_weights=True requires use_segmentation_map=True so that "
                "source_label is available to compute polyrigid weights."
            )
        self.spacing = option[("spacing_to_refer", (1, 1, 1), "")]
        self.img_after_resize = tuple(option[("img_after_resize", (160, 160), "")])
        self.load_projection_interval = option[("load_projection_interval", 1, "")]
        self.apply_hu_clip = option[("apply_hu_clip", False, "")]

        self.enable_aug = option[("augmentation_enable", True, "enable data augmentation")] and (phase == "train")
        self.offline_aug_variants = int(option[("offline_aug_variants", 4, "number of cached affine augmentation variants")])
        if not self.enable_aug:
            self.offline_aug_variants = 0
        self.aug_noise_prob = option[("aug_noise_prob", 0.5, "")]
        self.aug_affine_prob = option[("aug_affine_prob", 0.3, "")]
        self.aug_rotate_deg = option[("aug_rotate_deg", 5.0, "")]
        self.aug_translate_vox = option[("aug_translate_vox", 4.0, "")]
        self.aug_scale_range = option[("aug_scale_range", (0.95, 1.05), "")]
        self.reorient = torch.tensor(
            [[1, 0, 0, 0],
             [0, 0, 1, 0],
             [0, 1, 0, 0],
             [0, 0, 0, 1]], dtype=torch.float32)
        self.detector_spacing = option[("detector_spacing", 0.7255, "")]
        self.detector_width = option[("detector_width", 384, "")]
        self.split_list_json = option[(
            "split_list_json",
            None,
            "path to split json (list or manifest); None -> {data_path}/{phase}_list.json",
        )]

        self.identifier_list = []
        self.get_identifier_list()
        self.init_img_pool()

    def _default_split_list_path(self):
        list_name = _PHASE_LIST_FILES.get(self.phase)
        if list_name is None:
            return None
        path = os.path.join(self.data_path, list_name)
        return path if os.path.isfile(path) else None

    def _resolve_split_list_path(self):
        cfg = self.split_list_json
        if cfg is not None:
            if isinstance(cfg, dict):
                path = cfg.get(self.phase)
                if path:
                    return path
            elif isinstance(cfg, str):
                return cfg
            else:
                raise ValueError(f"split_list_json must be str or dict, got {type(cfg)}")
        return self._default_split_list_path()

    def _identifiers_from_json_payload(self, data):
        if isinstance(data, list):
            return list(data)
        if not isinstance(data, dict):
            raise ValueError(f"Split list json must be a list or dict, got {type(data)}")
        key = _PHASE_MANIFEST_KEYS.get(self.phase)
        if key and key in data:
            return list(data[key])
        alt_key = f"{self.phase}_identifiers"
        if alt_key in data:
            return list(data[alt_key])
        nested = data.get(f"{self.phase}_level_split") or data.get("sample_level_split")
        if isinstance(nested, dict):
            ids = nested.get(f"{self.phase}_identifiers")
            if ids is not None:
                return list(ids)
        raise ValueError(
            f"Cannot find identifiers for phase={self.phase} in split list json"
        )

    def _load_identifiers_from_json(self, json_path):
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        identifiers = self._identifiers_from_json_payload(data)
        validated = []
        missing = []
        for ident in identifiers:
            drr = os.path.join(self.data_path, f"{ident}-DRR.mhd")
            dsa = os.path.join(self.data_path, f"{ident}-DSA.mhd")
            if os.path.isfile(drr) and os.path.isfile(dsa):
                validated.append(ident)
            else:
                missing.append(ident)
        if missing:
            print(
                f"[TipsDataset2D:{self.phase}] skip {len(missing)} identifiers "
                f"missing DRR/DSA under {self.data_path}: {missing[:3]}"
                + (" ..." if len(missing) > 3 else "")
            )
        return validated

    def _scan_identifiers_from_dir(self):
        identifiers = []
        for file in sorted(os.listdir(self.data_path)):
            if file.endswith(_DRR_SUFFIX):
                identifiers.append(identifier_from_drr_filename(file))
        return identifiers

    def get_identifier_list(self):
        list_path = self._resolve_split_list_path()
        if list_path is not None:
            self.identifier_list = self._load_identifiers_from_json(list_path)
            print(
                f"[TipsDataset2D:{self.phase}] loaded {len(self.identifier_list)} "
                f"identifiers from {list_path}"
            )
        else:
            self.identifier_list = self._scan_identifiers_from_dir()
            print(
                f"[TipsDataset2D:{self.phase}] scanned {len(self.identifier_list)} "
                f"identifiers from {self.data_path}"
            )
        if self.max_num_for_loading > 0:
            self.identifier_list = self.identifier_list[: self.max_num_for_loading]

    def _init_projector_if_needed(self):
        if not hasattr(self, "_siddon") or self._siddon is None:
            self._siddon = Siddon(voxel_shift=0.0)
        if not hasattr(self, "_detector") or self._detector is None:
            self._detector = Detector(
                1020.0,
                self.detector_width,
                self.detector_width,
                self.detector_spacing,
                self.detector_spacing,
                0,
                0,
                self.reorient,
                reverse_x_axis=True,
                n_subsample=None,
            )

    def _render_majority_label_vectorized(
        self,
        mask: torch.Tensor,
        target_poses,
        affine_inv,
        max_samples: int = 1024,
        num_labels: int | None = None,
    ) -> torch.Tensor:
        """
        沿每条射线采样，返回“出现次数最多”的非零标签（多数投票）。
        mask: (D,H,W) or (B,D,H,W) or (B,1,D,H,W), 标签值 0..K，0 为背景
        source/target: (B,N,3) 世界坐标（会在函数内转体素）
        affine_inv: RigidTransform(affine.inverse())
        return: (B,N) 每条射线的 majority 非零标签；若全背景则返回 0
        """
        import torch.nn.functional as F

        # 0) 统一 mask 形状到 (B,1,D,H,W)
        if mask.dim() == 3:
            D, H, W = mask.shape
            mask_b = mask[None, None]
        elif mask.dim() == 4:
            Bm, D, H, W = mask.shape
            mask_b = mask[:, None]
        elif mask.dim() == 5:
            Bm, Cm, D, H, W = mask.shape
            mask_b = mask[:, :1]
        else:
            raise ValueError(f"Unexpected mask shape: {mask.shape}")
            
        poses = RigidTransform(target_poses)
        source, target = self._detector(poses, calibration=None)

        B, N, _ = target.shape
        if mask_b.shape[0] == 1 and B > 1:
            mask_b = mask_b.expand(B, -1, -1, -1, -1)
        elif mask_b.shape[0] != B:
            raise ValueError(f"mask batch {mask_b.shape[0]} != target batch {B}")
        # 1) 世界坐标 -> 体素坐标
        affine_inv = RigidTransform(affine_inv)
        source_voxel = affine_inv(source)  # (B,N,3)
        target_voxel = affine_inv(target)  # (B,N,3)
        ray_dir = source_voxel - target_voxel

        # 2) 沿射线均匀采样 points: (B,N,S,3)
        t = torch.linspace(0, 1, max_samples, device=mask_b.device, dtype=mask_b.dtype)
        t = t.view(1, 1, max_samples, 1)
        points = target_voxel.unsqueeze(2) + t * ray_dir.unsqueeze(2)

        # 3) 体素坐标归一化到 grid_sample 的 [-1,1]
        # 这里沿用你当前假设：points[...,0/1/2] 对应 D/H/W
        d = points[..., 0]
        h = points[..., 1]
        w = points[..., 2]


        d_norm = 2.0 * d / (D - 1) - 1.0
        h_norm = 2.0 * h / (H - 1) - 1.0
        w_norm = 2.0 * w / (W - 1) - 1.0

        # grid 最后一维必须是 (x,y,z)=(W,H,D)
        grid = torch.stack([w_norm, h_norm, d_norm], dim=-1)  # (B,N,S,3)
        grid = grid.permute(0, 2, 1, 3).contiguous().view(B, max_samples, 1, N, 3)

        # 4) 采样标签： (B,N,S)
        sampled = F.grid_sample(
            input=mask_b,
            grid=grid,
            mode="nearest",
            padding_mode="zeros",
            align_corners=True,
        )  # (B,1,S,1,N)

        sampled = sampled[:, 0, :, 0, :]            # (B,S,N)
        sampled = sampled.permute(0, 2, 1).long()   # (B,N,S)

        # 5) 多数投票（忽略背景 0）
        if num_labels is None:
            # +1 防止 max=0 时 one_hot 维度为0
            num_labels = int(sampled.max().item()) + 1
        num_labels = max(num_labels, 1)

        # one_hot: (B,N,S,C) -> 按 S 求和得到每条射线每个标签出现次数
        counts = F.one_hot(sampled.clamp(min=0), num_classes=num_labels).sum(dim=2)  # (B,N,C)

        # 忽略背景标签 0
        if num_labels > 1:
            counts[..., 0] = 0

        majority = counts.argmax(dim=-1)  # (B,N)

        # 若一条射线全为0，强制返回0
        has_foreground = (counts.sum(dim=-1) > 0)
        majority = torch.where(has_foreground, majority, torch.zeros_like(majority))

        return majority.to(mask_b.dtype).view(
            1, -1, self._detector.height, self._detector.width)
        
    def _run_renderer(self, density, target_poses, affine_inverse):
        poses = RigidTransform(target_poses)
        src, tgt = self._detector(poses, calibration=None)
        img_input = (tgt - src).norm(dim=-1).unsqueeze(1)
        src = RigidTransform(affine_inverse)(src)
        tgt = RigidTransform(affine_inverse)(tgt)
        return self._siddon(density, src, tgt, img_input).view(
            1, -1, self._detector.height, self._detector.width)

    def _preprocess_xray(self, img, crop, linearize, mask=None):
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
        return img

    def _load_data(self, identifier):
        source_proj = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(self.data_path, f"{identifier}-DRR.mhd")))
        target_proj = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(self.data_path, f"{identifier}-DSA.mhd")))
        def get_mask(proj):
            mask = proj > 0
            from scipy.ndimage import binary_erosion
            k = 5  # 想收几像素就设几
            mask = binary_erosion(mask, iterations=k)
            return mask
        mask = get_mask(target_proj)
        if os.path.exists(os.path.join(self.data_path, f"{identifier}-MASK.mhd")):
            source_mask = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(self.data_path, f"{identifier}-MASK.mhd")))
            source_mask = source_mask.astype(np.uint8)
            source_mask_mask = np.zeros_like(source_mask, dtype=np.float32)
            source_mask_mask[mask] = source_mask[mask]
            source_mask = source_mask_mask
        else:
            source_mask = None
        source_proj = self._preprocess_xray(source_proj, crop=0, linearize=False, mask=mask)
        target_proj = self._preprocess_xray(target_proj, crop=0, linearize=True, mask=mask)
        return source_proj, target_proj, source_mask

    def _load_warp_pose(self, identifier):
        rot_path = os.path.join(self.warp_path, f'{identifier.replace("drr", "poses_rot_")}.pt')
        xyz_path = os.path.join(self.warp_path, f'{identifier.replace("drr", "poses_xyz_")}.pt')
        return {
            "poses_rot": torch.load(rot_path, map_location="cpu", weights_only=True),
            "poses_xyz": torch.load(xyz_path, map_location="cpu", weights_only=True),
        }

    def _cache_is_valid(self, npz_path):
        if not (os.path.exists(npz_path)):
            return False

        try:
            with np.load(npz_path) as npz:
                required_keys = {"cache_version", "source_proj", "target_proj", "spacing"}
                if self.has_label:
                    required_keys.add("source_seg_proj")
                if self.offline_aug_variants > 0:
                    required_keys.update({"aug_source_proj", "aug_target_proj", "aug_affines"})
                    if self.has_label:
                        required_keys.add("aug_source_seg_proj")
                if not required_keys.issubset(set(npz.files)):
                    return False

                cache_version = int(np.asarray(npz["cache_version"]).reshape(-1)[0])
                if cache_version != self.cache_version:
                    return False

                if tuple(npz["source_proj"].shape[-2:]) != self.img_after_resize:
                    return False

                if self.offline_aug_variants > 0:
                    if npz["aug_source_proj"].shape[0] != self.offline_aug_variants:
                        return False
                    if npz["aug_target_proj"].shape[0] != self.offline_aug_variants:
                        return False
                    if npz["aug_affines"].shape[0] != self.offline_aug_variants:
                        return False
        except Exception:
            return False

        return True

    def _normalize_projection_stack(self, proj):
        proj = np.ascontiguousarray(proj.astype(np.float32))
        proj_min = float(proj.min())
        proj_max = float(proj.max())
        return (proj - proj_min) / (proj_max - proj_min + 1e-6)

    def _resize_projection_stack(self, proj, mode):
        tensor = torch.from_numpy(np.ascontiguousarray(proj.astype(np.float32))).unsqueeze(0).unsqueeze(0)
        kwargs = {"align_corners": False} if mode == "bilinear" else {}
        resized = F.interpolate(
            tensor,
            size=self.img_after_resize,
            mode=mode,
            **kwargs,
        )
        return resized.squeeze(0).numpy().astype(np.float32)

    def _prepare_projection_stack(self, proj):
        return np.ascontiguousarray(
            # self._resize_projection_stack(self._normalize_projection_stack(proj), mode="bilinear")
            self._resize_projection_stack(proj, mode="bilinear")
        )

    def _prepare_label_projection_stack(self, proj):
        proj = (np.ascontiguousarray(proj.astype(np.float32)).astype(np.float32))
        return np.ascontiguousarray(self._resize_projection_stack(proj, mode="nearest"))

    def _build_offline_augmented_bank(
        self,
        identifier,
        source_proj_np,
        target_proj_np,
        source_label_np=None,
    ):
        if self.offline_aug_variants <= 0:
            return None

        source_proj_np = np.ascontiguousarray(source_proj_np.astype(np.float32))
        target_proj_np = np.ascontiguousarray(target_proj_np.astype(np.float32))
        if source_label_np is not None:
            source_label_np = np.ascontiguousarray(source_label_np.astype(np.float32))

        if source_proj_np.shape != target_proj_np.shape:
            raise ValueError(
                f"Source/target projection shape mismatch for {identifier}: "
                f"{source_proj_np.shape} vs {target_proj_np.shape}"
            )
        if source_proj_np.ndim != 3:
            raise ValueError(
                f"Expected [P, H, W] projection stack for {identifier}, got {source_proj_np.shape}"
            )
        if source_label_np is not None and source_label_np.shape != source_proj_np.shape:
            raise ValueError(
                f"source_label shape {source_label_np.shape} != source_proj shape "
                f"{source_proj_np.shape} for {identifier}"
            )

        rotate_ranges = self._resolve_axis_ranges(self.aug_rotate_deg, symmetric=True)
        translate_ranges = self._resolve_axis_ranges(self.aug_translate_vox, symmetric=True)
        scale_min, scale_max = self._resolve_scale_range(self.aug_scale_range)

        def _apply_shared_2d_affine(src_np, tgt_np, theta_2x3, lab_np=None):
            src_t = torch.from_numpy(src_np).unsqueeze(0).to(dtype=torch.float32)
            tgt_t = torch.from_numpy(tgt_np).unsqueeze(0).to(dtype=torch.float32)
            theta = torch.from_numpy(theta_2x3).unsqueeze(0).to(dtype=torch.float32)
            grid = F.affine_grid(theta, size=src_t.shape, align_corners=False)
            src_aug = F.grid_sample(
                src_t, grid, mode="bilinear", padding_mode="border", align_corners=False
            )
            tgt_aug = F.grid_sample(
                tgt_t, grid, mode="bilinear", padding_mode="border", align_corners=False
            )
            lab_aug_np = None
            if lab_np is not None:
                lab_t = torch.from_numpy(lab_np).unsqueeze(0).to(dtype=torch.float32)
                lab_aug = F.grid_sample(
                    lab_t, grid, mode="nearest", padding_mode="zeros", align_corners=False
                )
                lab_aug_np = lab_aug.squeeze(0).cpu().numpy().astype(np.float32)
            return (
                src_aug.squeeze(0).cpu().numpy().astype(np.float32),
                tgt_aug.squeeze(0).cpu().numpy().astype(np.float32),
                lab_aug_np,
            )

        p, h, w = source_proj_np.shape
        _ = p

        angle_deg = float(np.random.uniform(rotate_ranges[2][0], rotate_ranges[2][1]))
        angle_rad = np.deg2rad(angle_deg)
        scale = float(np.random.uniform(scale_min, scale_max))
        tx_pix = float(np.random.uniform(translate_ranges[0][0], translate_ranges[0][1]))
        ty_pix = float(np.random.uniform(translate_ranges[1][0], translate_ranges[1][1]))

        tx_norm = 2.0 * tx_pix / max(float(w), 1.0)
        ty_norm = 2.0 * ty_pix / max(float(h), 1.0)

        cos_a = float(np.cos(angle_rad))
        sin_a = float(np.sin(angle_rad))
        theta_2x3 = np.array(
            [
                [scale * cos_a, -scale * sin_a, tx_norm],
                [scale * sin_a, scale * cos_a, ty_norm],
            ],
            dtype=np.float32,
        )

        source_proj_aug, target_proj_aug, source_label_aug = _apply_shared_2d_affine(
            source_proj_np, target_proj_np, theta_2x3, lab_np=source_label_np,
        )
        return source_proj_aug, target_proj_aug, source_label_aug

    def _preprocess_and_cache(self, identifier):
        npz_path = os.path.join(self.cache_dir, f"{identifier}.npz")
        if self._cache_is_valid(npz_path):
            return

        self._init_projector_if_needed()
        source_proj, target_proj, source_mask = self._load_data(identifier)
        target_proj_np = self._prepare_projection_stack(target_proj.astype(np.float32))
        if source_mask is not None:
            source_mask_np = self._prepare_label_projection_stack(source_mask)
        else:
            source_mask_np = None
        source_proj_np = self._prepare_projection_stack(
            source_proj.astype(np.float32)
        )

        save_dict = {
            "cache_version": np.array([self.cache_version], dtype=np.int16),
            "source_proj": source_proj_np,
            "target_proj": target_proj_np,
            "spacing": np.array(self.spacing, dtype=np.float32),
            "source_mask": source_mask_np,
        }

        np.savez_compressed(npz_path, **save_dict)


    def init_img_pool(self):
        print(f"[{self.phase}] Checking / building cache in {self.cache_dir}")
        pbar = pb.ProgressBar(
            widgets=[pb.Percentage(), pb.Bar(), pb.ETA()],
            maxval=len(self.identifier_list)).start()

        for i, identifier in enumerate(self.identifier_list):
            self._preprocess_and_cache(identifier)
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
        raise ValueError(f"Unsupported axis range config: {value}")

    def _resolve_scale_range(self, value):
        arr = np.asarray(value, dtype=np.float32)
        if arr.ndim == 0:
            delta = float(arr)
            return 1.0 - delta, 1.0 + delta
        if arr.size == 2:
            return float(arr[0]), float(arr[1])
        raise ValueError(f"Unsupported scale range config: {value}")

    def _normalized_to_voxel_h(self, vol_shape):
        d_size, h_size, w_size = [int(x) for x in vol_shape]
        mat = torch.eye(4, dtype=torch.float32)
        if w_size > 1:
            mat[0, 0] = (w_size - 1) / 2.0
            mat[0, 3] = (w_size - 1) / 2.0
        else:
            mat[0, 0] = 0.0
            mat[0, 3] = 0.0
        if h_size > 1:
            mat[1, 1] = (h_size - 1) / 2.0
            mat[1, 3] = (h_size - 1) / 2.0
        else:
            mat[1, 1] = 0.0
            mat[1, 3] = 0.0
        if d_size > 1:
            mat[2, 2] = (d_size - 1) / 2.0
            mat[2, 3] = (d_size - 1) / 2.0
        else:
            mat[2, 2] = 0.0
            mat[2, 3] = 0.0
        return mat

    def _voxel_to_normalized_h(self, vol_shape):
        d_size, h_size, w_size = [int(x) for x in vol_shape]
        mat = torch.eye(4, dtype=torch.float32)
        if w_size > 1:
            mat[0, 0] = 2.0 / (w_size - 1)
            mat[0, 3] = -1.0
        else:
            mat[0, 0] = 0.0
            mat[0, 3] = 0.0
        if h_size > 1:
            mat[1, 1] = 2.0 / (h_size - 1)
            mat[1, 3] = -1.0
        else:
            mat[1, 1] = 0.0
            mat[1, 3] = 0.0
        if d_size > 1:
            mat[2, 2] = 2.0 / (d_size - 1)
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
            [1.0, 0.0, 0.0, 0.0],
            [0.0, cosx, -sinx, 0.0],
            [0.0, sinx, cosx, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ], dtype=np.float32)
        rot_y = np.array([
            [cosy, 0.0, siny, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [-siny, 0.0, cosy, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ], dtype=np.float32)
        rot_z = np.array([
            [cosz, -sinz, 0.0, 0.0],
            [sinz, cosz, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
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

    def _apply_affine_grid(self, arr, grid, mode="bilinear", padding_mode="border"):
        if arr.ndim == 3:
            tensor = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)
        elif arr.ndim == 4:
            tensor = torch.from_numpy(arr).unsqueeze(0)
        else:
            raise ValueError(f"Expected 3D or 4D array, got shape {arr.shape}")
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
            stacked, grid, mode="bilinear", padding_mode="border")
        source_aug, density_aug = stacked[0], stacked[1]

        source_seg_aug = source_seg
        if source_seg is not None:
            source_seg_aug = self._apply_affine_grid(
                source_seg, grid, mode="nearest", padding_mode="zeros")

        weights_aug = weights
        if weights is not None:
            weights_aug = self._apply_affine_grid(
                weights, grid, mode="nearest", padding_mode="zeros")

        new_affine = affine @ voxel_to_input
        return source_aug, density_aug, source_seg_aug, weights_aug, new_affine, grid
        
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

    def _augment_2d_proj(self, proj):
        """Single-stack wrapper; uses shared helper with one input."""
        augmented, tag = self._augment_2d_projs_shared(proj)
        return augmented, tag

    def __getitem__(self, idx):
        idx = idx % len(self.identifier_list)
        identifier = self.identifier_list[idx]

        with np.load(os.path.join(self.cache_dir, f"{identifier}.npz")) as npz:
            source_proj = np.ascontiguousarray(npz["source_proj"].astype(np.float32))
            target_proj = np.ascontiguousarray(npz["target_proj"].astype(np.float32))
            spacing = np.ascontiguousarray(npz["spacing"].astype(np.float32))
            source_mask = np.ascontiguousarray(npz["source_mask"].astype(np.uint8))

        if self.enable_aug:
            label_for_aug = source_mask if self.has_label else None
            source_proj, target_proj, label_aug = self._build_offline_augmented_bank(
                identifier,
                source_proj,
                target_proj,
                source_label_np=label_for_aug,
            )
            source_proj, target_proj, _ = self._augment_2d_projs_shared(
                source_proj, target_proj)
            if label_aug is not None:
                # Cast back to uint8 to keep label semantics; nearest sampling
                # preserves integer values so rounding is safe.
                source_mask = np.ascontiguousarray(
                    np.rint(label_aug).clip(0, 255).astype(np.uint8)
                )

        sample = {
            "source_proj": source_proj,
            "target_proj": target_proj,
            "source_label": source_mask,
            "spacing": spacing,
        }

        if self.provide_weights:
            lab2d = source_mask
            if lab2d.ndim == 3:
                lab2d = lab2d[0]
            try:
                _, weights = compute_weights_2d(
                    lab2d,
                    labels=self.label_groups,
                    weightfn=self.weightfn,
                    normalize=True,
                )
            except ValueError:
                # Empty / all-zero label fallback: uniform weights across K segments
                weights = torch.full(
                    (max(self.num_segments, 1), int(lab2d.shape[-2]), int(lab2d.shape[-1])),
                    1.0 / max(self.num_segments, 1),
                    dtype=torch.float32,
                )
            weights = self._pad_or_truncate_weights(weights, self.num_segments)
            sample["weights"] = weights.numpy().astype(np.float32)

        if self.transform:
            sample["source_proj"] = self.transform(sample["source_proj"])
            sample["source_label"] = self.transform(sample["source_label"])
        return sample, identifier

    @staticmethod
    def _pad_or_truncate_weights(w: torch.Tensor, K: int) -> torch.Tensor:
        """Match the channel count of ``w`` to ``K`` via zero-pad or truncation."""
        k_case = int(w.shape[0])
        if k_case == K:
            return w
        if k_case < K:
            pad = torch.zeros((K - k_case, *w.shape[1:]), dtype=w.dtype)
            return torch.cat([w, pad], dim=0)
        return w[:K]

    def __len__(self):
        return len(self.identifier_list)

    def transform_hu_to_density(self, volume, bone_attenuation_multiplier):
        volume = volume.to(torch.float32)
        air = torch.where(volume <= -800)
        soft_tissue = torch.where((-800 < volume) & (volume <= 350))
        bone = torch.where(350 < volume)
        density = torch.empty_like(volume)
        density[air] = volume[soft_tissue].min()
        density[soft_tissue] = volume[soft_tissue]
        density[bone] = volume[bone] * bone_attenuation_multiplier
        density -= density.min()
        density /= density.max()
        return density

    def canonicalize(self, volume, is_label=False):
        """Move the Subject's isocenter to the origin in world coordinates"""
        isocenter = volume.get_center()
        tinv = np.array([
            [1, 0, 0, -isocenter[0]],
            [0, 1, 0, -isocenter[1]],
            [0, 0, 1, -isocenter[2]],
            [0, 0, 0, 1]], dtype=np.float64)
        if is_label:
            return LabelMap(tensor=volume.data, affine=tinv.dot(volume.affine))
        return ScalarImage(tensor=volume.data, affine=tinv.dot(volume.affine))
