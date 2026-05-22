import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import SimpleITK as sitk
import torch
import torch.nn.functional as F
import torchio
from diffdrr.detector import Detector
from diffdrr.pose import RigidTransform
from diffdrr.renderers import Siddon
from torchio import LabelMap, ScalarImage
from scipy.spatial.transform import Rotation

DATA_PATH = "/home/zzx/data/pair_CT_DSA_10"
OUTPUT_DIR = os.path.join(DATA_PATH, "drr_generated")
TARGET_SPACING = (2.2, 2.2, 2.2)
TARGET_SIZE = (160, 160, 160)
# TARGET_SPACING = (1.1, 1.1, 1.1)
# TARGET_SIZE = (320, 320, 320)

# 与 FluoroDataset 保持一致
REORIENT = torch.tensor(
    [
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
    ],
     dtype=torch.float32)

def rigid_from_6params(rx_deg, ry_deg, rz_deg, tx, ty, tz):
    Rm = Rotation.from_euler("xyz", [rx_deg, ry_deg, rz_deg], degrees=True).as_matrix()
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = Rm
    T[:3, 3] = np.array([tx, ty, tz], dtype=np.float64)
    return T


def canonicalize(volume: ScalarImage, case_name: str) -> ScalarImage:
    """Move the volume's isocenter to the origin in world coordinates."""
    isocenter = volume.get_center()
    tinv = np.array([
        [1, 0, 0, -isocenter[0]],
        [0, 1, 0, -isocenter[1]],
        [0, 0, 1, -isocenter[2]],
        [0, 0, 0, 1],
    ], dtype=np.float64)
    # 你的参数（顺序若不对需改）
    affine_dict = {
        "chenchuanlai": (-2.50484,6.68063,0.821655,69.817,18.4138,-24.9237),
        "chengshijiu": (-1.45579,5.03901,0.431279,19.1936,-1.98783,-36.2865),
        "cuihaiping": (-1.98277,1.90433,0.041432,-19.9544,-14.1942,-62.2494),
        "liaijiao": (0.609258,-8.12951,3.02794,6.00707,32.0073,-142.665),
        "liufang": (-5.33246,1.23107,-0.252757,-12.7675,-2.59383,-39.9518),
        "liwanggui": (0.255239,-1.58393,0.540072,0.616907,24.1804,-50.1729),
        "lixiaofa": (1.42632,6.14718,-1.3745,33.0599,22.3215,75.9592),
        "meichuanyao": (1.35667,0.798763,0.0509796,22.5106,34.1446,79.8066),
        "shenlixin": (-0.538263,-4.0883,1.07946,57.2886,17.8118,-141.624),
        "shimingzhi": (-1.68855,7.32938,0.546483,20.4227,75.6821,-0.69562),
    }

    T_user_lps = rigid_from_6params(*affine_dict[case_name])
    # T_user = np.linalg.inv(T_user)
    # LPS → RAS 转换: 翻转 x, y 轴
    S = np.diag([-1.0, -1.0, 1.0, 1.0])
    T_user_ras = S @ T_user_lps @ S

    affine_new = T_user_ras @ tinv @ volume.affine
    # affine_new = tinv @ volume.affine
    # return ScalarImage(tensor=volume.data, affine=tinv.dot(volume.affine))
    return ScalarImage(tensor=volume.data, affine=affine_new)


def transform_hu_to_density(volume: torch.Tensor, bone_attenuation_multiplier: float = 2.0) -> torch.Tensor:
    """Same HU->density rule used in FluoroDataset."""
    volume = volume.to(torch.float32)
    density = torch.empty_like(volume)
    # air_threshold = -800
    air_threshold = 0
    soft_tissue_threshold = 350
    bone_threshold = 1200
    soft_tissue = torch.where((air_threshold < volume) & (volume <= soft_tissue_threshold))
    air = torch.where(volume <= air_threshold)
    bone = torch.where(volume > soft_tissue_threshold)
    ultra_bone = torch.where(volume > bone_threshold)
    density[air] = volume[soft_tissue].min()
    density[soft_tissue] = volume[soft_tissue]
    density[bone] = volume[bone] * bone_attenuation_multiplier
    # density[bone] = volume[soft_tissue].max()
    # density[ultra_bone] = volume[ultra_bone] * 2
    density -= density.min()
    density /= density.max()
    return density


def load_and_preprocess_ct(ct_mhd_path: str, case_name: str) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Returns:
        density_xyz: (X, Y, Z) float32 tensor
        affine_inv:  (4, 4) float32 tensor
    """
    source_img = ScalarImage(ct_mhd_path)
    source_img.data = source_img.data - 1024.0
    source_img.data[torch.where(source_img.data > 2048)] = 2048
    source_img = canonicalize(source_img, case_name)
    # tfm = torchio.transforms.Compose([
    #     torchio.transforms.Resample(TARGET_SPACING),
    #     torchio.transforms.CropOrPad(TARGET_SIZE, padding_mode=-2048),
    # ])
    # source_img = tfm(source_img)
    density = transform_hu_to_density(source_img.data, bone_attenuation_multiplier=5.0)
    density_xyz = density.squeeze(0).to(torch.float32).cpu()
    affine = torch.as_tensor(source_img.affine, dtype=torch.float32)
    return density_xyz, affine.inverse()


def load_ct_and_mask(
    ct_mhd_path: str,
    seg_path: Optional[str],
    case_name: str,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
    """
    Load CT and (optional) segmentation, applying the same canonicalize so they
    share the world-to-voxel affine.

    Returns:
        density_xyz: (X, Y, Z) float32 tensor
        mask_xyz:    (X, Y, Z) int16 tensor or None
        affine_inv:  (4, 4) float32 tensor
    """
    source_img = ScalarImage(ct_mhd_path)
    source_img.data = source_img.data - 1024.0
    source_img.data[torch.where(source_img.data > 2048)] = 2048
    source_img = canonicalize(source_img, case_name)
    canonical_affine = np.array(source_img.affine, dtype=np.float64).copy()

    density = transform_hu_to_density(source_img.data, bone_attenuation_multiplier=1.0)
    density_xyz = density.squeeze(0).to(torch.float32).cpu()
    affine_inv = torch.as_tensor(canonical_affine, dtype=torch.float32).inverse()

    mask_xyz: Optional[torch.Tensor] = None
    if seg_path is not None and os.path.isfile(seg_path):
        seg_img = LabelMap(seg_path)
        # CT and seg share identical raw geometry; reuse canonical affine directly.
        seg_img = LabelMap(tensor=seg_img.data, affine=canonical_affine)
        mask_xyz = seg_img.data.squeeze(0).to(torch.int16).cpu()
    return density_xyz, mask_xyz, affine_inv


def make_renderer(
    sdd: float,
    detector_width: int,
    detector_height: int,
    dx: float,
    dy: float,
) -> Tuple[Siddon, Detector]:
    siddon = Siddon(voxel_shift=0.0)
    detector = Detector(
        float(sdd),
        int(detector_width),
        int(detector_height),
        float(dx),
        float(dy),
        0,
        0,
        REORIENT,
        reverse_x_axis=True,
        n_subsample=None,
    )
    return siddon, detector


def render_drr(
    density_xyz: torch.Tensor,
    extrinsic: torch.Tensor,
    affine_inv: torch.Tensor,
    siddon: Siddon,
    detector: Detector,
) -> torch.Tensor:
    """Render one DRR from one extrinsic. Returns (1, H, W)."""
    pose = RigidTransform(extrinsic.unsqueeze(0).to(torch.float32))
    src, tgt = detector(pose, calibration=None)
    img_input = (tgt - src).norm(dim=-1).unsqueeze(1)
    src = RigidTransform(affine_inv)(src)
    tgt = RigidTransform(affine_inv)(tgt)
    proj = siddon(density_xyz, src, tgt, img_input).view(1, detector.height, detector.width)
    return proj.detach().cpu()


def render_mask_first_hit(
    mask_xyz: torch.Tensor,
    extrinsic: torch.Tensor,
    affine_inv: torch.Tensor,
    detector: Detector,
    max_samples: int = 256,
) -> torch.Tensor:
    """
    Render a first-hit (first non-zero label along each ray) projection of a
    multi-label mask, sharing geometry with `render_drr`.

    Args:
        mask_xyz:   (X, Y, Z) integer-valued label volume in CT voxel order.
        extrinsic:  (4, 4) cam->world pose (same as `render_drr`).
        affine_inv: (4, 4) world->voxel affine (same as `render_drr`).
        detector:   diffdrr Detector instance.
        max_samples: number of samples along each ray.

    Returns:
        (1, H, W) int16 tensor with first-hit label values (0 where empty).
    """
    pose = RigidTransform(extrinsic.unsqueeze(0).to(torch.float32))
    src, tgt = detector(pose, calibration=None)  # world coords, (1, N, 3)
    src_v = RigidTransform(affine_inv)(src)       # voxel coords
    tgt_v = RigidTransform(affine_inv)(tgt)

    # mask: (X,Y,Z) -> (1,1,Z,Y,X) so that grid_sample's (x,y,z) <-> (W=X, H=Y, D=Z)
    Xs, Ys, Zs = int(mask_xyz.shape[0]), int(mask_xyz.shape[1]), int(mask_xyz.shape[2])
    mask_5d = mask_xyz.to(torch.float32).permute(2, 1, 0)[None, None]

    B, N, _ = tgt_v.shape  # B=1, N = detector.height * detector.width
    ray_dir = src_v - tgt_v  # (B, N, 3)

    t = torch.linspace(0.0, 1.0, max_samples, device=mask_5d.device, dtype=mask_5d.dtype)
    t = t.view(1, 1, max_samples, 1)
    points = tgt_v.unsqueeze(2) + t * ray_dir.unsqueeze(2)  # (B, N, S, 3) voxel xyz

    x = points[..., 0]
    y = points[..., 1]
    z = points[..., 2]
    x_norm = 2.0 * x / max(Xs - 1, 1) - 1.0
    y_norm = 2.0 * y / max(Ys - 1, 1) - 1.0
    z_norm = 2.0 * z / max(Zs - 1, 1) - 1.0
    grid = torch.stack([x_norm, y_norm, z_norm], dim=-1)  # (B, N, S, 3)
    grid = grid.permute(0, 2, 1, 3).contiguous().view(B, max_samples, 1, N, 3)

    sampled = F.grid_sample(
        input=mask_5d,
        grid=grid,
        mode="nearest",
        padding_mode="zeros",
        align_corners=True,
    )  # (B, 1, S, 1, N)
    sampled = sampled[:, 0, :, 0, :].permute(0, 2, 1).contiguous()  # (B, N, S)

    nonzero = sampled != 0
    idx = torch.arange(max_samples, device=sampled.device).view(1, 1, max_samples)
    INF = max_samples + 1
    idx_masked = torch.where(nonzero, idx, torch.full_like(idx, INF))
    first_idx = idx_masked.min(dim=2).values  # (B, N)

    first_idx_clamped = first_idx.clamp(0, max_samples - 1).unsqueeze(2)
    first_label = torch.gather(sampled, 2, first_idx_clamped).squeeze(2)
    first_label = torch.where(first_idx < INF, first_label, torch.zeros_like(first_label))

    out = first_label.round().to(torch.int16).view(1, detector.height, detector.width)
    return out.detach().cpu()


def find_first_ct_mhd(case_dir: str) -> Optional[str]:
    ct_dir = os.path.join(case_dir, "CT")
    if not os.path.isdir(ct_dir):
        return None
    mhd_files = sorted(Path(ct_dir).glob("**/*.mhd"))
    # mhd_files = sorted(Path(ct_dir).glob("**/data_affined.mhd"))
    if len(mhd_files) == 0:
        return None
    return str(mhd_files[0])


def find_seg_nii(case_dir: str) -> Optional[str]:
    """Return path to <case>/Segmentation/seg.nii.gz if it exists."""
    seg_path = os.path.join(case_dir, "Segmentation", "vertebrae.nii.gz")
    if os.path.isfile(seg_path):
        return seg_path
    seg_dir = os.path.join(case_dir, "Segmentation")
    if os.path.isdir(seg_dir):
        candidates = sorted(Path(seg_dir).glob("**/seg.nii.gz"))
        if candidates:
            return str(candidates[0])
    return None


def sanitize_name(path_str: str) -> str:
    return path_str.replace("\\", "/").replace("/", "__").replace(" ", "_")


def _find_case_dsa_info_pt(case_dir: Path) -> Optional[Path]:
    """优先读取 case/DSA/dsa_mhd_info.pt；否则回退到 DSA 下任意同名文件。"""
    direct = case_dir / "DSA" / "dsa_mhd_info.pt"
    if direct.exists():
        return direct
    matches = sorted((case_dir / "DSA").glob("**/dsa_mhd_info.pt")) if (case_dir / "DSA").exists() else []
    return matches[0] if matches else None


def collect_dsa_records_from_cases(data_path: str) -> List[Dict]:
    """
    从每个 case 的 DSA/dsa_mhd_info.pt 读取渲染记录。
    记录里应包含 extrinsic 与 spacing/size 等几何参数。
    """
    records: List[Dict] = []
    root = Path(data_path)
    for case_dir in sorted(root.iterdir()):
        if not case_dir.is_dir():
            continue
        info_pt = _find_case_dsa_info_pt(case_dir)
        if info_pt is None:
            continue

        info_obj = torch.load(str(info_pt), map_location="cpu", weights_only=False)
        case_records = info_obj if isinstance(info_obj, list) else [info_obj]

        for rec in case_records:
            if not isinstance(rec, dict):
                continue
            merged = dict(rec)
            merged["case_name"] = merged.get("case_name", case_dir.name)
            if "mhd_path" not in merged:
                merged["mhd_path"] = str(info_pt)

            # 支持两种字段风格: dx/dy + detector_width/height，或 ElementSpacing + DimSize
            if ("dx" not in merged or "dy" not in merged) and "ElementSpacing" in merged:
                sp = merged["ElementSpacing"]
                vals = [float(x) for x in (sp.split() if isinstance(sp, str) else sp)]
                if len(vals) >= 2:
                    merged["dx"], merged["dy"] = vals[0], vals[1]
            if ("detector_width" not in merged or "detector_height" not in merged) and "DimSize" in merged:
                ds = merged["DimSize"]
                vals = [int(float(x)) for x in (ds.split() if isinstance(ds, str) else ds)]
                if len(vals) >= 2:
                    merged["detector_width"], merged["detector_height"] = vals[0], vals[1]

            records.append(merged)
    return records

def drr2fluro(drr_np):
    # 1) 归一化到非负，避免数值问题
    L = drr_np - drr_np.min()
    L = L / (L.max() + 1e-8)
    # 2) Beer-Lambert: 积分 -> 透过强度（更像X-ray）
    k = 3.0  # 可调: 1.5~6 常见
    I = np.exp(-k * L)
    # 3) 显示增强（两种二选一）
    # A: 直接用透过强度（骨更暗，空气更亮）
    drr_like = I
    # B: 医学里常见的 log 域（接近 -log(I/I0)）
    # drr_like = -np.log(I + 1e-6)
    # 4) 映射到 [0,1]
    drr_like = (drr_like - drr_like.min()) / (drr_like.max() - drr_like.min() + 1e-8)
    return drr_like

def main() -> None:
    # os.makedirs(OUTPUT_DIR, exist_ok=True)
    records: List[Dict] = collect_dsa_records_from_cases(DATA_PATH)
    generated_index: List[Dict] = []
    skipped: List[Dict] = []
    ct_cache: Dict[str, Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor]] = {}

    # Reuse detector/siddon for same geometry tuple.
    projector_cache: Dict[Tuple[float, int, int, float, float], Tuple[Siddon, Detector]] = {}
    output_path = "/home/zzx/data/tips_drr_org"
    os.makedirs(output_path, exist_ok=True)
    dsa_path = "/home/zzx/data/tips_drr/val"
    for rec in records:
        case_name = rec.get("case_name")
        if case_name != "shenlixin":
            continue
        extrinsic = rec.get("extrinsic")
        if not case_name or extrinsic is None:
            skipped.append({"case_name": case_name, "reason": "missing_case_or_extrinsic"})
            continue

        sdd = rec.get("DistanceSourceToDetector")
        dx = rec.get("dx")
        dy = rec.get("dy")
        detector_width = rec.get("detector_width")
        detector_height = rec.get("detector_height")
        new_detector_width = 512
        new_detector_height = 512
        new_dx = detector_width * dx / new_detector_width
        new_dy = detector_height * dy / new_detector_height
        if sdd is None or dx is None or dy is None or detector_width is None or detector_height is None:
            skipped.append({"case_name": case_name, "reason": "missing_geometry"})
            continue
        if float(dx) <= 0 or float(dy) <= 0 or int(detector_width) <= 0 or int(detector_height) <= 0:
            skipped.append({"case_name": case_name, "reason": "invalid_spacing_or_width"})
            continue

        case_dir = os.path.join(DATA_PATH, case_name)
        ct_mhd_path = find_first_ct_mhd(case_dir)
        if ct_mhd_path is None:
            skipped.append({"case_name": case_name, "reason": "missing_ct_mhd"})
            continue

        seg_path = find_seg_nii(case_dir)
        if case_name not in ct_cache:
            density_xyz, mask_xyz, affine_inv = load_ct_and_mask(ct_mhd_path, seg_path, case_name)
            ct_cache[case_name] = (density_xyz, mask_xyz, affine_inv)
        density_xyz, mask_xyz, affine_inv = ct_cache[case_name]

        # geom_key = (float(sdd), int(detector_width), int(detector_height), float(dx), float(dy))
        geom_key = (float(sdd), int(new_detector_width), int(new_detector_height), float(new_dx), float(new_dy))
        if geom_key not in projector_cache:
            projector_cache[geom_key] = make_renderer(*geom_key)
        siddon, detector = projector_cache[geom_key]

        extrinsic_world2cam = torch.as_tensor(extrinsic, dtype=torch.float32)
        extrinsic_cam2world = torch.inverse(extrinsic_world2cam)
        drr = render_drr(density_xyz, extrinsic_cam2world, affine_inv, siddon, detector)

        # case_out_dir = os.path.join(OUTPUT_DIR, case_name)
        # os.makedirs(case_out_dir, exist_ok=True)
        mhd_path = rec.get("mhd_path", f"{case_name}_unknown.mhd")
        save_name = sanitize_name(Path(mhd_path).stem) + ".nii.gz"
        # save_path = os.path.join(case_out_dir, save_name)

        drr_np = drr.numpy().astype(np.float32)  # (1, H, W)
        drr_np = np.flip(drr_np, axis=1)
        drr_np = drr_np[0, :, :]
        drr_img = sitk.GetImageFromArray(drr_np)
        def resize_sitk_2d_to(img, out_w=512, out_h=512, interpolator=sitk.sitkLinear):
            in_size = img.GetSize()       # (nx, ny)
            in_spacing = img.GetSpacing() # (sx, sy)
            out_spacing = (
                in_spacing[0] * in_size[0] / float(out_w),
                in_spacing[1] * in_size[1] / float(out_h),
            )
            ref = sitk.Image(int(out_w), int(out_h), img.GetPixelID())
            ref.SetOrigin(img.GetOrigin())
            ref.SetDirection(img.GetDirection())
            ref.SetSpacing(out_spacing)
            return sitk.Resample(img, ref, sitk.Transform(), interpolator, 0.0, img.GetPixelIDValue())
        dsa_img = sitk.ReadImage(os.path.join(dsa_path, case_name + "-DSA.mhd"))
        dsa_img = resize_sitk_2d_to(dsa_img, 512, 512)
        # drr_img.CopyInformation(dsa_img)
        drr_img.SetSpacing([new_dx, new_dy])
        # drr_np = drr2fluro(drr_np)
        sitk.WriteImage(drr_img, os.path.join(output_path, case_name + "-DRR.mhd"))
        sitk.WriteImage(dsa_img, os.path.join(output_path, case_name + "-DSA.mhd"))

        mask_out_path: Optional[str] = None
        unique_labels: List[int] = []
        if mask_xyz is not None:
            mask_proj = render_mask_first_hit(
                mask_xyz, extrinsic_cam2world, affine_inv, detector
            )  # (1, H, W) int16
            mask_np = mask_proj.numpy().astype(np.uint8)
            mask_np = np.flip(mask_np, axis=1)
            mask_np_2d = mask_np[0]
            mask_img = sitk.GetImageFromArray(mask_np_2d)
            mask_img.SetSpacing([new_dx, new_dy])
            mask_out_path = os.path.join(output_path, case_name + "-vertebrae_mask.mhd")
            sitk.WriteImage(mask_img, mask_out_path)
            unique_labels = [int(v) for v in np.unique(mask_np_2d).tolist()]
        else:
            skipped.append({"case_name": case_name, "reason": "missing_segmentation"})

        generated_index.append({
            "case_name": case_name,
            "ct_mhd_path": ct_mhd_path,
            "seg_path": seg_path,
            "input_mhd_path": mhd_path,
            # "output_drr_path": save_path,
            "output_mask_path": mask_out_path,
            "unique_labels": unique_labels,
            "shape": list(drr_np.shape),
            "min": float(drr_np.min()),
            "max": float(drr_np.max()),
            "dx": float(dx),
            "dy": float(dy),
            "detector_width": int(detector_width),
            "detector_height": int(detector_height),
            "sdd": float(sdd),
        })
        print(f"Generated DRR for {case_name}" + ("" if mask_out_path is None else " (+ MASK)"))
        # break

    index_path = os.path.join(OUTPUT_DIR, "generated_index.json")
    with open(index_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "data_path": DATA_PATH,
                "info_source": "per-case DSA/dsa_mhd_info.pt",
                "output_dir": OUTPUT_DIR,
                "num_total_records": len(records),
                "num_generated": len(generated_index),
                "num_skipped": len(skipped),
                "generated": generated_index,
                "skipped": skipped,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(f"[Done] Generated {len(generated_index)} DRRs")
    print(f"[Done] Skipped {len(skipped)} records")
    print(f"[Done] Index saved to: {index_path}")


if __name__ == "__main__":
    main()
