"""
Project 3D mask and fiducials onto 2D X-ray using the given pose.

Two tasks:
  1. Project fiducials (3D points) -> 2D pixel coordinates on the X-ray
  2. Project mask (3D volume) -> 2D DRR-style projection on the X-ray
"""

import torch
import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt
from xvr.dicom import read_xray
from diffdrr.pose import RigidTransform
from diffdrr.detector import Detector, make_intrinsic_matrix
from diffdrr.renderers import Siddon
from diffdrr.data import read
from torchio import ScalarImage, LabelMap, Subject

def save_drr_as_nifti(
    img_tensor,
    filepath,
    delx,
    dely, 
    x0,
    y0,
    is_mask=False,
):
    """
    灏� DRR 鍥惧儚鎴� mask 淇濆瓨涓� NIfTI (.nii.gz) 鏍煎紡銆�
    
    Args:
        img_tensor: torch.Tensor [H, W] - 鍗曞紶鍥惧儚
        filepath: str - 淇濆瓨璺緞锛堝 'drr.nii.gz'锛�
        drr_obj: DRR 瀵硅薄 - 鐢ㄤ簬鎻愬彇鍑犱綍鍙傛暟
        is_mask: bool - 鏄惁涓� mask
    """
    import torch
    import numpy as np
    import os
    from torchio import ScalarImage, LabelMap
    
    # 杞崲涓� numpy
    img_np = img_tensor.detach().cpu().numpy()
    
    # NIfTI 闇€瑕佽嚦灏� 3D锛屾坊鍔犳繁搴︾淮搴� [D=1, H, W]
    if img_np.ndim == 2:
        img_np = img_np[np.newaxis, ...]  # [1, H, W]
    
    # 娣诲姞閫氶亾缁村害鐢ㄤ簬 torchio [C=1, D=1, H, W]
    img_np = img_np[..., np.newaxis]
    
    # 杞崲涓� torch tensor
    img_data = torch.from_numpy(img_np).float()
    # 鏋勫缓 4x4 affine 鐭╅樀
    # 鏍煎紡: [[sx, 0, 0, ox],
    #        [0, sy, 0, oy],
    #        [0, 0, sz, oz],
    #        [0, 0, 0, 1]]
    affine = np.eye(4)
    affine[0, 0] = delx      # X 鏂瑰悜缂╂斁锛堝垪锛�
    affine[1, 1] = dely      # Y 鏂瑰悜缂╂斁锛堣锛�
    affine[2, 2] = 1.0       # Z 鏂瑰悜锛堟繁搴�=1锛屾棤缂╂斁锛�
    affine[0, 3] = x0        # X 鏂瑰悜鍋忕Щ
    affine[1, 3] = y0        # Y 鏂瑰悜鍋忕Щ
    affine[2, 3] = 0.0       # Z 鏂瑰悜鍋忕Щ
    
    # 鍒涘缓 torchio 鍥惧儚瀵硅薄
    if is_mask:
        img_obj = LabelMap(tensor=img_data.to(torch.uint8), affine=affine)
    else:
        img_obj = ScalarImage(tensor=img_data, affine=affine)
    
    # 淇濆瓨
    os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)
    img_obj.save(filepath)
    
    print(f"  鉁� Saved: {filepath} (shape: {img_np.shape}, spacing: [{delx:.3f}, {dely:.3f}, 1.0] mm)")
    
    return filepath

def render_majority_label_vectorized(
    mask: torch.Tensor,
    source: torch.Tensor,
    target: torch.Tensor,
    affine_inv,
    max_samples: int = 128,
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

    B, N, _ = target.shape
    if mask_b.shape[0] == 1 and B > 1:
        mask_b = mask_b.expand(B, -1, -1, -1, -1)
    elif mask_b.shape[0] != B:
        raise ValueError(f"mask batch {mask_b.shape[0]} != target batch {B}")

    # 1) 世界坐标 -> 体素坐标
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

    return majority.to(mask_b.dtype)

def render_first_label_vectorized(
    mask: torch.Tensor,
    source: torch.Tensor,
    target: torch.Tensor,
    max_samples: int = 128,
) -> torch.Tensor:
    """
    澶歭abel mask 鐨� first-hit锛堟渶鍓嶉潰闈�0鏍囩锛夋姇褰便€�
    mask: (D,H,W) or (B,D,H,W) or (B,1,D,H,W)  鍊间负 0..K
    source/target: (B,N,3)  锛堜竴鑸� target 鏄� detector 鍍忕礌鐐瑰湪涓栫晫鍧愭爣/浣撶礌鍧愭爣鐨勬槧灏勶級
    return: (B,N) 姣忔潯灏勭嚎鐨勭涓€涓潪0鏍囩锛堟壘涓嶅埌鍒�0锛�
    """
    import torch.nn.functional as F

    # -------- 0) 缁熶竴 mask 褰㈢姸鍒� (B,1,D,H,W) --------
    if mask.dim() == 3:
        D, H, W = mask.shape
        mask_b = mask[None, None]  # (1,1,D,H,W)
    elif mask.dim() == 4:
        # (B,D,H,W)
        Bm, D, H, W = mask.shape
        mask_b = mask[:, None]     # (B,1,D,H,W)
    elif mask.dim() == 5:
        # (B,1,D,H,W) 鎴� (B,C,D,H,W)锛涜繖閲屽彧鍙栫涓€閫氶亾
        Bm, Cm, D, H, W = mask.shape
        if Cm != 1:
            mask_b = mask[:, :1]
        else:
            mask_b = mask
    else:
        raise ValueError(f"Unexpected mask shape: {mask.shape}")

    # source/target: (B,N,3)
    B, N, _ = target.shape

    # 鑻� mask 鍙湁 batch=1锛屼絾 source/target 鏄� B>1锛屽氨 expand
    if mask_b.shape[0] == 1 and B > 1:
        mask_b = mask_b.expand(B, -1, -1, -1, -1)
    elif mask_b.shape[0] != B:
        raise ValueError(f"mask batch {mask_b.shape[0]} != target batch {B}")

    # -------- 1) 杞埌浣撶礌鍧愭爣锛堜綘宸叉湁锛�--------
    source_voxel = affine_inv(source)  # (B,N,3) 浣撶礌鍧愭爣绯�
    target_voxel = affine_inv(target)  # (B,N,3)

    # 灏勭嚎鏂瑰悜锛氫粠 target -> source
    ray_dir = source_voxel - target_voxel  # (B,N,3)

    # -------- 2) 閲囨牱鐐癸細points (B,N,S,3) --------
    t = torch.linspace(0, 1, max_samples, device=mask_b.device, dtype=mask_b.dtype)  # (S,)
    t = t.view(1, 1, max_samples, 1)  # (1,1,S,1)
    points = target_voxel.unsqueeze(2) + t * ray_dir.unsqueeze(2)  # (B,N,S,3)

    # -------- 3) 褰掍竴鍖栧埌 grid_sample 闇€瑕佺殑 [-1,1]锛屾敞鎰忛『搴� (x,y,z)=(W,H,D) --------
    # points[...,0] 鏄� D 杞�? 杩樻槸 x? 鍙栧喅浜庝綘 affine_inverse 鐨勫畾涔夛紒
    # 浣犲師浠ｇ爜鎸� (D,H,W) 鏉ュ綊涓€鍖� 0,1,2 涓変釜鍒嗛噺銆傝繖閲屼繚鐣欎綘鐨勫亣璁撅細
    # points[...,0] 瀵瑰簲 D锛宲oints[...,1] 瀵瑰簲 H锛宲oints[...,2] 瀵瑰簲 W銆�
    # 浣� grid_sample 鐨� grid 鏈€鍚庝竴缁撮『搴忔槸 (x,y,z) => (W,H,D)锛屾墍浠ヨ閲嶆帓锛�
    d = points[..., 0]
    h = points[..., 1]
    w = points[..., 2]

    d_norm = 2.0 * d / (D - 1) - 1.0
    h_norm = 2.0 * h / (H - 1) - 1.0
    w_norm = 2.0 * w / (W - 1) - 1.0

    # grid 鐨勬渶鍚庣淮蹇呴』鏄� (x,y,z) = (w_norm, h_norm, d_norm)
    grid = torch.stack([w_norm, h_norm, d_norm], dim=-1)  # (B,N,S,3)

    # reshape 鎴� (B, D_out=S, H_out=1, W_out=N, 3)
    grid = grid.permute(0, 2, 1, 3).contiguous().view(B, max_samples, 1, N, 3)

    # -------- 4) grid_sample锛氳緭鍑� (B,1,S,1,N) --------
    sampled = F.grid_sample(
        input=mask_b,           # (B,1,D,H,W)
        grid=grid,              # (B,S,1,N,3)
        mode="nearest",         # label 蹇呴』 nearest
        padding_mode="zeros",
        align_corners=True,
    )  # (B,1,S,1,N)

    sampled = sampled[:, 0, :, 0, :]  # (B,S,N)
    sampled = sampled.permute(0, 2, 1).contiguous()  # (B,N,S)

    # -------- 5) first-hit锛氭瘡鏉″皠绾垮彇绗竴涓潪0鏍囩 --------
    # sampled: (B,N,S), 鍊间负 0..K
    nonzero = sampled != 0  # bool (B,N,S)

    # 鎵惧埌 first index锛氱敤 argmax(绱) 鐨勬妧宸�
    # 鏂瑰紡锛氭妸闈為浂杞负0/1锛岀劧鍚庢壘绗竴涓�1鐨勪綅缃�
    # 鍏堝鐞嗏€滃叏涓�0鈥濈殑鎯呭喌
    idx = torch.arange(max_samples, device=sampled.device).view(1, 1, max_samples)  # (1,1,S)
    INF = max_samples + 1
    idx_masked = torch.where(nonzero, idx, torch.full_like(idx, INF))  # (B,N,S)
    first_idx = idx_masked.min(dim=2).values  # (B,N)

    # gather 瀵瑰簲鏍囩
    first_idx_clamped = first_idx.clamp(0, max_samples - 1).unsqueeze(2)  # (B,N,1)
    first_label = torch.gather(sampled, 2, first_idx_clamped).squeeze(2)  # (B,N)

    # 鍏ㄤ负0鐨勫皠绾跨疆0
    first_label = torch.where(first_idx < INF, first_label, torch.zeros_like(first_label))
    return first_label
# ============================================================
# 1. Load data
# ============================================================
data_path = "/home/zzx/data/deepfluoro/subject01"
xray_idx = 0

# Volume and mask
subject = read(f"{data_path}/volume.nii.gz", f"{data_path}/mask.nii.gz", orientation="PA")
volume = subject.volume
mask = subject.mask

# X-ray image
xray, *_ = read_xray(f"{data_path}/xrays/{xray_idx:03d}.dcm", crop=100)

# Pose (4x4 extrinsic matrix) and intrinsics
pose_data = torch.load(f"{data_path}/xrays/{xray_idx:03d}.pt", weights_only=False)
pose_matrix = pose_data["pose"]  # (1, 4, 4)
intrinsics = pose_data["intrinsics"]
# intrinsics: sdd=1020.0, delx=0.194, dely=0.194,
#             x0=-0.097, y0=-0.097, height=1436, width=1436

# Fiducials: 3D landmarks in world coordinates (1, 14, 3)
fiducials = torch.load(f"{data_path}/fiducials.pt", weights_only=False)

print(f"Pose shape: {pose_matrix.shape}")
print(f"Fiducials shape: {fiducials.shape}")
print(f"Volume shape: {volume.shape}, Mask shape: {mask.shape}")

# ============================================================
# 2. Setup detector (same as FluoroDataset)
# ============================================================
sdd = intrinsics["sdd"]       # 1020.0 mm
# delx = intrinsics["delx"]     # 0.194 mm/pixel
# dely = intrinsics["dely"]     # 0.194 mm/pixel
x0 = intrinsics["x0"]         # -0.097 mm
y0 = intrinsics["y0"]         # -0.097 mm
# x0 = 0         # -0.097 mm
# y0 = 0        # -0.097 mm
# height = intrinsics["height"] # 1436
# width = intrinsics["width"]   # 1436
delx = 0.7255
dely = 0.7255
height = 384
width = 384

# Reorient matrix (SAL->SPR, same as in FluoroDataset)
reorient = torch.tensor(
    [[1, 0, 0, 0],
     [0, 0, 1, 0],
     [0, 1, 0, 0],
     [0, 0, 0, 1]], dtype=torch.float32)

detector = Detector(
    sdd=sdd, height=height, width=width,
    delx=delx, dely=dely, x0=x0, y0=y0,
    reorient=reorient, reverse_x_axis=True, n_subsample=None,
)

# ============================================================
# 3. Project fiducials (3D points -> 2D pixels)
# ============================================================
# diffdrr convention: pose is camera-to-world (extrinsic)
# perspective_projection logic from diffdrr/drr.py:
#   extrinsic_inv = (reorient @ pose)^{-1}
#   p_cam = extrinsic_inv(pts_world)
#   p_proj = K @ p_cam
#   p_2d = p_proj[:2] / p_proj[2]
#   adjust for image origin (flip y and optionally x)

pose_rt = RigidTransform(pose_matrix)  # (1, 4, 4)
K = make_intrinsic_matrix(detector)    # (3, 3)

# Transform world points to camera frame
extrinsic = detector.reorient.compose(pose_rt)
extrinsic_inv = extrinsic.inverse()
pts_cam = extrinsic_inv(fiducials)  # (1, 14, 3)

# Perspective projection: K @ pts_cam, then divide by z
pts_proj = torch.einsum("ij, bnj -> bni", K, pts_cam)
z = pts_proj[..., -1:].clone()
pts_2d = pts_proj / z  # (1, 14, 3)

# Adjust for image conventions (origin at upper-left)
pts_2d[..., 1] = height - pts_2d[..., 1]
pts_2d[..., 0] = width - pts_2d[..., 0]  # reverse_x_axis=True

fiducials_2d = pts_2d[..., :2]  # (1, 14, 2) in pixel coords
print(f"\nProjected fiducials (2D pixel coords):\n{fiducials_2d[0]}")

# ============================================================
# 4. Project mask (3D volume -> 2D via DRR rendering)
# ============================================================
# Treat the mask as a volume and render a DRR using Siddon ray-casting
# This produces a "projected mask" on the detector plane

mask_density = subject.mask.data.to(torch.float32).squeeze()[None, None]

# Get affine from the volume (maps voxel -> world)
affine = torch.as_tensor(subject.volume.affine, dtype=torch.float32).unsqueeze(0)
affine_inv = RigidTransform(affine.inverse())

# Get source and target ray endpoints in world coords
src_pts, tgt_pts = detector(pose_rt, calibration=None)

# Render: ray-cast through the mask volume
# mask_proj = render_first_label_vectorized(mask_density, src_pts, tgt_pts)
mask_proj = render_majority_label_vectorized(mask_density, src_pts, tgt_pts, affine_inv, max_samples=1024)
mask_proj = mask_proj.view(-1, height, width)  # (1, 1, H, W)
mask_proj_np = mask_proj[0].detach().cpu().numpy()
mask_proj_np[mask_proj_np==7] = 0
print(f"\nMask projection shape: {mask_proj_np.shape}")
print(f"Mask projection range: [{mask_proj_np.min():.3f}, {mask_proj_np.max():.3f}]")
# sitk.WriteImage(sitk.GetImageFromArray(mask_proj_np), f"{data_path}/mask_projection.nii.gz")
# ============================================================
# 5. Visualization
# ============================================================
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# (a) X-ray with fiducials overlaid
fid_px = fiducials_2d[0].detach().cpu().numpy()
axes[0].scatter(fid_px[:, 0], fid_px[:, 1], c="red", s=40, marker="+", linewidths=1.5)
for i, (x, y) in enumerate(fid_px):
    axes[0].annotate(str(i), (x, y), color="yellow", fontsize=7, ha="center", va="bottom")
axes[0].set_title("X-ray + Fiducials")
axes[0].axis("off")

# (b) Mask DRR projection
axes[1].imshow(mask_proj_np, cmap="hot")
axes[1].set_title("Mask Projection (DRR)")
axes[1].axis("off")

# (c) Overlay
axes[2].imshow(mask_proj_np, cmap="hot", alpha=0.4)
axes[2].scatter(fid_px[:, 0], fid_px[:, 1], c="lime", s=40, marker="+", linewidths=1.5)
axes[2].set_title("Overlay")
axes[2].axis("off")

plt.tight_layout()
plt.savefig("/home/zzx/data/deepfluoro/subject01/projection_result.png", dpi=150)
print("\nSaved to projection_result.png")
