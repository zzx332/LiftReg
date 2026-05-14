import os
import numpy as np
import SimpleITK as sitk
import torch
import torch.nn.functional as F
from scipy.spatial.transform import Rotation


def rigid_from_6params(rx_deg, ry_deg, rz_deg, tx, ty, tz):
    rm = Rotation.from_euler("xyz", [rx_deg, ry_deg, rz_deg], degrees=True).as_matrix()
    t = np.eye(4, dtype=np.float32)
    t[:3, :3] = rm.astype(np.float32)
    t[:3, 3] = np.array([tx, ty, tz], dtype=np.float32)
    return t


def sitk_affine_ijk_to_world(img):
    spacing = np.asarray(img.GetSpacing(), dtype=np.float32)  # (sx, sy, sz)
    direction = np.asarray(img.GetDirection(), dtype=np.float32).reshape(3, 3)
    origin = np.asarray(img.GetOrigin(), dtype=np.float32)
    a = np.eye(4, dtype=np.float32)
    a[:3, :3] = direction @ np.diag(spacing)
    a[:3, 3] = origin
    return a


def voxel_to_norm_h(d, h, w):
    m = np.eye(4, dtype=np.float32)
    m[0, 0] = 2.0 / max(w - 1, 1)
    m[0, 3] = -1.0
    m[1, 1] = 2.0 / max(h - 1, 1)
    m[1, 3] = -1.0
    m[2, 2] = 2.0 / max(d - 1, 1)
    m[2, 3] = -1.0
    return m


def norm_to_voxel_h(d, h, w):
    m = np.eye(4, dtype=np.float32)
    m[0, 0] = max(w - 1, 1) / 2.0
    m[0, 3] = max(w - 1, 1) / 2.0
    m[1, 1] = max(h - 1, 1) / 2.0
    m[1, 3] = max(h - 1, 1) / 2.0
    m[2, 2] = max(d - 1, 1) / 2.0
    m[2, 3] = max(d - 1, 1) / 2.0
    return m


data_path = "/home/zzx/data/pair_CT_DSA_10/cuihaiping/CT/13________1.5mm"
ct_img = sitk.ReadImage(os.path.join(data_path, "redata.mhd"))
ct_arr = sitk.GetArrayFromImage(ct_img).astype(np.float32)  # [D, H, W]

# CT -> intra-op DSA rigid transform in physical/world coordinates (mm).
t_user = rigid_from_6params(
-1.98277,1.90433,0.041432,-19.9544,-14.1942,-62.2494
)

d, h, w = ct_arr.shape
a_ijk2world = sitk_affine_ijk_to_world(ct_img)
a_world2ijk = np.linalg.inv(a_ijk2world).astype(np.float32)

# Rotate around physical center, then apply user translation.
c_phys = np.asarray(
    ct_img.TransformContinuousIndexToPhysicalPoint(((w - 1) / 2.0, (h - 1) / 2.0, (d - 1) / 2.0)),
    dtype=np.float32,
)
t_to_center = np.eye(4, dtype=np.float32)
t_to_center[:3, 3] = -c_phys
t_back = np.eye(4, dtype=np.float32)
t_back[:3, 3] = c_phys
t_world = t_back @ t_user @ t_to_center

# grid_sample uses inverse mapping: out -> in
m_ijk = a_world2ijk @ np.linalg.inv(t_world).astype(np.float32) @ a_ijk2world

v2n = voxel_to_norm_h(d, h, w)
n2v = norm_to_voxel_h(d, h, w)
theta_h = v2n @ m_ijk @ n2v
theta = torch.from_numpy(theta_h[:3, :]).unsqueeze(0)

ct_t = torch.from_numpy(ct_arr).unsqueeze(0).unsqueeze(0)  # [1,1,D,H,W]
grid = F.affine_grid(theta, size=ct_t.shape, align_corners=True)
ct_warp = F.grid_sample(
    ct_t,
    grid,
    mode="bilinear",
    padding_mode="border",
    align_corners=True,
)

ct_arr_warp = ct_warp.squeeze(0).squeeze(0).cpu().numpy().astype(np.float32)
ct_img_warp = sitk.GetImageFromArray(ct_arr_warp)
ct_img_warp.CopyInformation(ct_img)
sitk.WriteImage(ct_img_warp, os.path.join(data_path, "data_affined.mhd"))