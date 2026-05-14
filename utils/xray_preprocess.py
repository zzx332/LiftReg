from xvr.dicom import read_xray
import numpy as np
import torch
import torch.nn.functional as F
import SimpleITK as sitk

def _prepare_projection_stack(proj):
    return np.ascontiguousarray(
        _resize_projection_stack(_normalize_projection_stack(proj), mode="bilinear")
    )
def _normalize_projection_stack( proj):
    proj = np.ascontiguousarray(proj.astype(np.float32))
    proj_min = float(proj.min())
    proj_max = float(proj.max())
    return (proj - proj_min) / (proj_max - proj_min + 1e-6)

def _resize_projection_stack( proj, mode):
    tensor = torch.from_numpy(np.ascontiguousarray(proj.astype(np.float32))).unsqueeze(0)
    kwargs = {"align_corners": False} if mode == "bilinear" else {}
    resized = F.interpolate(
        tensor,
        size=(160, 160),
        mode=mode,
        **kwargs,
    )
    return resized.squeeze(0).numpy().astype(np.float32)
xray, *_ = read_xray("/home/zzx/data/deepfluoro/subject06/xrays/022.dcm", crop=100, linearize=False)
target_proj_np = _prepare_projection_stack(xray.squeeze(0).numpy().astype(np.float32))
sitk.WriteImage(sitk.GetImageFromArray(target_proj_np), "/home/zzx/data/deepfluoro/subject06/xray022.nii.gz")