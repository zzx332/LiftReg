import os
import torch
import torch.nn.functional as F
import numpy as np
import SimpleITK as sitk
import pandas as pd
from scipy.ndimage import uniform_filter
from xvr.dicom import read_xray
from torchvision.transforms.functional import center_crop

def ssim(img1, img2, win_size=7, data_range=1.0, K1=0.01, K2=0.03):
    """Compute mean SSIM between two 2D images (numpy arrays, float32)."""
    C1 = (K1 * data_range) ** 2
    C2 = (K2 * data_range) ** 2

    # squeeze to 2D
    img1 = img1.squeeze().astype(np.float64)
    img2 = img2.squeeze().astype(np.float64)

    mu1 = uniform_filter(img1, size=win_size)
    mu2 = uniform_filter(img2, size=win_size)

    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = uniform_filter(img1 * img1, size=win_size) - mu1_sq
    sigma2_sq = uniform_filter(img2 * img2, size=win_size) - mu2_sq
    sigma12 = uniform_filter(img1 * img2, size=win_size) - mu1_mu2

    numerator = (2 * mu1_mu2 + C1) * (2 * sigma12 + C2)
    denominator = (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)

    ssim_map = numerator / denominator
    return float(np.mean(ssim_map))

def dice(pred, gt, label=[1,2,3,4,5,6]):
    dice_scores = {}
    for label_id in label:
        fixed_mask = (gt == label_id)
        moving_mask = (pred == label_id)
        
        intersection = np.sum(fixed_mask & moving_mask)
        union = np.sum(fixed_mask) + np.sum(moving_mask)
        
        if union > 0:
            d = 2.0 * intersection / union
        else:
            d = 0.0
        
        dice_scores[label_id] = d
    return dice_scores

def normalize_projection_stack(proj):
    proj = np.ascontiguousarray(proj.astype(np.float32))
    proj_min = float(proj.min())
    proj_max = float(proj.max())
    return (proj - proj_min) / (proj_max - proj_min + 1e-6)

def resize_projection_stack(proj, mode, img_after_resize=(160,160)):
    if isinstance(proj, torch.Tensor):
        tensor = proj.float().unsqueeze(0)
    else:
        tensor = torch.from_numpy(np.ascontiguousarray(proj.astype(np.float32))).unsqueeze(0)
    if tensor.dim() == 3:
        tensor = tensor.unsqueeze(0)
    kwargs = {"align_corners": False} if mode == "bilinear" else {}
    resized = F.interpolate(
        tensor,
        size=img_after_resize,
        mode=mode,
        **kwargs,
    )
    return resized.squeeze(0).numpy().astype(np.float32)

def prepare_projection_stack(proj):
    return np.ascontiguousarray(
        resize_projection_stack(normalize_projection_stack(proj), mode="bilinear")
    )

pred_path = r"/home/zzx/data/exp_liftreg_2d/2d_xray/predictions"
data_path = r"/home/zzx/data/deepfluoro"
npz_path = r"/home/zzx/data/deepfluoro/deepfluoro_val/preprocessed_xray2d"

results = []

for file in sorted(os.listdir(pred_path)):
    if "fixed" not in file:
        continue
    file_name = file[:13]
    suject_name = file_name.split('_')[0]
    slice_num = file_name.split('_')[1]
    npz = np.load(os.path.join(npz_path, f"{file_name}.npz"))
    xrays = f"{data_path}/{suject_name}/xrays"
    xray, *_ = read_xray(f"{xrays}/{slice_num}.dcm", crop=100, linearize=False)
    target_proj = prepare_projection_stack(xray.squeeze(0).numpy().astype(np.float32))
    source_proj = np.ascontiguousarray(npz["source_proj"].astype(np.float32))
    source_seg = np.ascontiguousarray(npz["source_seg_proj"].astype(np.float32))
    target_seg = torch.load(f"{data_path}/{suject_name}/segmentations/{slice_num}.pt", weights_only=False)
    *_, height, width = target_seg.shape
    target_seg = center_crop(target_seg, (height - 100, width - 100))
    target_seg = target_seg.squeeze()
    resized_target_seg = resize_projection_stack(target_seg, mode="nearest")
    sitk.WriteImage(sitk.GetImageFromArray(resized_target_seg), os.path.join(pred_path, f"{suject_name}_{slice_num}_target_seg.nii.gz"))
    resized_target_seg = resized_target_seg.squeeze()
    warped_proj = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(pred_path, f"{file_name}_warped.nii.gz")))
    warped_seg = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(pred_path, f"{file_name}_warped_seg.nii.gz")))

    # normalize warped_proj the same way
    warped_proj = normalize_projection_stack(warped_proj).squeeze()
    source_proj_norm = normalize_projection_stack(source_proj).squeeze()
    target_proj_2d = target_proj.squeeze()

    dice_before = dice(source_seg.squeeze(), resized_target_seg)
    dice_after = dice(warped_seg.squeeze(), resized_target_seg)
    ssim_before = ssim(source_proj_norm, target_proj_2d)
    ssim_after = ssim(warped_proj, target_proj_2d)

    row = {
        "subject": suject_name,
        "id": file_name,
        "ssim_before": ssim_before,
        "ssim_after": ssim_after,
    }
    for label_id in [1,2,3,4,5,6]:
        row[f"dice_L{label_id}_before"] = dice_before.get(label_id, 0.0)
        row[f"dice_L{label_id}_after"] = dice_after.get(label_id, 0.0)
    row["dice_mean_before"] = np.mean([dice_before.get(l, 0.0) for l in [1,2,3,4,5,6]])
    row["dice_mean_after"] = np.mean([dice_after.get(l, 0.0) for l in [1,2,3,4,5,6]])

    results.append(row)
    print(f"{file_name}: SSIM {ssim_before:.4f}->{ssim_after:.4f}  Dice mean {row['dice_mean_before']:.4f}->{row['dice_mean_after']:.4f}")

df = pd.DataFrame(results)
# append mean row
mean_row = df.select_dtypes(include=[np.number]).mean().to_dict()
mean_row["subject"] = "MEAN"
mean_row["id"] = ""
df = pd.concat([df, pd.DataFrame([mean_row])], ignore_index=True)

excel_path = os.path.join(pred_path, "evaluation_results.xlsx")
df.to_excel(excel_path, index=False)
print(f"\nResults saved to {excel_path}")
print(df.to_string(index=False))
