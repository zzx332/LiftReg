from totalsegmentator.python_api import totalsegmentator
import nibabel as nib
from pathlib import Path
import numpy as np

def main():
    data_path = "/home/zzx/data/tips_dataset3d"
    for case_name in sorted(os.listdir(data_path)):
        case_dir = Path(data_path) / case_name
        if not case_dir.is_dir():
            continue
        matches = sorted(case_dir.glob("*_volume.nii.gz"))
        if not matches:
            print(f"[skip] {case_name}: no *_volume.nii.gz")
            continue

        ct_input = str(matches[0])
        liver_img = totalsegmentator(
            input=ct_input,          # 你当前版本用 input=；有的版本是 input_path=
            roi_subset=["liver"],
            # task="total_anatomy_abdo",  # 按需
            nr_thr_resamp=2,           # 可选：降并行，减 RAM
            nr_thr_saving=1,
        )

        ref = nib.load(ct_input)
        liver_data = liver_img.get_fdata()
        liver_mask = (liver_data > 0).astype(np.uint8)   # 前景 -> 1，背景 -> 0
        liver_out = nib.Nifti1Image(
            liver_mask,
            affine=ref.affine,
            header=ref.header,
        )
        nib.save(liver_out, str(case_dir / "liver.nii.gz"))
        print(f"肝脏分割完成: {case_name} -> {case_dir / 'liver.nii.gz'}")

if __name__ == "__main__":
    import os
    main()