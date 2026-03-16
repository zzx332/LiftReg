import numpy as np
import os
import SimpleITK as sitk
data_path = "/home/zzx/data/data/test_shift_only"
npz = np.load(os.path.join(data_path, "preprocessed", f'subject06_drr1.npz'))
target_volume = npz['target_volume']
target_proj = npz['target_proj']
sitk.WriteImage(sitk.GetImageFromArray(target_proj), os.path.join(data_path, "target_proj.nii.gz"))
sitk.WriteImage(sitk.GetImageFromArray(np.transpose(target_volume, (2,1,0))), os.path.join(data_path, "target_volume.nii.gz"))
# print(npz.keys())