import numpy as np
import os
import SimpleITK as sitk

data_dir = "/home/zzx/data/deepfluoro/deepfluoro_train/preprocessed_xray2d"
npz = np.load(os.path.join(data_dir, "subject01_000.npz")) 
source_seg_proj = npz["source_seg_proj"]  # 获取名为 "source_seg_proj" 的数组
source_proj = npz["source_proj"]  # 获取名为 "source_seg_proj" 的数组
sitk.WriteImage(sitk.GetImageFromArray(source_seg_proj), os.path.join("/home/zzx/data/deepfluoro", "subject01_000_source_seg_proj.nii.gz"))  # 将数组保存为 NIfTI 格式的图像文件
sitk.WriteImage(sitk.GetImageFromArray(source_proj), os.path.join("/home/zzx/data/deepfluoro", "subject01_000_source_proj.nii.gz"))  # 将数组保存为 NIfTI 格式的图像文件
print(npz.files)  # 输出包含的数组名称