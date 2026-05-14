import os
import numpy as np
import nibabel as nib

data_path = r"D:\dataset\CTA_DSA\DeepFluoro\extracted_data\17-1882\diffdrr"
file_list = os.listdir(data_path)

# 可选：过滤特定类型的文件
# 例如只保留 .nii.gz 文件
nii_files = [f.replace('.nii.gz', '') for f in file_list if f.endswith('.nii.gz')]

# 或者保留所有文件
all_files = file_list

# 将文件列表转换为 numpy 数组并保存
file_array = np.array(nii_files)  # 或 np.array(all_files)

# 保存为 npy 文件
output_file = r"D:\dataset\CTA_DSA\DeepFluoro\exp_liftreg\data\test\data_id.npy"
np.save(output_file, file_array)

print(f"文件列表已保存到: {output_file}")
print(f"总共 {len(file_array)} 个文件")
print(f"前5个文件: {file_array[:5]}")