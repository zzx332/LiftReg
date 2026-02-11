from torchio import ScalarImage 
import numpy as np

volume =r"D:\dataset\CTA_DSA\DeepFluoro\exp_liftreg\data\val\img\subject01_source.nii.gz"
volume = ScalarImage(volume)
isocenter = volume.get_center()
Tinv = np.array(
    [
        [1.0, 0.0, 0.0, -isocenter[0]],
        [0.0, 1.0, 0.0, -isocenter[1]],
        [0.0, 0.0, 1.0, -isocenter[2]],
        [0.0, 0.0, 0.0, 1.0],
    ]
)
new_affine = Tinv.dot(volume.affine)
volume_centered = ScalarImage(
    tensor=volume.data,      # 原始体数据
    affine=new_affine     # 新 affine
)
# 保存
output_path = r"D:\dataset\CTA_DSA\DeepFluoro\exp_liftreg\data\val\img\subject01_source_iso.nii.gz"
volume_centered.save(output_path)
print(f"已保存到: {output_path}")