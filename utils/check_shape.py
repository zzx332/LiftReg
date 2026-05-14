import os
import numpy as np

roots = [
    "/home/zzx/data/extracted_data_shift_only3/preprocessed",
    "/home/zzx/data/data/test_shift_only1/preprocessed",
]

expected_3d = (160, 160, 160)

for root in roots:
    if not os.path.isdir(root):
        print(f"[skip] no dir: {root}")
        continue
    print(f"\n=== scan: {root} ===")
    bad = 0
    for fn in sorted(os.listdir(root)):
        if not fn.endswith(".npz"):
            continue
        p = os.path.join(root, fn)
        try:
            z = np.load(p)
            s = z["source"].shape if "source" in z else None
            d = z["density"].shape if "density" in z else None
            tv = z["target_volume"].shape if "target_volume" in z else None
            tp = z["target_proj"].shape if "target_proj" in z else None

            # 只要关键3D字段不是(160,160,160)就报
            if s != expected_3d or d != expected_3d or tv != expected_3d:
                bad += 1
                print(f"[BAD] {fn}")
                print(f"      source={s}, density={d}, target_volume={tv}, target_proj={tp}")

        except Exception as e:
            bad += 1
            print(f"[ERR] {fn}: {e}")

    print(f"bad files: {bad}")