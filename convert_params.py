import numpy as np
import math
import torch
import os


extrinsic , *_ = torch.load(os.path.join(r"D:\dataset\CTA_DSA\DeepFluoro\exp_liftreg\val\deepfluoro\subject01","subject01_pose.pt"), weights_only=False)["pose"]
R = extrinsic[:3, :3].numpy()
t = extrinsic[:3, 3].numpy()
# R = np.array([
#     [ 0.0014173903036862612,  5.773301836597966e-06,  0.9999991059303284],
#     [-0.9999907612800598,    -0.004075835924595594,  0.0014174006646499038],
#     [ 0.004075841512531042,  -0.9999918341636658,   -0.0],
# ])
# t = np.array([-157.58221435546875, -2.6757495403289795, -631.7764892578125])
d = R[2, :]                      # row3 = R^T * ez
d = d / np.linalg.norm(d)
C = np.dot(R.T, t.T)
sid = np.linalg.norm(C)
dx, dy, dz = d
primary = math.degrees(math.atan2(dx, -dy))
secondary = math.degrees(math.atan2(dz, math.sqrt(dx*dx + dy*dy)))

print(primary, secondary, sid)
