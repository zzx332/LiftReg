"""
DICOM C-arm 参数转换为外参矩阵 (Extrinsic Matrix)

将 DICOM XA (X-ray Angiography) 中的 C-arm 几何参数转换为
4x4 外参矩阵 [R|t]，用于 2D-3D 配准或 DRR 生成。

DICOM 参数说明：
  - DistanceSourceToDetector (SDD): 射线源到探测器的距离 (mm)
  - DistanceSourceToPatient / SourceToIsoCenterDistance (SID): 射线源到等中心点的距离 (mm)
  - PositionerPrimaryAngle (α): C-arm 绕患者纵轴旋转角度 (度)
      正值 = RAO (右前斜位), 负值 = LAO (左前斜位)
  - PositionerSecondaryAngle (β): C-arm 绕患者左右轴旋转角度 (度)
      正值 = CRA (颅侧倾斜), 负值 = CAU (尾侧倾斜)

坐标系约定 (DICOM 患者坐标系):
  - X 轴: 患者左侧 (L)
  - Y 轴: 患者后方 (P)
  - Z 轴: 患者头侧 (S, Superior)

外参矩阵将世界坐标 (等中心点为原点) 转换为相机坐标系。
"""

import numpy as np
import torch
from pathlib import Path
from typing import Dict, List

def rotation_x(angle_deg: float) -> np.ndarray:
    """绕 X 轴旋转矩阵"""
    theta = np.radians(angle_deg)
    c, s = np.cos(theta), np.sin(theta)
    return np.array([
        [1,  0,  0],
        [0,  c, -s],
        [0,  s,  c]
    ])


def rotation_y(angle_deg: float) -> np.ndarray:
    """绕 Y 轴旋转矩阵"""
    theta = np.radians(angle_deg)
    c, s = np.cos(theta), np.sin(theta)
    return np.array([
        [ c, 0, s],
        [ 0, 1, 0],
        [-s, 0, c]
    ])


def rotation_z(angle_deg: float) -> np.ndarray:
    """绕 Z 轴旋转矩阵"""
    theta = np.radians(angle_deg)
    c, s = np.cos(theta), np.sin(theta)
    return np.array([
        [c, -s, 0],
        [s,  c, 0],
        [0,  0, 1]
    ])


def dicom_params_to_extrinsic(
    distance_source_to_detector: float,
    distance_source_to_patient: float,
    primary_angle: float,
    secondary_angle: float,
) -> np.ndarray:
    """
    将 DICOM C-arm 参数转换为 4x4 外参矩阵。

    参数:
        distance_source_to_detector: 射线源到探测器距离 (mm), 即 SDD
        distance_source_to_patient: 射线源到等中心点距离 (mm), 即 SID
        primary_angle: 一次角 / 主角度 (度), LAO(-)/RAO(+)
        secondary_angle: 二次角 / 副角度 (度), CRA(+)/CAU(-)

    返回:
        extrinsic: 4x4 外参矩阵 [R|t; 0 0 0 1]

    说明:
        C-arm 初始位置 (0°, 0°): 射线源在患者正前方 (AP view)，
        即射线源位于 Y 轴负方向 (0, -SID, 0)，探测器在 (0, SDD-SID, 0)。

        旋转顺序:
        1. 绕 Z 轴旋转 primary_angle (LAO/RAO)
        2. 绕旋转后的 X 轴旋转 secondary_angle (CRA/CAU)

        外参矩阵将世界坐标系中的点转换到相机坐标系:
        - 相机坐标系: 原点在射线源, Z 轴指向探测器 (即射线传播方向)
    """

    SDD = distance_source_to_detector
    SID = distance_source_to_patient

    alpha = primary_angle     # 一次角 (度)
    beta = secondary_angle    # 二次角 (度)

    # ---- 步骤 1: 计算 C-arm 旋转矩阵 ----
    # 先绕 Z 轴旋转 primary_angle, 再绕 X 轴旋转 secondary_angle
    # R_carm = Rx(beta) @ Rz(alpha)
    R_carm = rotation_x(beta) @ rotation_z(alpha)

    # ---- 步骤 2: 计算射线源在世界坐标系中的位置 ----
    # 初始位置 (AP view): 射线源在 (0, -SID, 0)
    source_initial = np.array([0, -SID, 0])
    # 经过 C-arm 旋转后的射线源位置
    source_world = R_carm @ source_initial

    # ---- 步骤 3: 构建相机坐标系 ----
    # 相机 Z 轴: 从射线源指向等中心点 (原点)
    cam_z = -source_world / np.linalg.norm(source_world)  # 归一化

    # 选择一个 up 向量来确定相机的 X 和 Y 轴
    # 通常选择世界坐标系的 Z 轴 (头侧方向) 作为 up
    world_up = np.array([0, 0, 1.0])

    # 相机 X 轴: cam_z × world_up (叉积)
    cam_x = np.cross(cam_z, world_up)
    cam_x_norm = np.linalg.norm(cam_x)

    # 处理退化情况 (cam_z 平行于 world_up)
    if cam_x_norm < 1e-6:
        world_up = np.array([0, 1.0, 0])
        cam_x = np.cross(cam_z, world_up)
        cam_x_norm = np.linalg.norm(cam_x)

    cam_x = cam_x / cam_x_norm

    # 相机 Y 轴: cam_z × cam_x (确保右手系)
    cam_y = np.cross(cam_z, cam_x)
    cam_y = cam_y / np.linalg.norm(cam_y)

    # ---- 步骤 4: 构建外参矩阵 ----
    # 旋转矩阵: 将世界坐标转换到相机坐标
    R = np.vstack([cam_x, cam_y, cam_z])  # 3x3

    # 平移向量: t = -R @ source_world
    t = -R @ source_world

    # 组装 4x4 外参矩阵
    extrinsic = np.eye(4)
    extrinsic[:3, :3] = R
    extrinsic[:3, 3] = t

    return extrinsic


def dicom_params_to_projection_matrix(
    distance_source_to_detector: float,
    distance_source_to_patient: float,
    primary_angle: float,
    secondary_angle: float,
    detector_rows: int = 512,
    detector_cols: int = 512,
    pixel_spacing: float = 0.308,
) -> np.ndarray:
    """
    将 DICOM C-arm 参数转换为 3x4 投影矩阵 P = K @ [R|t]。

    参数:
        distance_source_to_detector: 射线源到探测器距离 (mm)
        distance_source_to_patient: 射线源到等中心点距离 (mm)
        primary_angle: 一次角 (度)
        secondary_angle: 二次角 (度)
        detector_rows: 探测器行数 (像素)
        detector_cols: 探测器列数 (像素)
        pixel_spacing: 探测器像素间距 (mm)

    返回:
        projection_matrix: 3x4 投影矩阵
    """

    SDD = distance_source_to_detector

    # 焦距 (像素单位)
    fx = SDD / pixel_spacing
    fy = SDD / pixel_spacing

    # 主点 (探测器中心)
    cx = detector_cols / 2.0
    cy = detector_rows / 2.0

    # 内参矩阵
    K = np.array([
        [fx,  0, cx],
        [ 0, fy, cy],
        [ 0,  0,  1]
    ])

    # 外参矩阵
    extrinsic = dicom_params_to_extrinsic(
        distance_source_to_detector,
        distance_source_to_patient,
        primary_angle,
        secondary_angle,
    )

    # 投影矩阵 P = K @ [R|t] (取外参矩阵的前3行)
    projection_matrix = K @ extrinsic[:3, :]

    return projection_matrix


def print_matrix(name: str, matrix: np.ndarray) -> None:
    """格式化打印矩阵"""
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")
    print(matrix)
    print()

def _parse_mhd_header(mhd_path: Path) -> Dict[str, str]:
    """读取 mhd 文本头，解析 key=value 字段。"""
    meta: Dict[str, str] = {}
    with mhd_path.open("r", encoding="utf-8", errors="ignore") as f:
        for raw in f:
            line = raw.strip()
            if not line or "=" not in line:
                continue
            key, value = line.split("=", 1)
            meta[key.strip()] = value.strip()
    return meta


def collect_dsa_mhd_info(data_path: str, if_save: bool = False) -> List[Dict]:
    """
    读取 data_path 下所有 case/DSA/*/*.mhd，并提取几何参数:
      - ElementSpacing -> dx, dy
      - DimSize -> detector_width
    """
    root = Path(data_path)
    records: List[Dict] = []

    for case_dir in sorted(root.iterdir()):
        if not case_dir.is_dir():
            continue
        dsa_dir = case_dir / "DSA"
        if not dsa_dir.exists():
            continue

        for mhd_path in sorted(dsa_dir.rglob("*.mhd")):
            meta = _parse_mhd_header(mhd_path)

            if "ElementSpacing" not in meta or "DimSize" not in meta:
                continue

            spacing_vals = [float(x) for x in meta["ElementSpacing"].split()]
            dim_vals = [int(float(x)) for x in meta["DimSize"].split()]
            if len(spacing_vals) < 2 or len(dim_vals) < 1:
                continue

            # 若 mhd 中包含几何参数，则计算该条目的外参
            # sdd = float(meta["DistanceSourceToDetector"]) if "DistanceSourceToDetector" in meta else None
            # sid = float(meta["DistanceSourceToPatient"]) if "DistanceSourceToPatient" in meta else None
            # primary = float(meta["PositionerPrimaryAngle"]) if "PositionerPrimaryAngle" in meta else None
            # secondary = float(meta["PositionerSecondaryAngle"]) if "PositionerSecondaryAngle" in meta else None
            sdd = float(meta["Extra_SourceToDetectorDistance"]) if "Extra_SourceToDetectorDistance" in meta else None
            sid = float(meta["Extra_SourceToIsoCenterDistance"]) if "Extra_SourceToIsoCenterDistance" in meta else None
            primary = float(meta["Extra_PrimaryAngle"]) if "Extra_PrimaryAngle" in meta else None
            secondary = float(meta["Extra_SecondaryAngle"]) if "Extra_SecondaryAngle" in meta else None

            extrinsic_tensor = None
            if None not in (sdd, sid, primary, secondary):
                extrinsic = dicom_params_to_extrinsic(
                    distance_source_to_detector=sdd,
                    distance_source_to_patient=sid,
                    primary_angle=primary,
                    secondary_angle=secondary,
                )
                extrinsic_tensor = torch.from_numpy(extrinsic).float()

            record = {
                # "case_name": case_dir.name,
                "case_name": Path(mhd_path).stem,
                "mhd_path": str(mhd_path),
                "ct_patient_name": case_dir.name,
                "dx": spacing_vals[0],
                "dy": spacing_vals[1],
                "detector_width": dim_vals[0],
                "ElementSpacing": meta["ElementSpacing"],
                "DimSize": meta["DimSize"],
                "DistanceSourceToDetector": sdd,
                "DistanceSourceToPatient": sid,
                "PositionerPrimaryAngle": primary,
                "PositionerSecondaryAngle": secondary,
                "extrinsic": extrinsic_tensor,  # 4x4 tensor, 缺参时为 None
            }
            if ("dx" not in records or "dy" not in records) and "ElementSpacing" in records:
                sp = record["ElementSpacing"]
                vals = [float(x) for x in (sp.split() if isinstance(sp, str) else sp)]
                if len(vals) >= 2:
                    record["dx"], record["dy"] = vals[0], vals[1]
            if ("detector_width" not in record or "detector_height" not in record) and "DimSize" in record:
                ds = record["DimSize"]
                vals = [int(float(x)) for x in (ds.split() if isinstance(ds, str) else ds)]
                if len(vals) >= 2:
                    record["detector_width"], record["detector_height"] = vals[0], vals[1]
            if if_save:
                save_path = f"{mhd_path.parent}/dsa_mhd_info.pt"
                print(save_path)
                torch.save(record, save_path)
            records.append(record)

    return records


# ============================================================
#  主程序: 使用图中的参数进行转换
# ============================================================
if __name__ == "__main__":
    data_path = "/home/zzx/data/pair_CT_DSA_10"
    records = collect_dsa_mhd_info(data_path)

    # print(f"共读取 {len(records)} 个 mhd")
    if records:
        print("示例记录:")
        print(records[0])
