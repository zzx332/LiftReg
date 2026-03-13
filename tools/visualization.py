import os
import sys
import numpy as np
import torch
import pyvista as pv

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from liftreg.dataset.FluoroDataset import FluoroDataset
from liftreg.utils import module_parameters as pars
from liftreg.utils.sdct_projection_utils import (
    backproj_grids_with_SOUV,
    make_centered_volume_xyz,
)

DU = 0.388
DV = 0.388


def load_sample():
    setting = pars.ParameterDict()
    setting.load_JSON(os.path.join(ROOT, "deepfluoro_task_setting.json"))
    option = setting["dataset"]

    dataset = FluoroDataset(
        data_path=option["data_path"],
        phase="train",
        option=option,
    )
    (sample, fname) = dataset[0]
    print("sample:", fname)
    return dataset, sample, fname


def to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def get_geometry(sample):
    S, O, U, V = sample["target_poses_SOUV"]

    # 当前代码里通常是 [1, 3]，这里统一拉平成 [3]
    S = to_numpy(S).reshape(-1, 3)[0]
    O = to_numpy(O).reshape(-1, 3)[0]
    U = to_numpy(U).reshape(-1, 3)[0]
    V = to_numpy(V).reshape(-1, 3)[0]
    return S, O, U, V


def get_projection_for_backprojection(sample):
    # 你的 backprojection 用的是：
    # target_proj.squeeze(0).permute(0, 2, 1).flip(dims=[2])
    proj = torch.as_tensor(sample["target_proj"], dtype=torch.float32)
    if proj.ndim == 2:
        proj = proj.unsqueeze(0)
    # proj_bp = proj.permute(0, 2, 1).flip(dims=[2])
    # proj_bp = proj.permute(0, 2, 1)
    proj_bp = proj.flip(dims=[1])
    return proj_bp[0].detach().cpu().numpy()  # [H, W]

def get_mask_projection_for_detector(dataset, sample):
    label_3d = torch.as_tensor(sample["source_label"], dtype=torch.float32).squeeze(0)  # [X,Y,Z]
    affine_inv = torch.as_tensor(sample["affine"], dtype=torch.float32).inverse()
    target_poses = sample["target_poses"]
    dataset._init_projector_if_needed()
    # 3D mask -> 2D projection，输出通常是 [1, P, H, W]
    mask_proj = dataset._run_renderer(label_3d, target_poses, affine_inv)

    # 和你 target_proj / backprojection 使用同一套方向变换
    # mask_proj = mask_proj.squeeze(0).permute(0, 2, 1).flip(dims=[2])
    # mask_proj = mask_proj.squeeze(0).permute(0, 2, 1)
    # mask_proj = mask_proj.squeeze(0).permute(0, 2, 1)
    mask_proj = mask_proj.squeeze(0).flip(dims=[1])

    # 先看第一个投影
    mask_proj_2d = mask_proj[0].detach().cpu().numpy()

    # 二值化，避免 line integral 后不是纯 0/1
    mask_proj_bin = (mask_proj_2d > 1e-3).astype(np.float32)
    return mask_proj_bin

def make_volume_box(source, spacing):
    D, H, W = source.shape
    sx, sy, sz = spacing

    x_min = -(W - 1) / 2 * sx
    x_max = +(W - 1) / 2 * sx
    y_min = -(H - 1) / 2 * sy
    y_max = +(H - 1) / 2 * sy
    z_min = -(D - 1) / 2 * sz
    z_max = +(D - 1) / 2 * sz

    return pv.Box(bounds=(x_min, x_max, y_min, y_max, z_min, z_max))


def make_detector_plane(O, U, V, proj_2d, du=DU, dv=DV):
    H, W = proj_2d.shape
    cu = (W - 1) / 2.0
    cv = (H - 1) / 2.0

    vv, uu = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    pts = (
        O[None, None, :]
        + (uu - cu)[..., None] * du * U[None, None, :]
        + (vv - cv)[..., None] * dv * V[None, None, :]
    )

    grid = pv.StructuredGrid(
        pts[:, :, 0],
        pts[:, :, 1],
        pts[:, :, 2],
    )
    grid.point_data["proj"] = proj_2d.reshape(-1, order="F")
    return grid


def add_detector_rays(plotter, S, O, U, V, proj_2d, stride=80, du=DU, dv=DV):
    H, W = proj_2d.shape
    cu = (W - 1) / 2.0
    cv = (H - 1) / 2.0
    v_steps = sorted(set(list(range(0, H, stride)) + [H - 1]))
    u_steps = sorted(set(list(range(0, W, stride)) + [W - 1]))
    for v in v_steps:
        for u in u_steps:
            hit = O + (u - cu) * du * U + (v - cv) * dv * V
            ray = pv.Line(S, hit)
            plotter.add_mesh(ray, color="yellow", opacity=0.18, line_width=1)
            
def add_target_volume(plotter, sample, spacing):
    target_volume = to_numpy(sample["target_volume"])[0].astype(np.float32)  # [D,H,W]
    D, H, W = target_volume.shape
    sx, sy, sz = spacing

    # PyVista 是 x,y,z；你的数组是 D,H,W，所以先转成 W,H,D
    volume_xyz = np.transpose(target_volume, (2, 1, 0))
    # volume_xyz = target_volume

    grid = pv.ImageData()
    grid.dimensions = np.array(volume_xyz.shape) + 1  # 作为 cell_data
    grid.spacing = (sx, sy, sz)
    # grid.origin = (-W * sx / 2.0, -H * sy / 2.0, -D * sz / 2.0)
    grid.origin = (-(W - 1) * sx / 2.0, -(H - 1) * sy / 2.0, -(D - 1) * sz / 2.0)
    grid.cell_data["target_volume"] = volume_xyz.flatten(order="F")

    # 体渲染
    plotter.add_volume(
        grid,
        scalars="target_volume",
        cmap="gray",
        opacity="sigmoid",
        shade=False,
    )

def add_label_surface(plotter, sample):
    label = to_numpy(sample["source_label"])[0].astype(np.float32)  # X, Y, Z
    affine = to_numpy(sample["affine"]).astype(np.float32)
    # D, H, W = label.shape
    W, H, D = label.shape
    # label = np.transpose(label, (2, 1, 0))
    # 不做 transpose，直接按 [D,H,W] 建格子
    # PyVista x=D, y=H, z=W，与 affine 期望的 (D,H,W) 一致
    grid = pv.ImageData()
    # grid.dimensions = np.array([D, H, W]) + 1
    grid.dimensions = np.array([W, H, D]) + 1
    grid.spacing = (1.0, 1.0, 1.0)
    grid.origin = (0.0, 0.0, 0.0)
    grid.cell_data["label"] = label.flatten(order="F")

    label_surface = grid.threshold(0.5, scalars="label")
    label_surface.transform(affine, inplace=True)

    plotter.add_mesh(
        label_surface,
        color="gold",
        opacity=0.35,
        smooth_shading=True,
        show_scalar_bar=False,
    )


def add_backprojection_slice(plotter, sample, S, O, U, V, proj_2d, spacing):
    source = to_numpy(sample["source"])[0]  # [D,H,W]
    D, H, W = source.shape

    proj_h, proj_w = proj_2d.shape
    cu = (proj_w - 1) / 2.0
    cv = (proj_h - 1) / 2.0

    S_t = torch.tensor(S, dtype=torch.float32).view(1, 1, 3)
    O_t = torch.tensor(O, dtype=torch.float32).view(1, 1, 3)
    U_t = torch.tensor(U, dtype=torch.float32).view(1, 1, 3)
    V_t = torch.tensor(V, dtype=torch.float32).view(1, 1, 3)

    grid = backproj_grids_with_SOUV(
        img_shape=(D, H, W),
        S=S_t,
        O=O_t,
        U=U_t,
        V=V_t,
        proj_w=proj_w,
        proj_h=proj_h,
        du=DU,
        dv=DV,
        cu=cu,
        cv=cv,
        device=torch.device("cpu"),
    )  # [1,1,D,H,W,2]

    xyz = make_centered_volume_xyz(
        D, H, W, device=torch.device("cpu"), spacing=tuple(spacing)
    )  # [D,H,W,3]

    mid_h = H // 2
    stride = 10

    uv_norm = grid[0, 0, ::stride, mid_h, ::stride, :]   # [d, w, 2]
    vox_xyz = xyz[::stride, mid_h, ::stride, :]          # [d, w, 3]

    u = (uv_norm[..., 0] + 1.0) * 0.5 * (proj_w - 1)
    v = (uv_norm[..., 1] + 1.0) * 0.5 * (proj_h - 1)

    det_xyz = (
        torch.tensor(O, dtype=torch.float32)[None, None, :]
        + (u - cu)[..., None] * DU * torch.tensor(U, dtype=torch.float32)[None, None, :]
        + (v - cv)[..., None] * DV * torch.tensor(V, dtype=torch.float32)[None, None, :]
    )

    vox_np = vox_xyz.reshape(-1, 3).numpy()
    det_np = det_xyz.reshape(-1, 3).numpy()

    # 只画点，不画全连线，避免太乱
    vox_cloud = pv.PolyData(vox_np)
    det_cloud = pv.PolyData(det_np)

    plotter.add_mesh(vox_cloud, color="deepskyblue", point_size=5, render_points_as_spheres=True)
    plotter.add_mesh(det_cloud, color="red", point_size=4, render_points_as_spheres=True)

    # 再随机抽一些连线
    n = len(vox_np)
    take = np.linspace(0, n - 1, min(80, n), dtype=int)
    for i in take:
        line = pv.Line(vox_np[i], det_np[i])
        plotter.add_mesh(line, color="lime", opacity=0.12, line_width=1)


def main():
    dataset, sample, fname = load_sample()

    source = to_numpy(sample["source"])[0]        # [D,H,W]
    spacing = np.asarray(sample["spacing"], dtype=np.float32)
    S, O, U, V = get_geometry(sample)
    proj_2d = get_projection_for_backprojection(sample)

    box = make_volume_box(source, spacing)
    detector = make_detector_plane(O, U, V, proj_2d)

    p = pv.Plotter()
    p.add_axes()
    p.add_title(f"Backprojection Visualization: {fname}")

    # 体数据：先只画包围盒，最清楚
    p.add_mesh(box, style="wireframe", color="red", line_width=2)

    # X-ray source
    p.add_mesh(pv.Sphere(radius=3.0, center=S), color="orange")

    # Detector plane with projection intensities
    p.add_mesh(
        detector,
        scalars="proj",
        cmap="gray",
        opacity=0.90,
        show_edges=False,
        lighting=False,
    )
    # 再把 mask 投影挂到同一个 detector 平面上
    mask_proj_2d = get_mask_projection_for_detector(dataset, sample)
    detector.point_data["mask"] = mask_proj_2d.reshape(-1, order="F")

    # 画 detector 上的 mask 轮廓
    mask_contour = detector.contour(isosurfaces=[0.5], scalars="mask")
    p.add_mesh(mask_contour, color="lime", line_width=3)
    # 射线示意
    # add_detector_rays(p, S, O, U, V, proj_2d, stride=90)
    add_target_volume(p, sample, spacing)
    add_label_surface(p, sample)
    # # 反投影切片：体素 -> detector 落点
    # add_backprojection_slice(p, sample, S, O, U, V, proj_2d, spacing)

    # p.show()
    # output_dir = os.path.join("/data/zhouzhexin/ctdsa", "visualization")
    output_dir = "/home/zzx/data/exp_liftreg/extracted_data_shift_only1/deepfluoro/2026_03_10_18_21_18/tests"
    os.makedirs(output_dir, exist_ok=True)

    html_path = os.path.join(output_dir, f"{fname}_backprojection.html")
    print(f"Saving interactive html to: {html_path}")

    p.export_html(html_path)
    p.close()


if __name__ == "__main__":
    main()