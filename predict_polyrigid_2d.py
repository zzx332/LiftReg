"""Standalone 2D polyrigid prediction.

Loads a `RegNet2D` checkpoint that was trained with ``use_polyrigid=True`` and
runs inference on the validation set described by the given JSON setting. For
every case the per-segment soft weights are computed on-the-fly from the 2D
``source_label`` projection via :func:`liftreg.utils.compute_weights_2d`.

Results are saved as ``.nii.gz`` images (source/fixed/warped/phi/weights), the
warped segmentation if available, and the predicted polyrigid translations as
``.npy``.
"""

import argparse
import os
import sys

import numpy as np
import SimpleITK as sitk
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from liftreg.utils import module_parameters as pars
from liftreg.utils.compute_weights_2d import compute_weights_2d
from liftreg.utils.general import get_class, make_dir
from liftreg.utils.utils import set_seed_for_demo


def parse_args():
    p = argparse.ArgumentParser(description="2D polyrigid inference")
    p.add_argument("-s", "--setting_path", required=True, type=str,
                   help="Path to the *_task_setting_2d.json used at training time")
    p.add_argument("-c", "--checkpoint", required=True, type=str,
                   help="Path to a polyrigid-trained RegNet2D checkpoint")
    p.add_argument("-o", "--save_path", required=True, type=str,
                   help="Directory to save predictions into")
    p.add_argument("--num_segments", type=int, default=None,
                   help="Override num_segments. Defaults to JSON train.num_segments (or 3).")
    p.add_argument("--weightfn", type=str, default="gravity",
                   choices=["gravity", "invdf", "exp"],
                   help="Weight function for compute_weights_2d (mirrors polypose)")
    p.add_argument("--phase", type=str, default="val",
                   choices=["train", "val", "test", "debug"],
                   help="Dataset phase to iterate (default: val)")
    return p.parse_args()


def _move_batch_to_device(batch, device):
    out = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            out[k] = v.to(device)
        else:
            out[k] = v
    return out


def _extract_label_2d(label_tensor: torch.Tensor) -> torch.Tensor:
    """Return a single (H, W) integer label image from a sample's source_label.

    ``TipsDataset2D`` caches ``source_mask`` as a (P, H, W) uint8 stack; after
    the DataLoader prepends a batch dim it becomes (1, P, H, W). We grab the
    first projection.
    """

    t = label_tensor.detach().cpu()
    while t.dim() > 2:
        t = t[0]
    return t.long()


def _pad_or_truncate_weights(w: torch.Tensor, K: int) -> torch.Tensor:
    """Match the channel count of ``w`` to ``K`` via zero-pad or truncation."""

    k_case = w.shape[0]
    if k_case == K:
        return w
    if k_case < K:
        pad = torch.zeros((K - k_case, *w.shape[1:]), dtype=w.dtype)
        return torch.cat([w, pad], dim=0)
    print(f"[warn] case has {k_case} segments > checkpoint K={K}; truncating")
    return w[:K]


def _save_tensor_as_nii(tensor: torch.Tensor, path: str):
    arr = tensor.detach().cpu().numpy()
    sitk.WriteImage(sitk.GetImageFromArray(arr), path)


def _save_case(save_path: str, case: str, batch: dict, output: dict, weights: torch.Tensor):
    make_dir(save_path)
    _save_tensor_as_nii(batch["source_proj"][0], os.path.join(save_path, f"{case}_source.nii.gz"))
    _save_tensor_as_nii(output["target_proj"][0], os.path.join(save_path, f"{case}_fixed.nii.gz"))
    _save_tensor_as_nii(output["warped_moving"][0], os.path.join(save_path, f"{case}_warped.nii.gz"))
    _save_tensor_as_nii(output["phi"][0], os.path.join(save_path, f"{case}_phi.nii.gz"))
    _save_tensor_as_nii(weights, os.path.join(save_path, f"{case}_weights.nii.gz"))

    if "warped_label" in output:
        _save_tensor_as_nii(output["warped_label"][0], os.path.join(save_path, f"{case}_warped_seg.nii.gz"))
    if "source_label" in batch:
        _save_tensor_as_nii(batch["source_label"][0].float(), os.path.join(save_path, f"{case}_source_seg.nii.gz"))

    if "polyrigid_params" in output:
        params = output["polyrigid_params"][0].detach().cpu().numpy()
        np.save(os.path.join(save_path, f"{case}_polyrigid_params.npy"), params)


def main():
    args = parse_args()
    set_seed_for_demo()

    setting = pars.ParameterDict()
    setting.load_JSON(args.setting_path)
    dataset_setting = setting["dataset"]
    train_setting = setting["train"]

    if not dataset_setting[("use_segmentation_map", False, "")]:
        print(
            "[warn] dataset.use_segmentation_map is False in the JSON. "
            "Forcing it to True for this run so that source_label is loaded."
        )
        dataset_setting["use_segmentation_map"] = True

    img_size = tuple(dataset_setting["img_after_resize"])
    K = args.num_segments if args.num_segments is not None else int(
        train_setting[("num_segments", 3, "polyrigid segments")]
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[info] device={device}  K={K}  weightfn={args.weightfn}")

    dataset_cls = get_class(dataset_setting["dataset_class"])
    data_root = (
        dataset_setting["val_data_path"]
        if args.phase == "val"
        else dataset_setting["data_path"]
    )
    dataset = dataset_cls(data_root, phase=args.phase, option=dataset_setting)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    model_cls = get_class(train_setting["model_class"])
    model = model_cls(
        img_size=img_size, use_polyrigid=True, num_segments=K,
    ).to(device).eval()

    if not os.path.isfile(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=device)
    state_dict = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"[warn] missing keys when loading checkpoint: {missing}")
    if unexpected:
        print(f"[warn] unexpected keys when loading checkpoint: {unexpected}")
    print(f"[info] loaded checkpoint: {args.checkpoint}")

    make_dir(args.save_path)

    with torch.no_grad():
        for batch, identifier in loader:
            case = identifier[0] if isinstance(identifier, (list, tuple)) else identifier
            if "source_label" not in batch:
                print(f"[skip] {case}: no source_label in sample")
                continue

            label_2d = _extract_label_2d(batch["source_label"])
            if int(label_2d.abs().sum().item()) == 0:
                print(f"[skip] {case}: source_label is all zeros")
                continue

            try:
                _, weights = compute_weights_2d(
                    label_2d, labels=None, weightfn=args.weightfn, normalize=True,
                )
            except ValueError as e:
                print(f"[skip] {case}: {e}")
                continue
            weights = _pad_or_truncate_weights(weights, K)

            batch = _move_batch_to_device(batch, device)
            batch["weights"] = weights.unsqueeze(0).to(device).float()

            output = model(batch)
            _save_case(args.save_path, case, batch, output, weights)
            print(f"[ok] {case}  K_case={int((weights.sum(dim=(1,2)) > 0).sum())}")

    print(f"[done] outputs saved to {args.save_path}")


if __name__ == "__main__":
    main()
