"""2D analogue of polypose.weights.compute_weights.

Given a 2D integer label map, compute per-segment soft blending weights for
the polyrigid warp. Mirrors the 3D version in spirit (EDT + gravity), but
operates purely on 2D images via scipy's CPU EDT (no cupy dependency).
"""

from typing import Optional, Sequence, Tuple, Union

import numpy as np
import torch
from scipy.ndimage import distance_transform_edt


LabelGroup = Union[int, Sequence[int]]


def _to_numpy_label(label_map) -> np.ndarray:
    if isinstance(label_map, torch.Tensor):
        arr = label_map.detach().cpu().numpy()
    else:
        arr = np.asarray(label_map)
    if arr.ndim != 2:
        raise ValueError(f"label_map must be 2D (H, W); got shape {arr.shape}")
    if not np.issubdtype(arr.dtype, np.integer):
        arr = np.rint(arr).astype(np.int64)
    return arr


def compute_weights_2d(
    label_map,
    labels: Optional[Sequence[LabelGroup]] = None,
    weightfn: str = "gravity",
    normalize: bool = True,
    spacing: Optional[Sequence[float]] = None,
    eps: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute the 2D polyrigid blending weights.

    Args:
        label_map: ``(H, W)`` integer tensor / ndarray. Value 0 is background.
        labels: list of rigid bodies. Each element is either a single label id
            or a list of label ids (their union forms one rigid body). When
            ``None``, every distinct non-zero label becomes its own rigid body
            (sorted ascending).
        weightfn: ``"gravity"`` | ``"invdf"`` | ``"exp"`` -- same semantics as
            ``polypose.compute_weights``.
        normalize: if ``True``, weights are normalized along the channel axis
            so they sum to 1 at every pixel.
        spacing: optional ``(sy, sx)`` pixel spacing passed to the EDT.
        eps: parameter for ``invdf`` / ``exp`` weight functions.

    Returns:
        segmentations: ``(H, W)`` int64 tensor (1..K, 0 = background).
        weights:       ``(K, H, W)`` float32 tensor.
    """

    label_np = _to_numpy_label(label_map)

    if labels is None:
        unique = [int(v) for v in np.unique(label_np) if int(v) != 0]
        labels = sorted(unique)

    if len(labels) == 0:
        raise ValueError("label_map has no non-zero labels to build segments from")

    sampling = tuple(spacing) if spacing is not None else None

    masks = []
    edtmaps = []
    for label in labels:
        group = [int(label)] if isinstance(label, (int, np.integer)) else [int(v) for v in label]
        structure = np.isin(label_np, group)
        if not structure.any():
            continue
        edt = distance_transform_edt(~structure, sampling=sampling)
        masks.append(torch.from_numpy(structure))
        edtmaps.append(torch.from_numpy(np.asarray(edt, dtype=np.float32)))

    if not masks:
        raise ValueError("None of the requested label groups appear in label_map")

    segmentations = torch.stack(
        [i * m.to(torch.int64) for i, m in enumerate(masks, start=1)]
    ).sum(dim=0)

    masses = torch.tensor([float(m.sum().item()) for m in masks], dtype=torch.float32)
    masses = masses / masses.sum().clamp_min(1e-8)

    weights = []
    for mass, edt in zip(masses, edtmaps):
        if weightfn == "gravity":
            w = mass.item() / (edt ** 2 + 1.0)
        elif weightfn == "invdf":
            w = 1.0 / (eps * edt ** 2 + 1.0)
        elif weightfn == "exp":
            w = torch.exp(eps * edt)
        else:
            raise ValueError(
                f"weightfn must be 'gravity', 'invdf', or 'exp', not {weightfn!r}"
            )
        weights.append(w)
    weights = torch.stack(weights)

    if normalize:
        weights = weights / (weights.sum(dim=0, keepdim=True) + 1e-8)

    return segmentations, weights.to(torch.float32)


__all__ = ["compute_weights_2d"]
