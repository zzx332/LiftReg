from pathlib import Path

import click
import torch
from diffdrr.pose import RigidTransform
from torchio import ScalarImage
from xvr.dicom import read_xray
import os
import sys
import shutil
# 添加项目路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
from liftreg.synthetic_data.models import fit_densese3, fit_densexyz, fit_polypose
from liftreg.synthetic_data.utils import KWARGS, get_training_frames, load_dataset

torch.manual_seed(6)

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
def load_xrays(xrays, frames, output_dir, subject_id):
    # Load ground truth X-ray images and camera poses
    imgs = []
    poses = []
    for idx in frames:
        xray, *_ = read_xray(f"{xrays}/{idx:03d}.dcm", crop=100)
        pose, *_ = torch.load(f"{xrays}/{idx:03d}.pt", weights_only=False)["pose"]

        imgs.append(xray)
        poses.append(pose)
    gt = torch.concat(imgs).cuda()
    poses = RigidTransform(torch.stack(poses)).cuda()
    os.makedirs(os.path.join(output_dir, "drr"), exist_ok=True)
    shutil.copy(f"{xrays}/{frames[0]:03d}.pt", os.path.join(output_dir, "drr", f"subject{subject_id:02d}_pose.pt"))
    return gt, poses


def save(subject_id, model_name, volume, model):
    # Save the warped volumes
    affine = ScalarImage(volume).affine
    warped_volume, warped_mask = model.warp_subject(affine=affine)
    warped_volume.save(f"results/{subject_id}/{model_name}_volume.nii.gz")
    warped_mask.save(f"results/{subject_id}/{model_name}_mask.nii.gz")

    # Save the warp and its Jacobian determinant
    torch.save(model, f"results/{subject_id}/{model_name}.ckpt")
    jacdet = (model.warp.jacdet < 0).to(torch.float32).mean().item() * 100
    with open(f"results/{subject_id}/{model_name}.txt", "w") as f:
        f.write(f"{jacdet:.4f}")


# @click.command()
# @click.option("--subject_id", type=click.IntRange(1, 6))
# @click.option("--model_name", type=click.Choice(["polypose", "densexyz", "densese3"]))
def main():
    # Load the required data
    model_name = "polypose"
    output_dir = rf'/home/zzx/extracted_data_shift_only1'
    for subject_id in range(1, 7):
        volume, mask, xrays, _ = load_dataset(subject_id)
        frames = get_training_frames(subject_id)
        gt, poses = load_xrays(xrays, frames, output_dir, subject_id)
        # Run the registration
        if model_name == "polypose":
            img, mask = fit_polypose(gt, poses, volume, mask, KWARGS, subject_id)
        elif model_name == "densexyz":
            model = fit_densexyz(gt, poses, volume, mask, KWARGS)
        elif model_name == "densese3":
            model = fit_densese3(gt, poses, volume, mask, KWARGS)
        else:
            raise ValueError(f"Unrecognized model_name {model_name}")
        print(subject_id, "finished")

    # Save the output
    # Path(f"results/{subject_id}").mkdir(parents=True, exist_ok=True)
    # save(subject_id, model_name, volume, model)


if __name__ == "__main__":
    main()
