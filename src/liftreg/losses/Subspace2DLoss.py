import mermaid.finite_differences as fdt
import numpy as np
import torch
import torch.nn as nn
from ..utils.utils import sigmoid_decay
from ..utils.general import get_class


class loss(nn.Module):
    def __init__(self, opt):
        super(loss, self).__init__()
        self.sim_factor = opt[('sim_factor', 10, 'sim factor')]
        self.sim = get_class(opt["sim_class", 'layers.losses.NCCLoss', 'Similarity class'])()
        self.initial_reg_factor = opt[('initial_reg_factor', 10, 'initial regularization factor')]
        """initial regularization factor"""
        self.min_reg_factor = opt[('min_reg_factor', 1e-3, 'minimum regularization factor')]
        """minimum regularization factor"""
        self.reg_factor_decay_from = opt[('reg_factor_decay_from', 10, 'regularization factor starts to decay from # epoch')]

    def forward(self, input):
        # Parse input data
        warped = input["warped_proj"]
        target = input["target_proj"]
        params = input["params"]
        # pca_coefs = input["pca_coefs"]
        epoch = input["epoch"]
        warped = self.normalize_intensity(warped, linear_clip=True, clip_range=[0, 50])
        target = self.normalize_intensity(target, linear_clip=True, clip_range=[0, 50])
        sim_loss = self.sim(warped, target)
        reg_loss = self.compute_reg_loss(params)
        total_loss = self.sim_factor * sim_loss  + self.get_reg_factor(epoch) * reg_loss
        # total_loss = self.get_reg_factor(epoch) * reg_loss
        outputs = {
            "total_loss": total_loss,
            "sim_loss": sim_loss.item(),
            "reg_loss": reg_loss.item()
        }

        return outputs

    def normalize_intensity(self, img, linear_clip=False, clip_range=None):
        """
        a numpy image, normalize into intensity [-1,1]
        (img-img.min())/(img.max() - img.min())
        :param img: image
        :param linear_clip:  Linearly normalized image intensities so that the 95-th percentile gets mapped to 0.95; 0 stays 0
        :return:
        """

        if linear_clip:
            if clip_range is not None:
                img[img<clip_range[0]] = clip_range[0]
                img[img>clip_range[1]] = clip_range[1]
                normalized_img = (img-clip_range[0]) / (clip_range[1] - clip_range[0]) 
            else:
                img = img - img.min()
                normalized_img =img / np.percentile(img, 95) * 0.95
        else:
            # If we normalize in HU range of softtissue
            min_intensity = img.min()
            max_intensity = img.max()
            normalized_img = (img-img.min())/(max_intensity - min_intensity)
        # normalized_img = normalized_img * 2 - 1
        return normalized_img

    def get_reg_factor(self, epoch):
        """
        get the regularizer factor according to training strategy

        :return:
        """
        decay_factor = 2
        factor_scale = float(
            max(sigmoid_decay(epoch, static=self.reg_factor_decay_from, k=decay_factor) * self.initial_reg_factor , self.min_reg_factor))
        return factor_scale
    
    def compute_reg_loss(self, affine_param):
        disp = affine_param
        spacing = 1. / ( np.array(affine_param.shape[2:]) - 1)
        fd = fdt.FD_torch(spacing*2)
        l2 = fd.dXc(disp[:, 0, ...])**2 +\
            fd.dYc(disp[:, 0, ...])**2 +\
            fd.dZc(disp[:, 0, ...])**2 +\
            fd.dXc(disp[:, 1, ...])**2 +\
            fd.dYc(disp[:, 1, ...])**2 +\
            fd.dZc(disp[:, 1, ...])**2 +\
            fd.dXc(disp[:, 2, ...])**2 +\
            fd.dYc(disp[:, 2, ...])**2 +\
            fd.dZc(disp[:, 2, ...])**2


        reg = torch.mean(l2)
        return reg
