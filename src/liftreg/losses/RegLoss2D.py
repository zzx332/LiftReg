import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from ..utils.general import get_class

class Grad2D(nn.Module):
    """
    N-D gradient loss.
    """
    def __init__(self, penalty='l2', loss_mult=None):
        super(Grad2D, self).__init__()
        self.penalty = penalty
        self.loss_mult = loss_mult

    def forward(self, y_pred, y_true=None):
        dy = torch.abs(y_pred[:, :, 1:, :] - y_pred[:, :, :-1, :])
        dx = torch.abs(y_pred[:, :, :, 1:] - y_pred[:, :, :, :-1])

        if self.penalty == 'l2':
            dy = dy * dy
            dx = dx * dx

        d = torch.mean(dx) + torch.mean(dy)
        grad = d / 2.0

        if self.loss_mult is not None:
            grad *= self.loss_mult
        return grad


class loss(nn.Module):
    """
    Combined loss for 2D registration.
    """
    def __init__(self, opt):
        super(loss, self).__init__()
        self.sim_loss = get_class(opt["sim_class", 'layers.losses.NCC2D', 'Similarity class'])()
        self.reg_loss = Grad2D(penalty='l2')
        self.sim_factor = opt[('sim_factor', 1.0, 'similarity factor')]
        self.reg_factor = opt[('reg_factor', 0.1, 'regularization factor')]
        self.use_dice = opt[('use_dice', False, 'use dice loss')]
        
    def dice_loss(self, y_true, y_pred):
        intersection = torch.sum(y_true * y_pred)
        union = torch.sum(y_true) + torch.sum(y_pred)
        return 1.0 - (2. * intersection + 1e-5) / (union + 1e-5)

    def forward(self, output):
        # output dict comes from RegNet2D
        # which outputs: warped_moving, target_proj, phi
        # and optionally: velocity (SVF mode), polyrigid_params, warped_label

        target = output['target_proj']
        warped_moving = output['warped_moving']
        phi = output['phi']
        _, _, H, W = phi.shape
        # 与 align_corners=True 的归一化一致
        # phi_norm_y = 2.0 * phi[:, 0:1] / max(H - 1, 1)
        # phi_norm_x = 2.0 * phi[:, 1:2] / max(W - 1, 1)
        # phi_norm = torch.cat([phi_norm_y, phi_norm_x], dim=1)
        # For SVF mode, regularize the velocity field (phi is automatically smoothed by integration)
        field_for_reg = output.get('velocity', phi)
        loss_sim = self.sim_loss(target, warped_moving) * self.sim_factor
        loss_reg = self.reg_loss(field_for_reg) * self.reg_factor
        
        total_loss = loss_sim + loss_reg
        losses = {
            'sim_loss': loss_sim,
            'reg_loss': loss_reg,
            'total_loss': total_loss
        }
        
        if self.use_dice and 'warped_label' in output and 'target_label' in output:
            loss_dice = self.dice_loss(output['target_label'], output['warped_label'])
            losses['dice_loss'] = loss_dice
            losses['total_loss'] += loss_dice
            
        return losses
