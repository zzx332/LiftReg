# coding=utf-8
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import mermaid.finite_differences as fdt
from torch.autograd import Variable
from math import exp

###############################################################################
# Functions
###############################################################################
class MSELoss(nn.Module):
    """
    A implementation of the mean squared error (MSE)
    """
    def forward(self,input, target):
        return nn.MSELoss(reduction="mean")(input, target)

class NCCLoss(nn.Module):
    """
    A implementation of the normalized cross correlation (NCC)
    """
    def forward(self,input, target):
        input = input.view(input.shape[0], -1)
        target = target.view(target.shape[0], -1)
        input_minus_mean = input - torch.mean(input, 1).view(input.shape[0],1) + 1e-10
        target_minus_mean = target - torch.mean(target, 1).view(input.shape[0],1) + 1e-10
        nccSqr = ((input_minus_mean * target_minus_mean).mean(1)) / torch.sqrt(
                    ((input_minus_mean ** 2).mean(1)) * ((target_minus_mean ** 2).mean(1)))
        nccSqr =  nccSqr.mean()

        assert not torch.isnan(nccSqr), 'NCC loss is Nan.'

        return (1 - nccSqr)

class NCC2D(nn.Module):
    """
    Local (over window) normalized cross correlation loss.
    """
    def __init__(self, win=9):
        super(NCC2D, self).__init__()
        self.win = win

    def forward(self, y_true, y_pred):

        Ii = y_true
        Ji = y_pred

        # get dimension of volume
        # assumes Ii, Ji are sized [batch_size, *vol_shape, nb_feats]
        ndims = len(list(Ii.size())) - 2
        assert ndims == 2, "volumes should be 2 to 2 dimensions. found: %d" % ndims

        # set window size
        win = [self.win] * ndims

        # compute filters
        sum_filt = torch.ones([1, 1, *win]).to(y_true.device)

        pad_no = int(np.floor(win[0] / 2))

        if ndims == 2:
            stride = (1, 1)
            padding = (pad_no, pad_no)
        
        # get convolution function
        conv_fn = getattr(F, 'conv%dd' % ndims)

        # compute CC squares
        I2 = Ii * Ii
        J2 = Ji * Ji
        IJ = Ii * Ji

        I_sum = conv_fn(Ii, sum_filt, stride=stride, padding=padding)
        J_sum = conv_fn(Ji, sum_filt, stride=stride, padding=padding)
        I2_sum = conv_fn(I2, sum_filt, stride=stride, padding=padding)
        J2_sum = conv_fn(J2, sum_filt, stride=stride, padding=padding)
        IJ_sum = conv_fn(IJ, sum_filt, stride=stride, padding=padding)

        win_size = np.prod(win)
        u_I = I_sum / win_size
        u_J = J_sum / win_size

        cross = IJ_sum - u_J * I_sum - u_I * J_sum + u_I * u_J * win_size
        I_var = I2_sum - 2 * u_I * I_sum + u_I * u_I * win_size
        J_var = J2_sum - 2 * u_J * J_sum + u_J * u_J * win_size
        # 数值稳定性增强
        I_var = I_var.clamp_min(0.0)  # 方差不应为负
        J_var = J_var.clamp_min(0.0)
        
        denom = (I_var * J_var).clamp_min(1e-5)  # 确保分母 >= 1e-5
        cc = cross * cross / denom

        return -torch.mean(cc)
        
class NGFLoss(torch.nn.Module):
    def __init__(self):
        super(NGFLoss,self).__init__()
        self.eps = 1e-10
    
    def forward(self, I0, I1):
        g_I0 = self._image_normalized_gradient(I0)
        g_I1 = self._image_normalized_gradient(I1)
        
        sim_per_pix = 1. - torch.mean(torch.bmm(g_I0.view(-1,1,2), g_I1.view(-1,2,1))**2)
        return sim_per_pix

    def _image_normalized_gradient(self, x):
        '''
        :param x. BxCxWxH
        '''
        g_x = F.pad(x[:,:,2:,:] - x[:,:,0:-2,:], (0,0,1,1), "constant", 0)
        g_y = F.pad(x[:,:,:,2:] - x[:,:,:,0:-2], (1,1,0,0), "constant", 0)

        # Use linear conditioin
        g_x[:,:,0:1,:] =  (x[:,:,1:2,:] - x[:,:,0:1,:])
        g_x[:,:,-1:,:] =  (x[:,:,-1:,:] - x[:,:,-2:-1,:])
        g_y[:,:,:,0:1] =  (x[:,:,:,1:2] - x[:,:,:,0:1])
        g_y[:,:,:,-1:] =  (x[:,:,:,-1:] - x[:,:,:,-2:-1])

        g = torch.stack([g_x, g_y], dim=-1)
        g = g/torch.sqrt(torch.sum(g**2, dim=-1, keepdim=True)+self.eps)
        return g


class GradientDifference(nn.Module):
    """Gradient Difference similarity (Penney et al., 1998).

    Designed for 2D-3D X-ray / DRR registration. Instead of comparing
    intensities (which differ across modalities and where one image contains
    structures the other lacks, e.g. contrast-filled vessels in DSA), it
    compares image gradients via

        G = mean( Av / (Av + Dv^2) ) + mean( Ah / (Ah + Dh^2) )

    where Dv = dV(fixed) - s * dV(moving), Dh = dH(fixed) - s * dH(moving),
    and Av, Ah are constants (here the variance of the fixed-image gradients).
    The 1/(A + d^2) form suppresses large gradient differences, making the
    measure robust to structures present in only one modality.

    forward(y_true, y_pred): both [B, 1, H, W]; returns a scalar in ~[-1, 0]
    (lower is better; -1 = perfectly aligned gradients).
    """

    def __init__(self, s=1.0, eps=1e-6):
        super(GradientDifference, self).__init__()
        self.s = s
        self.eps = eps

    @staticmethod
    def _grads(x):
        gh = F.pad(x[:, :, 1:, :] - x[:, :, :-1, :], (0, 0, 0, 1))  # d/dH
        gw = F.pad(x[:, :, :, 1:] - x[:, :, :, :-1], (0, 1, 0, 0))  # d/dW
        return gh, gw

    def forward(self, y_true, y_pred):
        gh_f, gw_f = self._grads(y_true)
        gh_m, gw_m = self._grads(y_pred)

        s = self.s
        if s is None:
            # Adaptive scale matching gradient magnitudes (Penney): s = <g_f, g_m> / <g_m, g_m>
            num = (gh_f * gh_m).sum() + (gw_f * gw_m).sum()
            den = (gh_m * gh_m).sum() + (gw_m * gw_m).sum() + self.eps
            s = (num / den).detach()

        Dv = gw_f - s * gw_m
        Dh = gh_f - s * gh_m

        # Constants A: variance of the fixed-image gradients (per call, stable scalar).
        Av = gw_f.var().clamp_min(self.eps)
        Ah = gh_f.var().clamp_min(self.eps)

        Gv = Av / (Av + Dv * Dv)
        Gh = Ah / (Ah + Dh * Dh)

        return -0.5 * (Gv.mean() + Gh.mean())


class PatternIntensity(nn.Module):
    """Pattern Intensity similarity (Penney et al., 1998 / Weese et al.).

    Operates on the difference image Idiff = fixed - s * moving. Within a
    neighborhood of radius r, it accumulates

        P = mean over offsets d of  sigma^2 / (sigma^2 + (Idiff(x) - Idiff(x+d))^2)

    Aligned images yield a flat difference image (small local variations) and
    thus a high P. The sigma^2/(sigma^2 + .) form makes it robust to structures
    that appear in only one modality (e.g. DSA contrast, bowel gas) because such
    large local differences contribute little. Higher P = better alignment, so
    the returned loss is -P.

    forward(y_true, y_pred): both [B, 1, H, W]; returns a scalar in ~[-1, 0].
    """

    def __init__(self, radius=3, sigma=0.5, s=1.0):
        super(PatternIntensity, self).__init__()
        self.radius = int(radius)
        self.sigma2 = float(sigma) ** 2
        self.s = s
        # Precompute neighborhood offsets within the radius (excluding center).
        offsets = []
        r = self.radius
        for di in range(-r, r + 1):
            for dj in range(-r, r + 1):
                if 0 < di * di + dj * dj <= r * r:
                    offsets.append((di, dj))
        self.offsets = offsets

    def forward(self, y_true, y_pred):
        idiff = y_true - self.s * y_pred  # [B, 1, H, W]
        sigma2 = self.sigma2

        acc = 0.0
        for di, dj in self.offsets:
            # Shift idiff by (di, dj) using pad+crop so there is no wrap-around;
            # the overlapping region is compared, borders are zero-padded which
            # contributes ~1 (aligned) and is negligible for small radius.
            shifted = torch.roll(idiff, shifts=(di, dj), dims=(2, 3))
            d = idiff - shifted
            acc = acc + sigma2 / (sigma2 + d * d)

        pattern = acc / max(len(self.offsets), 1)
        return -pattern.mean()