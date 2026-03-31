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

        cc = cross * cross / (I_var * J_var + 1e-5)

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