from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

from nnunetv2.training.nnUNetTrainer.clDice.soft_skeleton import SoftSkeletonize

NUM_ITER = 5

class SoftClDiceLoss(nn.Module):
    def __init__(self, apply_nonlin: Callable = None, batch_dice: bool = False, do_bg: bool = True, smooth = 1.):
        super(SoftClDiceLoss, self).__init__()
        self.soft_skeletonize = SoftSkeletonize(NUM_ITER)

        self.do_bg = do_bg
        self.batch_dice = batch_dice
        self.apply_nonlin = apply_nonlin
        self.smooth = smooth

    def forward(self, x, y, loss_mask=None):
        if self.apply_nonlin is not None:
            x = self.apply_nonlin(x)

        # one-hot encoding
        with torch.no_grad():
            if x.ndim != y.ndim:
                y = y.view((y.shape[0], 1, *y.shape[1:]))

            if x.shape == y.shape:
                # if this is the case then gt is probably already a one hot encoding
                y_onehot = y
            else:
                y_onehot = torch.zeros(x.shape, device=x.device, dtype=torch.float)
                y_onehot.scatter_(1, y.long(), 1)

            if not self.do_bg:
                y_onehot = y_onehot[:, 1:]

        # this one MUST be outside the with torch.no_grad(): context. Otherwise no gradients for you
        if not self.do_bg:
            x = x[:, 1:]

        skel_pred = self.soft_skeletonize(x)
        skel_true = self.soft_skeletonize(y_onehot)

        if loss_mask is not None:
            raise NotImplementedError("loss_mask not supported in clDice")

        # batch dice ignored for now
        # if self.batch_dice:
            # raise NotImplementedError("batch_dice not supported in clDice")

        tprec = (torch.sum(torch.multiply(skel_pred, y_onehot))+self.smooth)/(torch.sum(skel_pred)+self.smooth)    
        tsens = (torch.sum(torch.multiply(skel_true, x))+self.smooth)/(torch.sum(skel_true)+self.smooth)    
        # cl_dice = 1. - 2.0*(tprec*tsens)/(tprec+tsens)
        cl_dice = - 2.0*(tprec*tsens)/(tprec+tsens)
        return cl_dice

class soft_cldice2(nn.Module):
    def __init__(self, iter_=3, smooth = 1., do_background=True):
        super(soft_cldice2, self).__init__()
        self.iter = iter_
        self.smooth = smooth
        self.soft_skeletonize = SoftSkeletonize(NUM_ITER)
        self.do_background = do_background

    def forward(self, x, y):
        with torch.no_grad():
            if x.ndim != y.ndim:
                y = y.view((y.shape[0], 1, *y.shape[1:]))

            if x.shape == y.shape:
                # if this is the case then gt is probably already a one hot encoding
                y_onehot = y
            else:
                y_onehot = torch.zeros(x.shape, device=x.device, dtype=torch.int)
                y_onehot.scatter_(1, y.long(), 1)
        y = y_onehot
        if not self.do_background:
            y = y[:, 1:, :, :]
            x = x[:, 1:, :, :]
        skel_pred = self.soft_skeletonize(x)
        skel_true = self.soft_skeletonize(y)
        tprec = (torch.sum(torch.multiply(skel_pred, y))+self.smooth)/(torch.sum(skel_pred)+self.smooth)    
        tsens = (torch.sum(torch.multiply(skel_true, x))+self.smooth)/(torch.sum(skel_true)+self.smooth)    
        cl_dice = 1.- 2.0*(tprec*tsens)/(tprec+tsens)
        return cl_dice

def soft_dice(x, y):
    """[function to compute dice loss]

    Args:
        y ([float32]): [ground truth image]
        x ([float32]): [predicted image]

    Returns:
        [float32]: [loss value]
    """
    with torch.no_grad():
        if x.ndim != y.ndim:
            y = y.view((y.shape[0], 1, *y.shape[1:]))

        if x.shape == y.shape:
            # if this is the case then gt is probably already a one hot encoding
            y_onehot = y
        else:
            y_onehot = torch.zeros(x.shape, device=x.device, dtype=torch.int)
            y_onehot.scatter_(1, y.long(), 1)

    smooth = 1.
    intersection = torch.sum((y * x))
    coeff = (2. *  intersection + smooth) / (torch.sum(y) + torch.sum(x) + smooth)
    return (1. - coeff)


# class soft_dice_cldice(nn.Module):
#     def __init__(self, iter_=3, alpha=0.5, smooth = 1., exclude_background=False):
#         super(soft_dice_cldice, self).__init__()
#         self.iter = iter_
#         self.smooth = smooth
#         self.alpha = alpha
#         self.soft_skeletonize = SoftSkeletonize(NUM_ITER)
#         self.exclude_background = exclude_background

#     def forward(self, y_pred, y_true):
#         if self.exclude_background:
#             y_true = y_true[:, 1:, :, :]
#             y_pred = y_pred[:, 1:, :, :]
#         dice = soft_dice(y_pred, y_true)
#         skel_pred = self.soft_skeletonize(y_pred)
#         skel_true = self.soft_skeletonize(y_true)
#         tprec = (torch.sum(torch.multiply(skel_pred, y_true))+self.smooth)/(torch.sum(skel_pred)+self.smooth)    
#         tsens = (torch.sum(torch.multiply(skel_true, y_pred))+self.smooth)/(torch.sum(skel_true)+self.smooth)    
#         cl_dice = 1.- 2.0*(tprec*tsens)/(tprec+tsens)
#         return (1.0-self.alpha)*dice+self.alpha*cl_dice

if __name__ == '__main__':
    from nnunetv2.utilities.helpers import softmax_helper_dim1
    x = torch.rand((2, 3, 32, 32, 32)) # torch.Size([2, 3, 32, 32, 32]), BxCxHxWxD -> C is a separate channel for each class (here e.g. 0,1,2)
    y = torch.randint(0, 3, (2, 32, 32, 32)) # torch.Size([2, 32, 32, 32]), BxHxWxD

    x = softmax_helper_dim1(x)

    import time

    loss = SoftClDiceLoss(apply_nonlin=None, batch_dice=False, do_bg=False, smooth=1.)

    start = time.time()
    res = loss(x, y)
    end = time.time()
    print("Time: ", end - start)
    print("soft_cldice: ", res)

    print("dice: ", soft_dice(x, y))
    soft_cldice = soft_cldice2(do_background=False, smooth=1.)
    print("cl_dice: ", soft_cldice(x, y))
    