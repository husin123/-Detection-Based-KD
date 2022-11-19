# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..builder import LOSSES


@LOSSES.register_module()
class KLDivergence(nn.Module):
    """A measure of how one probability distribution Q is different from a
    second, reference probability distribution P.

    Args:
        tau (float): Temperature coefficient. Defaults to 1.0.
        reduction (str): Specifies the reduction to apply to the loss:
            ``'none'`` | ``'batchmean'`` | ``'sum'`` | ``'mean'``.
            ``'none'``: no reduction will be applied
            ``'batchmean'``: the sum of the output will be divided by the batchsize
            ``'sum'``: the output will be summed
            ``'mean'``: the output will be divided by the number of elements in the output
            Default: ``'batchmean'``
        loss_weight (float): Weight of loss. Defaults to 1.0.
    """

    def __init__(
        self,
        tau=1.0,
        reduction='batchmean',
        loss_weight=1.0,
        use_sigmoid=False,
    ):
        super(KLDivergence, self).__init__()
        self.tau = tau
        self.loss_weight = loss_weight

        accept_reduction = {'none', 'batchmean', 'sum', 'mean'}
        assert reduction in accept_reduction, \
            f'KLDivergence supports reduction {accept_reduction}, ' \
            f'but gets {reduction}.'
        self.reduction = reduction
        self.sigmoid = use_sigmoid

    def forward(self, preds_S, preds_T):
        """Forward computation.

        Args:
            preds_S (torch.Tensor): The student model prediction with
                shape (N, C, H, W) or shape (N, C).
            preds_T (torch.Tensor): The teacher model prediction with
                shape (N, C, H, W) or shape (N, C).

        Return:
            torch.Tensor: The calculated loss value.
        """
        if preds_T.ndim>2:
            num_classes= preds_S.shape[-1]
            preds_S = preds_S.view(-1,num_classes)
            preds_T = preds_T.view(-1,num_classes)
        preds_T = preds_T.detach()
        if self.sigmoid:
            _preds_S = (preds_S/ self.tau).sigmoid()[...,None]
            _preds_T = (preds_T/ self.tau).sigmoid()[...,None]
            _preds_S = torch.cat([_preds_S,1-_preds_S],-1).view(-1,2)
            _preds_T = torch.cat([_preds_T,1-_preds_T],-1).view(-1,2)
            _loss = = (self.tau**2) * F.kl_div(
            torch.log(_preds_S),
            preds_T,
            reduction=self.reduction)
        else:
            _loss = 0
        softmax_pred_T = F.softmax(preds_T / self.tau, dim=1)
        logsoftmax_preds_S = F.log_softmax(preds_S / self.tau, dim=1)
        loss = (self.tau**2) * F.kl_div(
            logsoftmax_preds_S, 
            softmax_pred_T, 
            reduction=self.reduction) + _loss
        return self.loss_weight * loss
