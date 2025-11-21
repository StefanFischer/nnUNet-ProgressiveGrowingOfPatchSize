import torch
from nnunetv2.training.loss.dice import SoftDiceLoss, MemoryEfficientSoftDiceLoss
from nnunetv2.training.loss.robust_ce_loss import RobustCrossEntropyLoss, TopKLoss
from nnunetv2.utilities.helpers import softmax_helper_dim1
from torch import nn


class DC_and_CE_loss(nn.Module):
    def __init__(self, soft_dice_kwargs, ce_kwargs, weight_ce=1, weight_dice=1, ignore_label=None,
                 dice_class=SoftDiceLoss):
        """
        Weights for CE and Dice do not need to sum to one. You can set whatever you want.
        :param soft_dice_kwargs:
        :param ce_kwargs:
        :param aggregate:
        :param square_dice:
        :param weight_ce:
        :param weight_dice:
        """
        super(DC_and_CE_loss, self).__init__()
        if ignore_label is not None:
            ce_kwargs['ignore_index'] = ignore_label

        self.weight_dice = weight_dice
        self.weight_ce = weight_ce
        self.ignore_label = ignore_label

        self.ce = RobustCrossEntropyLoss(**ce_kwargs)
        self.dc = dice_class(apply_nonlin=softmax_helper_dim1, **soft_dice_kwargs)

    def forward(self, net_output: torch.Tensor, target: torch.Tensor):
        """
        target must be b, c, x, y(, z) with c=1
        :param net_output:
        :param target:
        :return:
        """
        if self.ignore_label is not None:
            assert target.shape[1] == 1, 'ignore label is not implemented for one hot encoded target variables ' \
                                         '(DC_and_CE_loss)'
            mask = target != self.ignore_label
            # remove ignore label from target, replace with one of the known labels. It doesn't matter because we
            # ignore gradients in those areas anyway
            target_dice = torch.where(mask, target, 0)
            num_fg = mask.sum()
        else:
            target_dice = target
            mask = None

        dc_loss = self.dc(net_output, target_dice, loss_mask=mask) \
            if self.weight_dice != 0 else 0
        ce_loss = self.ce(net_output, target[:, 0]) \
            if self.weight_ce != 0 and (self.ignore_label is None or num_fg > 0) else 0

        result = self.weight_ce * ce_loss + self.weight_dice * dc_loss
        return result


class DC_and_BCE_loss(nn.Module):
    def __init__(self, bce_kwargs, soft_dice_kwargs, weight_ce=1, weight_dice=1, use_ignore_label: bool = False,
                 dice_class=MemoryEfficientSoftDiceLoss):
        """
        DO NOT APPLY NONLINEARITY IN YOUR NETWORK!

        target mut be one hot encoded
        IMPORTANT: We assume use_ignore_label is located in target[:, -1]!!!

        :param soft_dice_kwargs:
        :param bce_kwargs:
        :param aggregate:
        """
        super(DC_and_BCE_loss, self).__init__()
        if use_ignore_label:
            bce_kwargs['reduction'] = 'none'

        self.weight_dice = weight_dice
        self.weight_ce = weight_ce
        self.use_ignore_label = use_ignore_label

        self.ce = nn.BCEWithLogitsLoss(**bce_kwargs)
        self.dc = dice_class(apply_nonlin=torch.sigmoid, **soft_dice_kwargs)

    def forward(self, net_output: torch.Tensor, target: torch.Tensor):
        if self.use_ignore_label:
            # target is one hot encoded here. invert it so that it is True wherever we can compute the loss
            if target.dtype == torch.bool:
                mask = ~target[:, -1:]
            else:
                mask = (1 - target[:, -1:]).bool()
            # remove ignore channel now that we have the mask
            # why did we use clone in the past? Should have documented that...
            # target_regions = torch.clone(target[:, :-1])
            target_regions = target[:, :-1]
        else:
            target_regions = target
            mask = None

        dc_loss = self.dc(net_output, target_regions, loss_mask=mask)
        target_regions = target_regions.float()
        if mask is not None:
            ce_loss = (self.ce(net_output, target_regions) * mask).sum() / torch.clip(mask.sum(), min=1e-8)
        else:
            ce_loss = self.ce(net_output, target_regions)
        result = self.weight_ce * ce_loss + self.weight_dice * dc_loss
        return result


class DC_and_topk_loss(nn.Module):
    def __init__(self, soft_dice_kwargs, ce_kwargs, weight_ce=1, weight_dice=1, ignore_label=None):
        """
        Weights for CE and Dice do not need to sum to one. You can set whatever you want.
        :param soft_dice_kwargs:
        :param ce_kwargs:
        :param aggregate:
        :param square_dice:
        :param weight_ce:
        :param weight_dice:
        """
        super().__init__()
        if ignore_label is not None:
            ce_kwargs['ignore_index'] = ignore_label

        self.weight_dice = weight_dice
        self.weight_ce = weight_ce
        self.ignore_label = ignore_label

        self.ce = TopKLoss(**ce_kwargs)
        self.dc = SoftDiceLoss(apply_nonlin=softmax_helper_dim1, **soft_dice_kwargs)

    def forward(self, net_output: torch.Tensor, target: torch.Tensor):
        """
        target must be b, c, x, y(, z) with c=1
        :param net_output:
        :param target:
        :return:
        """
        if self.ignore_label is not None:
            assert target.shape[1] == 1, 'ignore label is not implemented for one hot encoded target variables ' \
                                         '(DC_and_CE_loss)'
            mask = (target != self.ignore_label).bool()
            # remove ignore label from target, replace with one of the known labels. It doesn't matter because we
            # ignore gradients in those areas anyway
            target_dice = torch.clone(target)
            target_dice[target == self.ignore_label] = 0
            num_fg = mask.sum()
        else:
            target_dice = target
            mask = None

        dc_loss = self.dc(net_output, target_dice, loss_mask=mask) \
            if self.weight_dice != 0 else 0
        ce_loss = self.ce(net_output, target) \
            if self.weight_ce != 0 and (self.ignore_label is None or num_fg > 0) else 0

        result = self.weight_ce * ce_loss + self.weight_dice * dc_loss
        return result


import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


BINARY_MODE: str = "binary"
MULTICLASS_MODE: str = "multiclass"
MULTILABEL_MODE: str = "multilabel"
EPS: float = 1e-10

def expand_onehot_labels(labels, target_shape, ignore_index):
    """Expand onehot labels to match the size of prediction."""
    bin_labels = labels.new_zeros(target_shape)
    valid_mask = (labels >= 0) & (labels != ignore_index)
    inds = torch.nonzero(valid_mask, as_tuple=True)

    if inds[0].numel() > 0:
        if labels.dim() == 3:
            bin_labels[inds[0], labels[valid_mask], inds[1], inds[2]] = 1
        elif labels.dim() == 4:
            bin_labels[inds[0], labels[valid_mask], inds[1], inds[2], inds[3]] = 1
        else:
            bin_labels[inds[0], labels[valid_mask]] = 1

    return bin_labels, valid_mask


def get_region_proportion(x: torch.Tensor, valid_mask: torch.Tensor = None) -> torch.Tensor:
    """Get region proportion
    Args:
        x : one-hot label map/mask
        valid_mask : indicate the considered elements
    """
    if valid_mask is not None:
        x = torch.einsum("bcxyz,bxyz->bcxyz", x, valid_mask)
        cardinality = torch.einsum("bxyz->b", valid_mask).unsqueeze(dim=1).repeat(1, x.shape[1])
        # if valid_mask.dim() == 4:
        #     x = torch.einsum("bcwh, bcwh->bcwh", x, valid_mask)
        #     cardinality = torch.einsum("bcwh->bc", valid_mask)
        # else:
        #     x = torch.einsum("bcwh,bwh->bcwh", x, valid_mask)
        #     cardinality = torch.einsum("bwh->b", valid_mask).unsqueeze(dim=1).repeat(1, x.shape[1])
    else:
        cardinality = x.shape[2] * x.shape[3] * x.shape[4]

    # region_proportion = (torch.einsum("bcxyz->bc", x) + EPS) / (cardinality + EPS)
    region_proportion = (torch.sum(x, dim=(2, 3, 4)) + EPS) / (cardinality + EPS)

    return region_proportion


class CompoundLoss(nn.Module):
    """
    The base class for implementing a compound loss:
        l = l_1 + alpha * l_2
    """
    def __init__(self, mode: str,
                 alpha: float = 1.,
                 factor: float = 1.,
                 step_size: int = 0,
                 max_alpha: float = 100.,
                 temp: float = 1.,
                 ignore_index: int = 255,
                 background_index: int = -1,
                 weight: Optional[torch.Tensor] = None) -> None:
        assert mode in {BINARY_MODE, MULTILABEL_MODE, MULTICLASS_MODE}
        super().__init__()
        self.mode = mode
        self.alpha = alpha
        self.max_alpha = max_alpha
        self.factor = factor
        self.step_size = step_size
        self.temp = temp
        self.ignore_index = ignore_index
        self.background_index = background_index
        self.weight = weight

    def cross_entropy(self, inputs: torch.Tensor, labels: torch.Tensor):
        if len(labels.shape) == len(inputs.shape):
            assert labels.shape[1] == 1
            labels = labels[:, 0]
        if self.mode == MULTICLASS_MODE:
            loss = F.cross_entropy(
                inputs, labels.long(), weight=self.weight, ignore_index=self.ignore_index)
        else:
            if labels.dim() == 3:
                labels = labels.unsqueeze(dim=1)
            loss = F.binary_cross_entropy_with_logits(inputs, labels.type(torch.float32))
        return loss

    def adjust_alpha(self, epoch: int) -> None:
        if self.step_size == 0:
            return
        if (epoch + 1) % self.step_size == 0:
            curr_alpha = self.alpha
            self.alpha = min(self.alpha * self.factor, self.max_alpha)
            print(
                "CompoundLoss : Adjust the tradoff param alpha : {:.3g} -> {:.3g}".format(curr_alpha, self.alpha)
            )

    def get_gt_proportion(self, mode: str,
                          labels: torch.Tensor,
                          target_shape,
                          ignore_index: int = 255):
        if mode == MULTICLASS_MODE:
            bin_labels, valid_mask = expand_onehot_labels(labels, target_shape, ignore_index)
        else:
            valid_mask = (labels >= 0) & (labels != ignore_index)
            if labels.dim() == 3:
                labels = labels.unsqueeze(dim=1)
            bin_labels = labels
        gt_proportion = get_region_proportion(bin_labels, valid_mask)
        return gt_proportion, valid_mask

    def get_pred_proportion(self, mode: str,
                            logits: torch.Tensor,
                            temp: float = 1.0,
                            valid_mask=None):
        if mode == MULTICLASS_MODE:
            preds = F.log_softmax(temp * logits, dim=1).exp()
        else:
            preds = F.logsigmoid(temp * logits).exp()
        pred_proportion = get_region_proportion(preds, valid_mask)
        return pred_proportion


class CrossEntropyWithL1(CompoundLoss):
    """
    Cross entropy loss with region size priors measured by l1.
    The loss can be described as:
        l = CE(X, Y) + alpha * |gt_region - prob_region|
    """
    def forward(self, inputs: torch.Tensor, labels: torch.Tensor):
        # ce term
        if len(labels.shape) == len(inputs.shape):
            assert labels.shape[1] == 1
            labels = labels[:, 0]
        labels = labels.long()

        loss_ce = self.cross_entropy(inputs, labels)
        # regularization
        gt_proportion, valid_mask = self.get_gt_proportion(self.mode, labels, inputs.shape)
        pred_proportion = self.get_pred_proportion(self.mode, inputs, temp=self.temp, valid_mask=valid_mask)
        loss_reg = (pred_proportion - gt_proportion).abs().mean()

        loss = loss_ce + self.alpha * loss_reg

        # return loss, loss_ce, loss_reg
        return loss

