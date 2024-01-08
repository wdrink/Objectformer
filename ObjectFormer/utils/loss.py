import torch
import torch.nn as nn
import torch.nn.functional as F

from ObjectFormer.utils.registries import LOSS_REGISTRY


@LOSS_REGISTRY.register()
class CrossEntropyLoss(nn.Module):
    """Cross Entropy Loss.
    Support two kinds of labels and their corresponding loss type. It's worth
    mentioning that loss type will be detected by the shape of ``cls_score``
    and ``label``.
    1) Hard label: This label is an integer array and all of the elements are
        in the range [0, num_classes - 1]. This label's shape should be
        ``cls_score``'s shape with the `num_classes` dimension removed.
    2) Soft label(probablity distribution over classes): This label is a
        probability distribution and all of the elements are in the range
        [0, 1]. This label's shape must be the same as ``cls_score``. For now,
        only 2-dim soft label is supported.
    """

    def __init__(self, loss_cfg):
        super().__init__()
        self.class_weight = (
            torch.Tensor(loss_cfg['CLASS_WEIGHT']).cuda()
            if 'CLASS_WEIGHT' in loss_cfg
            else None
        )

    def forward(self, outputs, samples, **kwargs):
        cls_score = outputs['logits']
        label = samples['bin_label_onehot']
        if cls_score.size() == label.size():
            # calculate loss for soft labels
            assert cls_score.dim() == 2, 'Only support 2-dim soft label'
            lsm = F.log_softmax(cls_score, 1)
            if self.class_weight is not None:
                lsm = lsm * self.class_weight.unsqueeze(0).cuda()
            loss_cls = -(label * lsm).sum(1)

            # default reduction 'mean'
            if self.class_weight is not None:
                loss_cls = loss_cls.sum() / torch.sum(
                    self.class_weight.unsqueeze(0).cuda() * label
                )
            else:
                loss_cls = loss_cls.mean()
        else:
            # calculate loss for hard label
            loss_cls = F.cross_entropy(
                cls_score, label, self.class_weight, **kwargs
            )
        return loss_cls


@LOSS_REGISTRY.register()
class BCELossWithLogits(nn.Module):
    def __init__(self, loss_cfg):
        super().__init__()
        self.loss_cfg = loss_cfg
        self.func = nn.BCEWithLogitsLoss().cuda()

    def forward(self, label, cls_score, **kwargs):
        loss_cls = self.func(
            cls_score, label, **kwargs
        )
        return loss_cls


@LOSS_REGISTRY.register()
class MSELoss(nn.Module):
    def __init__(self, loss_cfg):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, outputs, samples, **kwargs):
        pred_mask = outputs['pred_mask']
        gt_mask = samples['mask']
        loss = self.mse(pred_mask, gt_mask)
        return loss


@LOSS_REGISTRY.register()
class FocalLoss(nn.Module):
    def __init__(self, loss_cfg):
        super().__init__()
        self.alpha = loss_cfg['ALPHA'] if 'ALPHA' in loss_cfg.keys() else 1
        self.gamma = loss_cfg['GAMMA'] if 'GAMMA' in loss_cfg.keys() else 2
        self.logits = (
            loss_cfg['LOGITS'] if 'LOGITS' in loss_cfg.keys() else True
        )
        self.reduce = (
            loss_cfg['REDUCE'] if 'REDUCE' in loss_cfg.keys() else True
        )

    def forward(self, label, cls_score, **kwargs):
        if self.logits:
            BCE_loss = F.binary_cross_entropy_with_logits(
                cls_score, label, reduction='none'
            )
        else:
            BCE_loss = F.binary_cross_entropy(
                cls_score, label, reduction='none'
            )
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss

        if self.reduce:
            return torch.mean(F_loss)
        else:
            return F_loss



@LOSS_REGISTRY.register()
class DiceLoss(nn.Module):
    def __init__(self, loss_cfg):
        super().__init__()
        smooth = loss_cfg.get("smooth", 1.0)
        self.smooth = smooth
    
    def forward(self, pred, gt):
        gt = gt.view(-1)
        pred = pred.view(-1)

        intersection = (gt * pred).sum()
        dice = (2.0 * intersection + self.smooth) / (torch.square(gt).sum() + torch.square(pred).sum() + self.smooth)

        return 1.0 - dice