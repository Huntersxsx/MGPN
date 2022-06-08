import torch
import torch.nn.functional as F
import pdb


def bce_rescale_loss(scores, masks, targets, cfg):
    min_iou, max_iou, bias = cfg.MIN_IOU, cfg.MAX_IOU, cfg.BIAS
    joint_prob = torch.sigmoid(scores) * masks.float()
    target_prob = ((targets - min_iou) / (max_iou - min_iou)).clamp(0, 1)
    loss_value = F.binary_cross_entropy_with_logits(scores.masked_select(masks.byte()), target_prob.masked_select(masks.byte()))

    return loss_value, joint_prob



# def bce_rescale_loss(scores, masks, targets, coarse_score, cfg):
#     min_iou, max_iou, bias = cfg.MIN_IOU, cfg.MAX_IOU, cfg.BIAS
#     joint_prob = torch.sigmoid(scores) * masks.float()
#     target_prob = ((targets - min_iou) / (max_iou - min_iou)).clamp(0, 1)
#     loss_value = F.binary_cross_entropy_with_logits(scores.masked_select(masks.byte()), target_prob.masked_select(masks.byte()))

#     joint_prob2 = torch.sigmoid(coarse_score) * masks.float()
#     target_prob = ((targets - min_iou) / (max_iou - min_iou)).clamp(0, 1)
#     loss_value2 = F.binary_cross_entropy_with_logits(scores.masked_select(masks.byte()), target_prob.masked_select(masks.byte()))
#     loss_value = 5 * loss_value + loss_value2

#     return loss_value, joint_prob