import torch
import numpy as np

from enum import Enum
class LossNormType(Enum):
    NormByNumPositives = "norm_by_num_positives"
    NormByNumExamples = "norm_by_num_examples"
    NormByNumPosNeg = "norm_by_num_pos_neg"
    DontNorm = "dont_norm"


def prepare_loss_weights(labels, 
                         pos_cls_weight=1.0, 
                         neg_cls_weight=1.0,
                         loss_norm_type=LossNormType.NormByNumPositives, 
                         dtype=torch.float32):
    """
    DESC: 
        get  - classification (cls_weights) and 
             - regression (reg_weights) 
        weights from labels
    """

    cared = labels >= 0 # -1 = DontCare
    # cared: [N, num_anchors]
    positives = labels > 0
    negatives = labels == 0

    negative_cls_weights = negatives.type(dtype) * neg_cls_weight
    cls_weights = negative_cls_weights + pos_cls_weight * positives.type(dtype)
    reg_weights = positives.type(dtype)

    if loss_norm_type == LossNormType.NormByNumExamples:
        num_examples = cared.type(dtype).sum(1, keepdim=True)
        num_examples = torch.clamp(num_examples, min=1.0)
        cls_weights /= num_examples
        bbox_normalizer = positives.sum(1, keepdim=True).type(dtype)
        reg_weights /= torch.clamp(bbox_normalizer, min=1.0)
    elif loss_norm_type == LossNormType.NormByNumPositives:  # for focal loss
        pos_normalizer = positives.sum(1, keepdim=True).type(dtype)
        reg_weights /= torch.clamp(pos_normalizer, min=1.0)
        cls_weights /= torch.clamp(pos_normalizer, min=1.0)
    elif loss_norm_type == LossNormType.NormByNumPosNeg:
        pos_neg = torch.stack([positives, negatives], dim=-1).type(dtype)
        normalizer = pos_neg.sum(1, keepdim=True) # [N, 1, 2]
        cls_normalizer = (pos_neg * normalizer).sum(-1)  # [N, M]
        cls_normalizer = torch.clamp(cls_normalizer, min=1.0)
        normalizer = torch.clamp(normalizer, min=1.0)
        reg_weights /= normalizer[:, 0:1, 0]
        cls_weights /= cls_normalizer
    elif loss_norm_type == LossNormType.DontNorm:  # support ghm loss
        pos_normalizer = positives.sum(1, keepdim=True).type(dtype)
        reg_weights /= torch.clamp(pos_normalizer, min=1.0)
    else:
        raise ValueError(
            f"unknown loss norm type. available: {list(LossNormType)}")
    return cls_weights, reg_weights, cared



def one_hot(tensor, depth, dim=-1, on_value=1.0, dtype=torch.float32):
    tensor_onehot = torch.zeros(
        *list(tensor.shape), depth, dtype=dtype, device=tensor.device)
    tensor_onehot.scatter_(dim, tensor.unsqueeze(dim).long(), on_value)
    return tensor_onehot


def add_sin_difference(boxes1, boxes2, boxes1_rot, boxes2_rot, factor=1.0):
    if factor != 1.0:
        boxes1_rot = factor * boxes1_rot
        boxes2_rot = factor * boxes2_rot

    rad_pred_encoding = torch.sin(boxes1_rot) * torch.cos(boxes2_rot)
    rad_tg_encoding = torch.cos(boxes1_rot) * torch.sin(boxes2_rot)
    
    boxes1 = torch.cat([boxes1[..., :6], rad_pred_encoding, boxes1[..., 7:]],
                       dim=-1)
    boxes2 = torch.cat([boxes2[..., :6], rad_tg_encoding, boxes2[..., 7:]],
                       dim=-1)
    return boxes1, boxes2

def create_loss(loc_loss_ftor, 
                cls_loss_ftor, 
                box_preds, 
                cls_preds,
                cls_targets,
                cls_weights,
                reg_targets,
                reg_weights,
                num_class,
                encode_background_as_zeros=True,
                encode_rad_error_by_sin=True,
                sin_error_factor=1.0,
                box_code_size=7,
                num_direction_bins=2):
    
    batch_size = int(box_preds.shape[0])
    
    # Formating output to match the anchors format
    #box_preds = box_preds.view(batch_size, -1, box_code_size)
    box_preds = box_preds.reshape(*list(box_preds.shape)[:-1], 2, 7)
    box_preds = box_preds.permute(0, 3, 1, 2, 4).reshape(batch_size, -1, box_code_size)
    
    # no tuple
    # print cls shape
    #NOTE: see note on target_assigner.generate_anchors
    #print("allala ",cls_preds.shape)
    cls_preds = cls_preds.permute(0, 3, 1, 2)
    if encode_background_as_zeros:
        #cls_preds = cls_preds.view(batch_size, -1, num_class)
        cls_preds = cls_preds.reshape(batch_size, -1, num_class)
    else:
        cls_preds = cls_preds.view(batch_size, -1, num_class+1)
        
    
    '''
    cls_targets = cls_targets.squeeze(-1)
    one_hot_targets = one_hot(cls_targets, depth=num_class + 1, dtype=box_preds.dtype)
    if encode_background_as_zeros:
        one_hot_targets = one_hot_targets[..., 1:]
    '''
    if encode_rad_error_by_sin:
        box_preds, reg_targets = add_sin_difference(box_preds, reg_targets,
                                                    box_preds[..., 6:7], reg_targets[..., 6:7],
                                                    sin_error_factor)
    


    loc_losses = loc_loss_ftor(
        box_preds, reg_targets, weights=reg_weights) # [N, M]
    
    cls_losses = cls_loss_ftor(
        cls_preds, cls_targets, weights=cls_weights) # [N, M]
    
    #cls_losses = cls_loss_ftor(
    #    cls_preds, one_hot_targets, weights=cls_weights) # [N, M]

    return loc_losses, cls_losses




def get_pos_neg_loss(cls_loss, labels):
    # cls_loss: [N, num_anchors, num_class]
    # labels: [N, num_anchors]

    batch_size = cls_loss.shape[0]

    if cls_loss.shape[-1] == 1 or len(cls_loss.shape)==2:
        cls_pos_loss = (labels > 0).type_as(cls_loss) * cls_loss.view(batch_size, -1)
        cls_neg_loss = (labels == 0).type_as(cls_loss) * cls_loss.view(batch_size, -1)
        cls_pos_loss = cls_pos_loss.sum() / batch_size
        cls_neg_loss = cls_neg_loss.sum() / batch_size
    else:
        cls_pos_loss = cls_loss[..., 1:].sum() / batch_size
        cls_neg_loss = cls_loss[..., 0].sum() / batch_size
    return cls_pos_loss, cls_neg_loss


def limit_period(val, offset=0.5, period=np.pi):
    return val - torch.floor(val / period + offset) * period

def get_direction_target(anchors,
                         reg_targets,
                         use_one_hot=True,
                         dir_offset=0,
                         num_bins=2):

    batch_size = reg_targets.shape[0]
    anchors = anchors.view(batch_size, -1, anchors.shape[-1])
    rot_gt = reg_targets[..., 6] + anchors[..., 6]
    offset_rot = limit_period(rot_gt - dir_offset, 0, 2 * np.pi)
    dir_cls_targets = torch.floor(offset_rot / (2 * np.pi / num_bins)).long()
    dir_cls_targets = torch.clamp(dir_cls_targets, min=0, max=num_bins - 1)
    if use_one_hot:
        dir_cls_targets = one_hot(
            dir_cls_targets, num_bins, dtype=anchors.dtype)
    return dir_cls_targets