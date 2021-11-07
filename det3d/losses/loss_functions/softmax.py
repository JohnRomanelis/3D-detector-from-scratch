from .loss_base import Loss

import torch
import torch.nn as nn

def _softmax_cross_entropy_with_logits(logits, labels):
    param = list(range(len(logits.shape)))
    transpose_param = [0] + [param[-1]] + param[1:-1]
    logits = logits.permute(*transpose_param) # [N, ..., C] -> [N, C, ...]
    loss_ftor = nn.CrossEntropyLoss(reduction='none')   # Creation of layer (?) -TODO: FIX 
    loss = loss_ftor(logits, labels.max(dim=-1)[1])
    return loss

class WeightedSoftmaxClassificationLoss(Loss):
    """Softmax loss Function"""

    def __init__(self, logit_scale=1.0):
        """
            Constructor: 

            Args: 
                 - logit_scale: When this value is high, the prediction is "diffused" and
                                when this value is low, the prediction is made peakier.
                                (default 1.0)

        """
        self._logit_scale = logit_scale

    def _compute_loss(self, prediction_tensor, target_tensor, weights):
        """
            Args: 
                 - prediction tensor: [batch_size, num_anchors, num_classes]
                 - target_tensor: [batch_size, num_anchors, num_classes]
                 - weights: representing one-hot encoded classification targets
                            a float tensor of shape [batch_size, num_anchors]

            Returns: 
                 - loss: [batch_size, num_anchors]
        """


        num_classes = prediction_tensor.shape[-1]
        prediction_tensor = torch.div(prediction_tensor, self._logit_scale)

        per_row_cross_ent = (_softmax_cross_entropy_with_logits(
            labels=target_tensor.view(-1, num_classes),
            logits=prediction_tensor.view(-1, num_classes)
        ))
        return per_row_cross_ent.view(weights.shape) * weights