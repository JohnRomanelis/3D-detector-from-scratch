from .loss_base import Loss

import torch

class WeightedSmoothL1LocalizationLoss(Loss):
    """
        Smooth L1 localization loss function.

        The smooth L1_loss is defined elementwise as: 
               / .5 x^2   ,  if |x| < 1
               \ |x| - .5 ,  otherwise
            - where x is the difference between predictions and target.

       *See also Equation (3) in the Fast R-CNN paper by Ross Girshick (ICCV 2015)
    """

    def __init__(self, sigma=3.0, code_weights=None, codewise=True):
        super().__init__()

        self._sigma = sigma
        if code_weights is not None:
            self._code_weights = np.array(code_weights, dtype=float32)
            self._code_weights = torch.from_numpy(self._code_weights)
        else:
            self._code_weights = None
        self._codewise = codewise

    def _compute_loss(self, prediction_tensor, target_tensor, weights=None):
        """
            Compute loss function

            Args: 
                 - prediction_tensor: [batch_size, num_anchors, code_size]
                 - target_tensor: [batch_size, num_anchors, code_size]
                 - weights : [batch_size, num_anchors]

            Returns:
                 - [batch_size, num_anchors]

        """

        diff = prediction_tensor - target_tensor
        if self._code_weights is not None:
            code_weights = self._code_weights.type_as(prediction_tensor).to(target_tensor.device)
            diff = code_weights.view(1, 1, -1) * diff
        abs_diff = torch.abs(diff)
        # torch.le (less equal): computes input <= other element-wise, returns boolean tensor
        abs_diff_lt_1 = torch.le(abs_diff, 1 / (self._sigma**2)).type_as(abs_diff)
        loss = abs_diff_lt_1 * 0.5 * torch.pow(abs_diff * self._sigma, 2) \
            + (abs_diff - 0.5 / (self._sigma**2)) * (1. - abs_diff_lt_1)
        if self._codewise:
            anchorwise_smooth_l1norm = loss
            if weights is not None:
                anchorwise_smooth_l1norm *= weights.unsqueeze(-1)
        else:
            anchorwise_smooth_l1norm = torch.sum(loss, 2)
            if weights is not None:
                anchorwise_smooth_l1norm *= weights
        return anchorwise_smooth_l1norm


