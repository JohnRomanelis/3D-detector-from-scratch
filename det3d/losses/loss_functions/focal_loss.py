from .loss_base import Loss

import torch

class SigmoidFocalClassificationLoss(Loss):
    """ Sigmoid focal cross entropy loss.

    Focal loss down-weights well classified examples and focusses on the 
    hard examples. See https://arxiv.org/pdf/1708.02002.pdf for the loss definition.

    """

    def __init__(self, gamma=2.0, alpha=0.25):
        """
            Constructor.

        Args: 
             - gamma: exponent of the modulating factor (1 - p_t) ^ gamma
             - alpha: optional alpha weighting factor to balance positives vs negatives
             - all_zero_negatives: bool, if True, will treat all zero as background.
                                        else, will treat first label as backround.
                                        only affect alpha
        """

        self._alpha = alpha
        self._gamma = gamma


    def _compute_loss(self,
                      prediction_tensor,
                      target_tensor,
                      weights,
                      class_indices=None):

        """

        Args: 
            - prediction_tensor: [batch_size, num_anchors, num_classes]
            - target_tensor: [batch_size, num_anchors, num_classes]
            - weights: [batch_size, num_anchors]
            - class_indices: (Optional) A 1-D integer tensor of class indices.
                If provided, computes loss only for the specified class indices. 

        Returns:
             - loss: [batch_size, num_anchors, num_classes]

        """

        weights = weights.unsqueeze(2)
        ''' 
        ## TODO:
        #if class_indices is not None:
        #    weights *= indices_to_dense_vector(class_indices,
        #    prediction_tensor.shape[2]).view(1, 1, -1).type_as(prediction_tensor)
        #Code from SECOND implementation
        per_entry_cross_ent =  (_sigmoid_cross_entropy_with_logits(
                    labels=target_tensor, logits=prediction_tensor))

        prediction_probabilities = torch.sigmoid(prediction_tensor)

        p_t = ((target_tensor * prediction_probabilities) + 
                ((1 - target_tensor) * (1 - prediction_probabilities)))
        
        modulating_factor = 1.0
        if self._gamma:
            modulating_factor = torch.pow(1.0 - p_t, self._gamma)
        alpha_weight_factor = 1.0
        if self._alpha:
            alpha_weight_factor = (target_tensor * self._alpha + 
                                        (1 - target_tensor) * (1 - self._alpha))

        focal_cross_entropy_loss = (modulating_factor * alpha_weight_factor 
                                                        * per_entry_cross_ent)

        return focal_cross_entropy_loss * weights
        '''

        # getting network prediction probability
        pred_probs = torch.sigmoid(prediction_tensor)

        pred_probs = pred_probs * target_tensor + (1-pred_probs) * (1-target_tensor)

        alpha = self._alpha * target_tensor + (1-self._alpha) * (1-target_tensor)

        loss = - alpha * torch.pow(1 - pred_probs, self._gamma) * torch.log(pred_probs)

        return loss * weights