class Loss:
    """
        Abstract base class for a Loss
    """


    def __call__(self,
                 prediction_tensor, 
                 target_tensor, 
                 ignore_nan_targets=False, 
                 scope=None, 
                 **params):
        """
            Call the loss function

        Args: 
             - prediction tensor: an N-d tensor of shape [batch, anchors, ...]
                representing predicted quantities.
             - target_tensor: an N-d tensor of shape [batch, anchors, ...]
                representing regression or classification targets.
             - ignore_nan_targets: whether to ignore nan targets in the loss computation.
                e.g. can be used if the target tensor is missing groundtruth data that
                     shouldn't be factored into the loss.
             - scope: Op scope name. Defaults to 'Loss' if None.
             - **params: Additional keyword arguments for specific implementations of 
                the Loss.
        Returns:
             - loss: a tensor representing the value of the loss function
        """

        if ignore_nan_targets:
            target_tensor = torch.where(torch.isnan(target_tensor),
                                        prediction_tensor,
                                        target_tensor)
        
        return self._compute_loss(prediction_tensor, target_tensor, **params)


    def _compute_loss(self, prediction_tensor, target_tensor, **params):
        """
            Method to be overridden by implementations.

        Args: 
             - prediction_tensor: a tensor representing predicted quantities
             - target_tensor: a tensor representing regression or classification targets
             - **params: Additional keyword arguments for specific implementations of
                  the Loss.
        Returns:
            loss: an N-d tensor of shape [batch, anchors, ...] containing the loss per
                anchor
        
        """
        raise NotImplementedError