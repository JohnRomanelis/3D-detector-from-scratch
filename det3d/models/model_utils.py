import torch

def get_paddings_indicator(actual_num, max_num, axis=0):
    """
        
    Source: SECOND Detector 
            (https://github.com/traveller59/second.pytorch.git)

    Description: Creates a boolean mask indicating 
                the original values of a padded tensor 

    Args:
        - actual_num:   number of actual values (non-zero)
                            *type: tensor
                            *shape: [N, 1]
        - max_num:      maximum number of elements per raw
                            *type: scallar

    Returns:

    """

    # creating a "last" dimension
    actual_num = torch.unsqueeze(actual_num, axis+1)
    # actual_num shape: [N, M, 1]
    max_num_shape = [1] * len(actual_num.shape)
    max_num_shape[axis+1] = -1
    max_num = torch.arange(
        max_num, dtype=torch.int, device = actual_num.device
    ).view(max_num_shape)
    # tiled_actual_num: [[3,3,3,3,3], [4,4,4,4,4], [2,2,2,2,2]]
    # tiled_max_num: [[0,1,2,3,4], [0,1,2,3,4], [0,1,2,3,4]]
    paddings_indicator = actual_num.int() > max_num
    # padding_indicator shape: [batch_size, max_num]
    return paddings_indicator