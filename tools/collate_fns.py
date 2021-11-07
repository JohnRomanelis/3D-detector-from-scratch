import numpy as np 
from collections import defaultdict

def sparse_collate_fn(batch_list):
    """
    The torch dataloader returns a list of the examples contained in a batch.
    This function collates the example to be processed as a batch by the network.
    """

    # if an invalid key is used it will return an empty list
    # instead of raising a KeyError
    example_merged = defaultdict(list)
    

    # instead of having a list of dictionaries
    # we create a dictionary containing a list for each key
    for example in batch_list:
        for k, v in example.items():
            example_merged[k].append(v)

    # creating an empty dictionary to return the collated data
    ret = {}

    for key, elems in example_merged.items():

        if key in [
            'voxels', 'num_points', 'num_gt', 'voxel_labels',
            'gt_names', 'gt_labels', 'gt_boxes'
        ]:
            # concatenating the elements of each key to a single np.array
            # eg list of np.arrays with gt_names -> np.array with gt_names
            ret[key] = np.concatenate(elems, axis=0)
        elif key == 'metadata':
            ret[key] = elems # remains a list of metadata
        elif key == 'calib': # merging calib data of each frame
            # creating a new dict to store them
            ret[key] = {}
            # for the calibration file of each frame
            for elem in elems:
                # for every calib_key, actual_values
                for k1, v1 in elem.items():
                    # if this key is not yet in the ret dict
                    if k1 not in ret[key]:
                        # create a new dictionary entry, with the current key
                        # and add the value as a list containing only that item
                        ret[key][k1] = [v1]
                    # else (aka the list already exists)
                    else:
                        # add the value to the list
                        ret[key][k1].append(v1)
            # for each calibation key
            for k1, v1 in ret[key].items():
                # stacking the list elements to a numpy array
                ret[key][k1] = np.stack(v1, axis=0)

        elif key == 'coordinates':
            coors = []
            for i, coor in enumerate(elems):
                # NOTE: batch index is in the first dimension 
                #       (padding used for spconv and Minkowski Engine)
                coor_pad = np.pad(
                # coor matrix to pad
                # (0, 0), (1, 0) how to pad : (top, down), (left, right)
                # mode = 'constant' : pads with a constant value
                # constant_value : what value to use for padding
                # i indicates which element of the batch_list is used
                    coor, ((0, 0), (1, 0)), mode='constant', constant_values=i
                )
                coors.append(coor_pad)
            # stacking coors list to a np array             
            ret[key] = np.concatenate(coors, axis=0)


        elif key == 'points':
            pointed_batched = []
            for i, p in enumerate(elems):
                # NOTE: batch index is in the last dimension
                #       (padding used for torchsparse)
                points_pad = np.pad(
                    p, ((0, 0), (0, 1)), mode='constant', constant_values=i
                )

                pointed_batched.append(points_pad)

            # stacking points to a tensor
            ret[key] = np.concatenate(pointed_batched, axis=0)

        elif key == 'metrics':
            ret[key] = elems
        
        else:
            ret[key] = np.stack(elems, axis=0)

    return ret


    