import numpy as np
import os


def np_pc_from_bin_file(path, get_extra_args=False, num_args=1):
    """
        Loads a point cloud from a bin file 
        and returns a np.array of shape [N, 3] or [N, 4]

    Args:
         - path: path to the bin file
         - get_extra_args (bool): loading more arguments if provided
                            (i.e. lidar reflectance)
         - num_args: number of extra arguments
                    default one for reflectance
    """

    num_feats = 3 + num_args

    #loading velodyne file
    velo = np.fromfile(path, dtype=np.float32).reshape(-1, num_feats)
        
    if not get_extra_args:
        # keeping only the x, y, z coordinates
        velo = velo[:, :3]

    return velo
