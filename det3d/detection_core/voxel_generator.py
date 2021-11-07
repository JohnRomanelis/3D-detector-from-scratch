from spconv.utils import VoxelGeneratorV2
import numpy as np
from .coord_sys_utils import cart2cyl_np

def numpyfy(array):
    if not isinstance(array, np.ndarray):
        array = np.array(array)

    return array

class CylidricalVoxelGenerator():


    def __init__(self, 
                 grid_size, 
                 point_cloud_range, 
                 max_num_points):
        
        grid_size = numpyfy(grid_size)
        point_cloud_range = numpyfy(point_cloud_range)

        self.grid_size = grid_size
        self.point_cloud_range = point_cloud_range

        # computing voxel size to match the grid coordinates
        ranges = numpyfy([point_cloud_range[i+3] - point_cloud_range[i] for i in range(3)])
        self.voxel_size = ranges / grid_size

        # Create a voxel generator to generate the 
        self.voxel_generator = VoxelGeneratorV2(self.voxel_size, 
                                                point_cloud_range, 
                                                max_num_points)



    def generate(self, points, max_voxels=None, cylidrical_input=False):
        #print('points shape before ', points.shape)
        # transforming point cloud
        if not cylidrical_input:
            point_num_features = points.shape[-1]
            if point_num_features > 3: # more than xyz coords
                point_features = points[..., 3:]
                # transforming point cloud to cylidrical
                point_coords = cart2cyl_np(points[..., :3])
                points = np.concatenate([point_coords, point_features], axis=-1)
                
            else:
                # has only the x,y,z coordinates
                points = cart2cyl_np(points)

        #print('points shape after  ', points.shape)
        return self.voxel_generator.generate(points, max_voxels)