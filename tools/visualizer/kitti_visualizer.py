import open3d as o3d 
import numpy as np
from pathlib import Path
from abc import ABCMeta

from tools.string_ops import int2constString
from tools.loading_ops import np_pc_from_bin_file
from .visualizer_utils import KittiScene


class Visualizer(metaclass=ABCMeta):
    """

        Basic logic: 
            -> for every bounding box we have: - center coordinates, 
                                               - width, height, lenght 
                                               - rotation
          
            So in order to represent all the possible bounding boxes, 
            we create a cude mesh in open3d (corner coordinates). 
            Then we :
            - Scale the cube along the 3 different axis (xyz-wlh)
            - Rotate the box around based on the yaw axis
            - Translate its center from the origin to the center of the bbox
    """

    def __init__(self):
        self._scene = KittiScene()

    def add_lidar_pointcloud(self, pointcloud, pc_color=None):
        self._scene.add_lidar_pointcloud(pointcloud, pc_color)

    def add_multiple_boxes(self, bounding_boxes, bbox_color=None):
        self._scene.add_multiple_boxes(bounding_boxes, bbox_color)

    def draw(self):
        self._scene.draw()



class KittiVisualizer(Visualizer):
    
    
    """
    
    Args: 
         - pointcloud: np.array of shape [N, 3] 
                       contains the points of the point cloud

         - bounding_boxes: np.array of shape [N, 7] 
                           contains the bounding box information
                ** Bounding Box Representation 
                x, y, z, l, h, w, ry

    """

    def __init__(self, pointcloud=None, bounding_boxes=None, pc_color=None, bbox_color=None):
        
        super(KittiVisualizer, self).__init__()

        if pointcloud is not None:
            self.add_lidar_pointcloud(pointcloud, pc_color)

        if bounding_boxes is not None:
            self.add_multiple_boxes(bounding_boxes, bbox_color)

        #print(bounding_boxes)


    def load_pointcloud_from_file(self, path_to_lidar_binaries, metadata=None, color=None, use_reduced=True):
        # TODO: Load lidar pointcloud from file
        # use: np_pc_from_bin_file 
        # or create a class that will handle this type of 
        # requests (takes a file name, a path) 
        # ALSO, can i retrieve the path from the example
        # or shall the path be as an argument to this function?

        # path could either be direct path or path to the kitti/training folder
        final_path = Path(path_to_lidar_binaries)

        if metadata is not None:
            image_ind = metadata["image_idx"]
            image_ind = int2constString(image_ind)

            # selecting to load the whole point cloud or the reduced version
            if use_reduced:
                final_path = final_path / "velodyne_reduced"
            else:
                final_path = final_path / "velodyne"

            # adding metadata info to final path
            final_path = final_path / (image_ind + '.bin')    

        # loading the point cloud
        pc = np_pc_from_bin_file(final_path)
        
        self.add_lidar_pointcloud(pc, color)
   


if __name__== "__main__":
    
    
    pc = np.random.randn(1000, 3)
    pc_color = np.array([1.0, 0.0, 0.0])

    bbs = np.random.randn(10, 7)
    bbox_color = np.random.rand(10, 3)

    vs = KittiVisualizer(pc, bbs, pc_color=pc_color, bbox_color=bbox_color)

    vs.draw()
