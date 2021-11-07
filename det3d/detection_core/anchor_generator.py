import numpy as np
from abc import ABCMeta
from abc import abstractmethod, abstractproperty

# for CylidricalAnchorGeneratorRange
from .coord_sys_utils  import cyl2cart_numpy

class AnchorGenerator(object):
    """ Base class for AnchorGeneration

        Defines the basic methods an AnchorGeneration 
        subclass should have.
    """

    __metaclass__ = ABCMeta

    @abstractproperty
    def num_anchors_per_localization(self):
        pass
    
    @abstractproperty
    def ndim(self):
        pass

    @abstractmethod
    def generate(self, feature_map_size):
        pass


class AnchorGeneratorRange(AnchorGenerator):

    def __init__(self, 
                 anchor_ranges, 
                 sizes=[1.6, 3.9, 1.56],
                 rotations=[0.0, np.pi/2],
                 class_name=None,
                 match_threshold = -1,
                 unmatch_threshold = -1,
                 custom_values=(),
                 dtype=np.float32
                 ):
        
        super().__init__()
        self._sizes = sizes
        self._anchor_ranges = anchor_ranges
        self._rotations = rotations
        self._dtype = dtype
        self._class_name = class_name
        self.match_threshold = match_threshold
        self.unmatch_threshold = unmatch_threshold
        self._custom_values = custom_values

    @property
    def num_anchors_per_localization(self):
        num_rot = len(self._rotations)
        num_size = np.array(self._sizes).reshape([-1, 3]).shape[0]
        return num_rot * num_size

    @property
    def class_name(self):
        return self._class_name
    
    @property
    def ndim(self):
        # 7: [x, y, z, w, l , h, r]
        return 7 + len(self._custom_values)

    @property
    def custom_ndim(self):
        return len(self._custom_values)


    def generate(self, feature_map_size):   
        """
        Desc:
            Generates the anchors
            The feature map size indicates the number of anchors to generate per dimension
            (density of the anchor grid)
        Args:
             - feature_size: list [D, H, W](zyx)
        Returns:
             - anchors: [*feature_size, num_sizes, num_rots, 7] tensor.
            
            anchor example: [0, -40, -3, 1.6, 3.9, 1.56, 0]   
                             x    y   z   w    l     h   r
        """

        anchor_ranges = np.array(self._anchor_ranges, self._dtype)

        z_centers = np.linspace(
            self._anchor_ranges[2], self._anchor_ranges[5], feature_map_size[0], 
                dtype=self._dtype)
        y_centers = np.linspace(
            self._anchor_ranges[1], self._anchor_ranges[4], feature_map_size[1], 
                dtype=self._dtype)
        x_centers = np.linspace(
            self._anchor_ranges[0], self._anchor_ranges[3], feature_map_size[2],
                dtype=self._dtype)
        
        sizes = np.reshape(np.array(self._sizes, dtype=self._dtype), [-1, 3])
        rotations = np.array(self._rotations, dtype=self._dtype)

        # creating a meshgrid 
        rets = np.meshgrid(
            x_centers, y_centers, z_centers, rotations, indexing='ij')
        
        tile_shape = [1] * 5
        tile_shape[-2] = int(sizes.shape[0])
        for i in range(len(rets)):
            rets[i] = np.tile(rets[i][..., np.newaxis, :], tile_shape)
            rets[i] = rets[i] [..., np.newaxis] # adding a new dim for concatenation
        
        sizes = np.reshape(sizes, [1, 1, 1, -1, 1, 3])
        tile_size_shape = list(rets[0].shape)
        tile_size_shape[3] = 1
        sizes = np.tile(sizes, tile_size_shape)
        rets.insert(3, sizes)
        ret = np.concatenate(rets, axis=-1)
        ret = np.transpose(ret, [2, 1, 0, 3, 4, 5])

        return ret


class CylidricalAnchorGeneratorRange(AnchorGeneratorRange):

    def generate(self, feature_map_size):
        
        anchors = super().generate(feature_map_size)

        anchor_pos = anchors[..., :3]
        anchor_pos = cyl2cart_numpy(anchor_pos)

        anchors[..., :3] = anchor_pos

        return anchors



if __name__ == "__main__":

    anchor_generator = AnchorGeneratorRange([0, 0, 0, 1, 1, 1])
    anchors = anchor_generator.generate([2, 3, 4])
    print(anchors.shape)
