from .network_template import DetectionNetwork

from ..voxel_feature_extractors import MeanVFE, MeanVFECyl, OccupancyVoxelFeatures
from ..sparse_middle_extractors import SECONDGithubSME
from ..region_proposal_networks import SECONDGithubRPN

class SECONDGithub(DetectionNetwork):


    def __init__(self, cfg):
        
        super().__init__()

        self.vfe = MeanVFE(num_point_features=4)

        self.middle = SECONDGithubSME()

        self.rpn = SECONDGithubRPN()


    def forward(self, example):
        batch_size = example['anchors'].shape[0]
        out = self.vfe(example['voxels'], example['num_points'], example['coordinates'])
        out = self.middle(out, example['coordinates'], batch_size)
        out = self.rpn(out)

        return out
        
class SECONDGithub2(DetectionNetwork):
    
    # Uses as voxel features 
    #   - the distance from the origin
    #   - the height of the voxel
    #   - the lidar reflectance

    def __init__(self, cfg):
        super().__init__()

        self.vfe = MeanVFECyl(num_point_features=4)
        self.middle = SECONDGithubSME(input_features=3)
        self.rpn = SECONDGithubRPN()


    def forward(self, example):
        batch_size = example['anchors'].shape[0]
        out = self.vfe(example['voxels'], example['num_points'], example['coordinates'])
        out = self.middle(out, example['coordinates'], batch_size)
        out = self.rpn(out)

        return out


class SECONDGithubOccupancy(DetectionNetwork):
    
    # Uses as voxel features 
    #   - the distance from the origin
    #   - the height of the voxel
    #   - the lidar reflectance

    def __init__(self, cfg):
        super().__init__()

        self.vfe = OccupancyVoxelFeatures()
        self.middle = SECONDGithubSME(input_features=2)
        self.rpn = SECONDGithubRPN()


    def forward(self, example):
        batch_size = example['anchors'].shape[0]
        out = self.vfe(example['voxels'], example['num_points'], example['coordinates'])
        out = self.middle(out, example['coordinates'], batch_size)
        out = self.rpn(out)

        return out
