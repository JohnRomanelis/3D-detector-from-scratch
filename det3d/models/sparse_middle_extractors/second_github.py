import torch.nn as nn
from ..backbones_3d import VoxelBackbone8x

class SECONDGithubSME(nn.Module):

    def __init__(self, 
                 input_features=4):
        super().__init__()
        
        self.backbone3d = VoxelBackbone8x(input_features)


    def forward(self, voxel_features, coors, batch_size):
        
        # 3D feature extraction
        ret = self.backbone3d(voxel_features, coors, batch_size)

        # Projecting to dense grid
        ret = ret.dense()
        #print(ret.shape)
        N, C, D, H, W = ret.shape 
        ret = ret.view(N, C * D, H, W)
        #print(ret.shape)
        return ret