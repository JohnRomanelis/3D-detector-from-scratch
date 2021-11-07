import torch
import torch.nn as nn

from det3d.detection_core.coord_sys_utils import cart2cyl_with_features_torch


class MeanVFE(nn.Module):

    def __init__(self, num_point_features, **kwargs):
        super().__init__()
        self.num_point_features = num_point_features

    @property
    def get_output_feature_dim(self):
        return self.num_point_features

    def forward(self, features, num_voxels, coors):
        
        points_mean = features[:, :, :].sum(dim=1, keepdim=False)
        normalizer = torch.clamp_min(num_voxels.view(-1, 1), min=1.0).type_as(features)
        points_mean = points_mean / normalizer
        # points_mean = voxelwise features
        return points_mean


class MeanVFECyl(nn.Module):

    def __init__(self, num_point_features, **kwargs):
        super().__init__()
        self.num_point_features = num_point_features


    def forward(self, features, num_voxels, coors):
        # transforming point cloud to cylidrical coordinates
        features = cart2cyl_with_features_torch(features)
        # removing theta (angle) from the features
        features = features[..., [0, 2, 3]]
        # rest is same as MeanVFE
        points_mean = features[:, :, :].sum(dim=1, keepdim=False)
        normalizer = torch.clamp_min(num_voxels.view(-1, 1), min=1.0).type_as(features)
        points_mean = points_mean / normalizer
        # points_mean = voxelwise features
        return points_mean



class OccupancyVoxelFeatures(nn.Module):
    
    def forward(self, features, num_voxels, coors):

        #features = features[..., -2:]#.unsqueeze(-1) # keeping only reflectance and height
        features = features[..., -1].unsqueeze(-1) # keeping only reflectance
        ones = features.new_ones((*features.shape[:-1], 1))
        
        features = torch.cat([features, ones], dim=-1)

        points_mean = features[:, :, :].sum(dim=1, keepdim=False)
        normalizer = torch.clamp_min(num_voxels.view(-1, 1), min=1.0).type_as(features)
        points_mean = points_mean / normalizer
        # points_mean = voxelwise features
        return points_mean
