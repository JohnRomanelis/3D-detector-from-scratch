import torch 
import torch.nn as nn
import torch.nn.functional as F 

from det3d.models.model_utils import get_paddings_indicator

from det3d.detection_core.coord_sys_utils import cyl2cart_torch


class VFELayer(nn.Module):
    """
    Desc: 
        Implementation of VFE layer as presented in the SECOND detector architecture.
        Original implementation was in the VoxelNet

    Args:
         - Input Channels
         - Output Channels
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.units = out_channels//2
        self.linear = nn.Linear(in_channels, self.units)
        self.norm = nn.BatchNorm1d(self.units)

    def forward(self, input, mask=None):
        """
            Input Dimensions:
                input: [K, T, F]
                
                ** If we have batch size 3, then:
                    1. [K1, T, F] (K1 the number of non-empty voxels)
                    2. [K2, T, F] 
                    3. [K3, T, F] 

            Output: 
                [batch_size, D', H', W', F']
                    where: D', H', W' : number of voxels per dimension
                           F'         : number of generated features

        """
        # [K, T, 7] tensordot [7, units] = [K, T, units]
        voxel_count = input.shape[1]
        
        x = self.linear(input)
        x = self.norm(x.permute(0,2,1).contiguous()).permute(0,2,1).contiguous()
        
        pointwise = F.relu(x)
        # size: [K, T, units]

        aggregated = torch.max(pointwise, dim=1, keepdim=True)[0]
        # size: [K, 1, units]

        repeated = aggregated.repeat(1, voxel_count, 1)
        # size: [K, T, units]

        concatenated = torch.cat([pointwise, repeated], dim=2)
        # size: [K, T, 2 * units]

        return concatenated



class VoxelFeatureExtractor(nn.Module):
    """ 
        Creating the Voxel Feature Encoder 
        model used for the SECOND detector
    """

    def __init__(self,
                num_input_features=4,
                use_norm=True,
                num_filters=[32, 128],
                with_distance=False,
                voxel_size=(0.2, 0.2, 0.4),
                pc_range=(0, -40, -3, 70.4, 40, 1),  
                ):
        super().__init__()
        with_distance = True
        num_input_features += 3 # adding the mean features
        if with_distance:
            num_input_features += 1
        self._with_distance = with_distance

        # creating voxel feature extractor layer
        self.vfe1 = VFELayer(num_input_features, num_filters[0])
        self.vfe2 = VFELayer(num_filters[0], num_filters[1])
        
        # adding a linear layer
        self.linear = nn.Linear(num_filters[1], num_filters[1])
        self.norm = nn.BatchNorm1d(num_filters[1])

    def forward(self, features, num_voxels, coors):
        """
            Input Dimensions:
                - features: [K, T, F]
                    K: number of total voxels inside the batch
                    T: maximum number of points per voxel
                    F: number of features
                    ** If we have batch size 3, then:
                        1. [K1, T, F] (K1 the number of non-empty voxels)
                        2. [K2, T, F] 
                        3. [K3, T, F] 

                - num_voxels: [K]
                    : actual number of points inside the voxel
        """

        # calculating the mean position of all points inside a voxel
        # NOTE: do not devide with T, because some voxels may not be full
        points_mean = features[:, :, :3].sum(dim=1, 
            keepdim=True) / num_voxels.type_as(features).view(-1, 1, 1)
        
        features_relative = features[:,:,:3] - points_mean

        # Concating features to a feature vector
        # adding distance of points from the origin as an extra feature
        if self._with_distance:
            points_dist = torch.norm(features[:,:,:3], 2, 2, keepdim=True)
            features = torch.cat([features, features_relative, points_dist], dim=-1)
        else:
            features = torch.cat([features, features_relative], dim=-1)

        voxel_count = features.shape[1] #T: maximun number of points per voxel

        # mask : elementwise boolean mask that indicates if an element of
        #        of a tensor has an actual value or if it zero as a result
        #        of the padding. In other words, it masks the real points 
        #        inside a voxel.
        mask = get_paddings_indicator(num_voxels, voxel_count, axis=0)
        mask = torch.unsqueeze(mask, -1).type_as(features)

        # masking features because zero elements get large negative values after subtraction
        features *= mask
        # passing features through the VFELayers
        x = self.vfe1(features)
        x *= mask
        x = self.vfe2(x)
        x *= mask
        x = self.linear(x)
        x = self.norm(x.permute(0, 2, 1).contiguous()).permute(0, 2,
                                                               1).contiguous()   
        x = F.relu(x)
        x *= mask
        # x shape: [K, T, 128]
        
        # max aggregation to obtain voxelwise features
        voxelwise = torch.max(x, dim=1)[0]
        
        return voxelwise



class CylVoxelFeatureExtractor(nn.Module):
    """ 
    Creating a voxel feature extractor that takes as input the cylidrical 
    coordinates of the points.
    Functionalities:
        - Input Normalization
        - Ignoring input angle
    """


    def __init__(self, 
                 num_input_features=4, 
                 num_filters=[32, 128]):
        super().__init__()


        # Input Features:
        #  - r
        #  - height
        #  - reflectance
        #  - dr
        #  - dtheta
        #  - dheigh
        
        # total of 6 features
        num_input_features = 7 # mean distance, mean height

        # creating voxel feature extractor layer
        self.vfe1 = VFELayer(num_input_features, num_filters[0])
        self.vfe2 = VFELayer(num_filters[0], num_filters[1])

        # adding a linear layer
        self.linear = nn.Linear(num_filters[1], num_filters[1])
        self.norm = nn.BatchNorm1d(num_filters[1])


    def forward(self, features, num_voxels, coors):
        """
            Input Dimensions:
                - features: [K, T, F]
                    K: number of total voxels inside the batch
                    T: maximum number of points per voxel
                    F: number of features
                    ** If we have batch size 3, then:
                        1. [K1, T, F] (K1 the number of non-empty voxels)
                        2. [K2, T, F] 
                        3. [K3, T, F] 

                - num_voxels: [K]
                    : actual number of points inside the voxel
        """

        #features[..., :3] = cyl2cart_torch(features[..., :3])

        # calculating the mean position of all points inside a voxel
        # NOTE: do not devide with T, because some voxels may not be full
        points_mean = features[:, :, :3].sum(dim=1, 
            keepdim=True) / num_voxels.type_as(features).view(-1, 1, 1)
        
        features_relative = features[:,:,:3] - points_mean

        features_no_theta = features#[:, :, [0, 2, 3]]
        # normalizing features
        features_no_theta[..., 0] = features_no_theta[..., 0] / 40 #80
        features_no_theta[..., 2] = (features_no_theta[..., 2] + 1) / 2
        
        features = torch.cat([features_no_theta, features_relative], dim=-1)
        voxel_count = features.shape[1]


        # mask : elementwise boolean mask that indicates if an element of
        #        of a tensor has an actual value or if it zero as a result
        #        of the padding. In other words, it masks the real points 
        #        inside a voxel.
        mask = get_paddings_indicator(num_voxels, voxel_count, axis=0)
        mask = torch.unsqueeze(mask, -1).type_as(features)

        # masking features because zero elements get large negative values after subtraction
        features *= mask
        # passing features through the VFELayers
        x = self.vfe1(features)
        x *= mask
        x = self.vfe2(x)
        x *= mask
        x = self.linear(x)
        x = self.norm(x.permute(0, 2, 1).contiguous()).permute(0, 2,
                                                               1).contiguous()   
        x = F.relu(x)
        x *= mask
        # x shape: [K, T, 128]
        
        # max aggregation to obtain voxelwise features
        voxelwise = torch.max(x, dim=1)[0]
        
        return voxelwise
