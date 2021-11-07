import torch.nn as nn
import torch.nn.init as init
from math import sqrt
import MinkowskiEngine as ME
from functools import partial

def post_act_block(in_channels, out_channels, kernel_size, indice_key=None, stride=1, padding=0, 
                    conv_type='subm', norm_fn = None):

    conv = ME.MinkowskiConvolution(in_channels, 
                                   out_channels, 
                                   kernel_size=kernel_size, 
                                   stride=stride,
                                   bias=False, 
                                   dimension=3)
    init.kaiming_uniform_(conv.weights, sqrt(2))

    m = nn.Sequential(
        conv, 
        norm_fn(out_channels), 
        ME.MinkowskiReLU()
    )

    return m

class MinkowskiVoxelBackbone8x(nn.Module):

    def __init__(self, 
                 input_channels, 
                 grid_size = [1408, 1600, 40]):
        
        super().__init__()

        norm_fn = partial(ME.MinkowskiBatchNorm(), eps=1e-3, momentum=0.1)

        block = partial(post_act_block, norm_fn=norm_fn)

        self.input_conv = block(input_channels, 16, 3)

        self.conv1 = nn.Sequential(
            block(16, 16, 3)
        )

        self.conv2 = nn.Sequential(
            block(16, 32, 3, stride=2),
            block(32, 32, 3),
            block(32, 32, 3),
        )

        self.conv3 = nn.Sequential(
            block(32, 64, 3, stride=2),
            block(64, 64, 3),
            block(64, 64, 3),
        )

        self.conv4 = nn.Sequential(
            block(64, 128, 3, stride=2),
            block(128, 128, 3),
            block(128, 128, 3),
        )

        self.conv_out = nn.Sequential(
            block(128, 128, (3, 1, 1), stride = (2, 1, 1))
        )

    def forward(self, voxel_features, coords, batch_size):

        input_sp_tensor = ME.SparseTensor(
            features = voxel_features, 
            coordinates = coords, 
            device = 'cuda:0'
        )

        x = self.input_conv(input_sp_tensor)

        x_conv1 = self.conv1(x)
        x_conv2 = self.conv2(x_conv1)
        x_conv3 = self.conv3(x_conv2)
        x_conv4 = self.conv4(x_conv3)

        # [200, 176, 5] -> [200, 176, 2]
        out = self.conv_out(x_conv4)

        return out
