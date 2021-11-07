import torch 
import torch.nn as nn

import numpy as np
import spconv

class SparseMiddleExtractor(nn.Module):
    ''' 
        Sparse Middle Voxel Feature Extractor
            -->   Has 2 down convolutions layers

    '''


    def __init__(self, 
                 output_shape,
                 use_norm=True,
                 num_input_features=128,
                 num_filters_down1 = [64],
                 num_filters_down2 = [64, 64]            
                ):
        super(SparseMiddleExtractor, self).__init__()

        # BatchNormalization

        # 
        sparse_shape = np.array(output_shape[1:4]) + [1,0,0]
        # sparse_shape[0]=11

        #print(sparse_shape)
        self.sparse_shape = sparse_shape

        #
        # ...
        #

        self.voxel_output_shape = output_shape

        # constructing the layers
        middle_layers = []

        ###################
        # first down conv #
        ###################
        num_filters = [num_input_features] + num_filters_down1

        filter_pairs_d1 = [ [num_filters[i], num_filters[i + 1]]
                            for i in range(len(num_filters) - 1)  
                          ]
        
        for in_channels, out_channels in filter_pairs_d1:
            middle_layers.extend([
                spconv.SubMConv3d(in_channels, out_channels, 3, bias=False, indice_key="subm0"),
                nn.BatchNorm1d(out_channels),
                nn.ReLU()
            ])
        middle_layers.extend([
            spconv.SparseConv3d(num_filters[-1], #in_channels
                                num_filters[-1], #out_channels 
                                (3,1,1), #kernel size
                                (2, 1, 1), #stride
                                bias=False),
            nn.BatchNorm1d(num_filters[-1]),
            nn.ReLU()
        ])

        ####################
        # second down conv #
        ####################
        if len(num_filters_down1) == 0:
            num_filters = [num_filters[-1]] + num_filters_down2
        else:
            num_filters = [num_filters_down1[-1]] + num_filters_down2
        # creating filter pairs
        filter_pairs_d2 = [[num_filters[i], num_filters[i + 1]]
                            for i in range(len(num_filters) - 1)]

        for in_channels, out_channels in filter_pairs_d2:
            middle_layers.extend([
                spconv.SubMConv3d(in_channels, out_channels, 3, bias=False, indice_key="subm1"),
                nn.BatchNorm1d(out_channels),
                nn.ReLU()
            ])
        middle_layers.extend([
            spconv.SparseConv3d(num_filters[-1], #in_channels
                                num_filters[-1], #out_channels
                                (3, 1, 1), #kernel size
                                (2, 1, 1), #stride
                                bias=False),
            nn.BatchNorm1d(num_filters[-1]),
            nn.ReLU()
        ])
        
        # Creating a sequential layer
        self.middle_conv = spconv.SparseSequential(*middle_layers)
    


    def forward(self, voxel_features, coors, batch_size):
        # coors[:, 1] += 1
        # print(coors)
        coors = coors.int()
        #print("sparse shape ", self.sparse_shape)
        ret = spconv.SparseConvTensor(voxel_features, coors, 
                                        self.sparse_shape, batch_size)
        #print(ret.dense().shape)
        ret = self.middle_conv(ret)
        #print(ret.dense().shape)
        ret = ret.dense()
        #raise ValueError
        #print(ret.shape)
        N, C, D, H, W = ret.shape 
        ret = ret.view(N, C * D, H, W)
        #print(ret.shape)
        return ret