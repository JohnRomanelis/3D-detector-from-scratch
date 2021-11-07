import numpy as np

import torch
import torch.nn as nn
import torch.nn.init as init
from math import sqrt


class BaseBEVBackbone(nn.Module):

    def __init__(self, input_channels, 
                       layer_nums=[5, 5], 
                       layer_strides=[1, 2], 
                       num_filters=[128, 256], 
                       upsample_strides=[1, 2], 
                       num_upsample_filters=[256, 256]):
        super().__init__()


        # number of layers
        num_levels = len(layer_nums)

        c_in_list = [input_channels, *num_filters[:-1]]
        self.blocks = nn.ModuleList()
        self.deblocks = nn.ModuleList()

        for idx in range(num_levels):
            cur_layers = [
                nn.ZeroPad2d(1), 
                nn.Conv2d(
                    c_in_list[idx], num_filters[idx], kernel_size=3, 
                    stride=layer_strides[idx], padding=0, bias=False
                ), 
                nn.BatchNorm2d(num_filters[idx], eps=1e-3, momentum=0.01), 
                nn.ReLU()
            ]

            for k in range(layer_nums[idx]):
                cur_layers.extend([
                    nn.Conv2d(num_filters[idx], num_filters[idx], kernel_size=3, padding=1, bias=False), 
                    nn.BatchNorm2d(num_filters[idx], eps=1e-3, momentum=0.01), 
                    nn.ReLU()
                ])

            self.blocks.append(nn.Sequential(*cur_layers))

            if len(upsample_strides) > 0:
                stride = upsample_strides[idx]
                if stride >= 1:
                    self.deblocks.append(nn.Sequential(
                        nn.ConvTranspose2d(
                            num_filters[idx], num_upsample_filters[idx], 
                            upsample_strides[idx], #kernel size
                            stride=upsample_strides[idx], bias=False
                        ),
                        nn.BatchNorm2d(num_upsample_filters[idx], eps=1e-3, momentum=0.01), 
                        nn.ReLU()
                    ))
                else:
                    stride = np.round(1 / stride).astype(np.int)
                    self.deblocks.append(nn.Sequential(
                        nn.Conv2d(
                            num_filters[idx], num_upsample_filters[idx], 
                            stride, #kernel size
                            stride=stride, bias=False
                        ), 
                        nn.BatchNorm2d(num_upsample_filters[idx], eps=1e-3, momentum=0.01), 
                        nn.ReLU()
                    ))

        c_in = sum(num_upsample_filters)

        if len(upsample_strides) > num_levels:
            self.deblocks.append(nn.Sequential(
                nn.ConvTranspose2d(c_in, c_in, upsample_strides[-1], stride=upsample_strides[-1], bias=False), 
                nn.BatchNorm2d(c_in, eps=1e-3, momentum=0.01),
                nn.ReLU()
            ))

        self.num_bev_features = c_in

        #self.init_weights()

    def init_weights(self):
        for seq_layer in self.blocks:
            for layer in seq_layer:
                classname = layer.__class__.__name__
                if classname.find('Conv') != -1:
                    init.kaiming_uniform_(layer.weight, a=sqrt(2))

    def forward(self, x):

        ups = []
        for i in range(len(self.blocks)):
            x = self.blocks[i](x)

            if len(self.deblocks) > 0:
                ups.append(self.deblocks[i](x))
            else:
                ups.append(x)

        if len(ups) > 1:
            x = torch.cat(ups, dim=1)
        elif len(ups) == 1:
            x = ups[0]

        if len(self.deblocks) > len(self.blocks):
            x = self.deblocks[-1](x)


        return x