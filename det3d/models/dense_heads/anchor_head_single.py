import numpy as np
import torch 
import torch.nn as nn


class AnchorHeadSingle(nn.Module):


    def __init__(self, input_channels,
                       num_class=1, # maybe use 2
                       num_anchors_per_location=2, 
                       coder_size=7, 
                       use_direction_classifier=True, 
                       num_dir_bins=2):
        super().__init__()

        self.num_anchors_per_location = num_anchors_per_location
        self.num_class = num_class
        self.coder_size = coder_size

        self.conv_cls = nn.Conv2d(
            input_channels, 
            self.num_anchors_per_location * self.num_class, 
            kernel_size=1
        )

        self.conv_box = nn.Conv2d(
            input_channels, 
            self.num_anchors_per_location * self.coder_size,
            kernel_size=1
        )

        if use_direction_classifier:
            self.conv_dir_cls = nn.Conv2d(
                input_channels, 
                self.num_anchors_per_location * num_dir_bins, 
                kernel_size=1
            )
        else:
            self.conv_dir_cls = None

        self.init_weights()


    def init_weights(self):
        pi=0.01
        nn.init.constant_(self.conv_cls.bias, -np.log((1-pi) / pi))
        nn.init.normal_(self.conv_box.weight, mean=0, std=0.001)



    def forward(self, x):


        cls_preds = self.conv_cls(x)
        box_preds = self.conv_box(x)


        cls_preds = cls_preds.permute(0, 2, 3, 1).contiguous()  # [N, H, W, C]
        box_preds = box_preds.permute(0, 2, 3, 1).contiguous()  # [N, H, W, C]

        if self.conv_dir_cls is not None:
            dir_cls_preds = self.conv_dir_cls(x)
            dir_cls_preds = dir_cls_preds.permute(0, 2, 3, 1).contiguous()

        return cls_preds, box_preds, dir_cls_preds