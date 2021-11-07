import torch
import torch.nn as nn
import torch.nn.functional as F


import numpy as np

from ..backbones_2d import BaseBEVBackbone
from ..dense_heads import AnchorHeadSingle

class SECONDGithubRPN(nn.Module):
    # deprecated : Used only for voxelnet implementation

    def __init__(self, 
                use_norm=True, 
                num_class=2, 
                layer_nums=(3, 5, 5),
                layer_strides=(2, 2, 2),
                num_filters=(128, 128, 256),
                upsample_strides=(1, 2, 4),
                num_upsample_filters=(256, 256, 256), 
                num_input_features=128,
                num_anchor_per_loc=2,
                encode_backround_as_zeros=True, 
                use_direction_classifier=True, 
                use_groupnorm=False,
                num_groups=32, 
                box_code_size=7, 
                num_direction_bins=2
                ):

        super().__init__()

        self._use_direction_classifier = use_direction_classifier

        self.backbone2d = BaseBEVBackbone(input_channels=256)
        self.dense_head = AnchorHeadSingle(input_channels=512, 
                                           use_direction_classifier=use_direction_classifier)

    def forward(self, x):
        #print('in ', x.shape)

        # Extracting features from 2d bev image
        x = self.backbone2d(x)

        # extracting detection results
        cls_preds, box_preds, dir_cls_preds = self.dense_head(x)


        ret_dict = {
            "box_preds":box_preds,
            "cls_preds":cls_preds
        }

        if self._use_direction_classifier:
            ret_dict["dir_cls_preds"] = dir_cls_preds
        
        return ret_dict

