from .network_template import DetectionNetwork

import numpy as np
import torch.nn as nn
from ..region_proposal_networks.second_rpn import RPN
from ..voxel_feature_extractors.second_vfe import CylVoxelFeatureExtractor

from ..sparse_middle_extractors.second_sparse_middle import SparseMiddleExtractor


class CylidricalSecond(DetectionNetwork):

    def __init__(self, cfg):
        super(CylidricalSecond, self).__init__()
        
        # getting model cfg
        # model_cfg = cfg.model

        # Voxel Feature Extractor
        self.vfe = CylVoxelFeatureExtractor(num_input_features=4,
                                            num_filters=[32, 128])
        

        # Sparse Middle Extractor
        #grid_size = np.array([352, 400, 10])
        grid_size = np.array([352, 160, 10])
        vfe_output_shape = [1] + grid_size[::-1].tolist() + [128]

        self.middle = SparseMiddleExtractor(vfe_output_shape)

        # Region Proposal Network
        self.num_classes = 1 #model_cfg.rpn.num_classes

        self.rpn = RPN(num_class=self.num_classes)

    def forward(self, example):
        batch_size = example['anchors'].shape[0]
        out = self.vfe(example['voxels'], example['num_points'], example['coordinates'])
        out = self.middle(out, example['coordinates'], batch_size)
        out = self.rpn(out)

        return out