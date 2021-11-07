import torch
import torch.nn as nn
import torch.nn.functional as F


import numpy as np


class RPN(nn.Module):
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
        super(RPN, self).__init__()
        self._num_anchor_per_loc = num_anchor_per_loc
        self._use_direction_classifier = use_direction_classifier
        assert len(layer_nums) == 3
        assert len(layer_strides) == len(layer_nums)
        assert len(num_filters) == len(layer_nums)
        assert len(upsample_strides) == len(layer_nums)
        assert len(num_upsample_filters) == len(layer_nums)
        upsample_strides = [
            np.round(u).astype(np.int64) for u in upsample_strides
        ]
        factors = []
        for i in range(len(layer_nums)):
            assert int(np.prod(
                layer_strides[:i + 1])) % upsample_strides[i] == 0
            factors.append(
               np.prod(layer_strides[:i + 1]) // upsample_strides[i]) 
            
        assert all([x==factors[0] for x in factors])

        # 
        # ... 
        #

        # note that when stride > 1, conv2d with same padding isn't
        # equal to pad-conv2d. we should use pad-conv2d.

        block2_input_filters = num_filters[0]


        block1 = []
        block1.extend([
            nn.ZeroPad2d(1),
            nn.Conv2d(num_input_features, num_filters[0], 3,
            stride = layer_strides[0]),
            nn.BatchNorm2d(num_filters[0]),
            nn.ReLU()
        ])
        for i in range(layer_nums[0]):
            block1.append(nn.Conv2d(num_filters[0], num_filters[0], 3, padding=1))
            block1.append(nn.BatchNorm2d(num_filters[0]))
            block1.append(nn.ReLU())
        
        self.block1 = nn.Sequential(*block1)
            

        self.deconv1 = nn.Sequential(
            nn.ConvTranspose2d(
                num_filters[0], 
                num_upsample_filters[0],
                upsample_strides[0], 
                stride = upsample_strides[0]
            ), 
            nn.BatchNorm2d(num_upsample_filters[0]),
            nn.ReLU()
            )
        
        block2 = []
        block2.extend([
            nn.ZeroPad2d(1),
            nn.Conv2d(
                block2_input_filters, 
                num_filters[1], 
                3, 
                stride=layer_strides[1]
            ),
            nn.BatchNorm2d(num_filters[1]),
            nn.ReLU()
        ])
        for i in range(layer_nums[1]):
            block2.append(nn.Conv2d(num_filters[1], num_filters[1], 3, padding=1))
            block2.append(nn.BatchNorm2d(num_filters[1]))
            block2.append(nn.ReLU())

        self.block2 = nn.Sequential(*block2)            
        
        self.deconv2 = nn.Sequential(
            nn.ConvTranspose2d(
                num_filters[1],
                num_upsample_filters[1],
                upsample_strides[1],
                stride=upsample_strides[1]
            ),
            nn.BatchNorm2d(num_upsample_filters[1]),
            nn.ReLU()
        )

        block3 = []
        block3.extend([
            nn.ZeroPad2d(1),
            nn.Conv2d(num_filters[1], num_filters[2], 3, stride=layer_strides[2]),
            nn.BatchNorm2d(num_filters[2]),
            nn.ReLU()
        ])   

        for i in range(layer_nums[2]):
            block3.append(nn.Conv2d(num_filters[2], num_filters[2], 3, padding=1))
            block3.append(nn.BatchNorm2d(num_filters[2]))
            block3.append(nn.ReLU())

        self.block3 = nn.Sequential(*block3)

        self.deconv3 = nn.Sequential(
            nn.ConvTranspose2d(
                num_filters[2],
                num_upsample_filters[2],
                upsample_strides[2],
                stride=upsample_strides[2]
            ),
            nn.BatchNorm2d(num_upsample_filters[2]),
            nn.ReLU(),
        )

        if encode_backround_as_zeros:
            num_cls = num_anchor_per_loc * num_class
        else:
            num_cls = num_anchor_per_loc * (num_class + 1)

        self.conv_cls = nn.Conv2d(sum(num_upsample_filters), num_cls, 1)
        self.conv_box = nn.Conv2d(
            sum(num_upsample_filters), num_anchor_per_loc * box_code_size, 1
        )

        if use_direction_classifier:
            self.conv_dir_cls = nn.Conv2d(
                sum(num_upsample_filters), 
                num_anchor_per_loc * num_direction_bins, 1
            )

        #if self._use_rc_net:
        #    self.conv_rc = nn.Conv2d(
        #        sum(num_upsample_filters), num_anchor_per_loc * box_code_size,
        #        1)


    def forward(self, x):
        #print('in ', x.shape)
        x = self.block1(x)
        #print("x1 ", x.shape)
        up1 = self.deconv1(x)
        #print("up1 ", up1.shape)

        x = self.block2(x)
        #print("x2 ",x.shape)
        up2 = self.deconv2(x)
        #print("up2 ", up2.shape)

        x = self.block3(x)
        #print("x3 ",x.shape)
        up3 = self.deconv3(x)
        #print(up1.shape, up2.shape, up3.shape)
        x = torch.cat([up1, up2, up3], dim=1)
        #print("out ", x.shape)
        box_preds = self.conv_box(x)
        cls_preds = self.conv_cls(x)

        # [N, C, y(H), x(W)]
        box_preds = box_preds.permute(0, 2, 3, 1).contiguous()
        cls_preds = cls_preds.permute(0, 2, 3, 1).contiguous()

        #print(cls_preds.shape)

        ret_dict = {
            "box_preds":box_preds,
            "cls_preds":cls_preds
        }

        if self._use_direction_classifier:
            dir_cls_preds = self.conv_dir_cls(x)
            dir_cls_preds = dir_cls_preds.permute(0, 2, 3, 1).contiguous()
            ret_dict["dir_cls_preds"] = dir_cls_preds
        
        #if self._use_rc_net:
        #    rc_preds = self.conv_rc(x)
        #    rc_preds = rc_preds.permute(0, 2, 3, 1).contiguous()
        #    ret_dict["rc_preds"] = rc_preds

        return ret_dict