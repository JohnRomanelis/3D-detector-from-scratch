from abc import ABCMeta
from abc import abstractmethod, abstractproperty

# for coders in numpy
import numpy as np
# for torch decoder (inference)
import torch


# BoxCoder abstract class
class BoxCoder(object):
    
    __metaclass__ = ABCMeta

    @abstractproperty
    def code_size(self):
        pass

    def encode(self, boxes, anchors):
        return self._encode(boxes, anchors)

    def decode(self, encodings, anchors):
        return self._decode(encodings, anchors)

    @abstractmethod
    def _encode(self, boxes, anchor):
        pass

    @abstractmethod
    def _decode(self, encodings, anchors):
        pass


class GroundBox3dCoder(BoxCoder):

    def __init__(self, linear_dim=False, vec_encode=False, custom_ndim=0):
        super().__init__()
        self.linear_dim = linear_dim
        self.vec_encode = vec_encode
        self.custom_ndim = custom_ndim


    @property
    def code_size(self):
        res = 8 if self.vec_encode else 7
        return self.custom_ndim + res

    def _encode(self, boxes, anchors):
        """ 
        box encoder for VoxelNet in lidar
        
        Desc: Receives boxes(prediction targets) and the anchors and generates 
              for each anchor the regression target. Then use BoxCoder.decode to 
              get the predicted bounding box.
        Args:
            - boxes ([N, 7 + ?], Tensor): normal boxes: x, y, z, w, l, h, r, custom values
            - anchors([N, 7] Tensor): anchors
        """
        # NOTE: need to convert boxes to z center format

        box_ndim = anchors.shape[-1]

        cas, cgs = [], []
        if box_ndim > 7:
            xa, ya, za, wa, la, ha, ra, *cas = np.split(anchors, box_ndim, axis=1)
            xg, yg, zg, wg, lg, hg, rg, *cgs = np.split(boxes, box_ndim, axis=1)
        else:
            xa, ya, za, wa, la, ha, ra = np.split(anchors, box_ndim, axis=1)
            xg, yg, zg, wg, lg, hg, rg = np.split(boxes, box_ndim, axis=1)

        diagonal = np.sqrt(la**2 + wa**2)

        xt = (xg - xa) / diagonal
        yt = (yg - ya) / diagonal
        zt = (zg - za) / ha

        cts = [g - a for g, a in zip(cgs, cas)]

        lt = np.log(lg / la)
        wt = np.log(wg / wa)
        ht = np.log(hg / ha)

        rt = rg - ra

        return np.concatenate([xt, yt, zt, wt, lt, ht, rt, *cts], axis=1)


    def _decode(self, box_encodings, anchors):
        """
            box decoder for VoxelNet in lidar
        Args: 
            - boxes ([N, 7] np.array): normal boxes: x, y, z, w, l, h, r
            - anchors ([N, 7] np.array): anchors
        """

        box_ndim = anchors.shape[-1]
        cas, cts = [], []

        if box_ndim > 7: 
            xa, ya, za, wa, la, ha, ra, *cas = np.split(anchors, box_ndim, axis=-1) 
            xt, yt, zt, wt, lt, ht, rt, *cts = np.split(box_encodings, box_ndim, axis=-1)
        else:
            xa, ya, za, wa, la, ha, ra = np.split(anchors, box_ndim, axis=-1)
            xt, yt, zt, wt, lt, ht, rt = np.split(box_encodings, box_ndim, axis=-1)

        diagonal = np.sqrt(la**2 + wa**2)
        xg = xt * diagonal + xa
        yg = yt * diagonal + ya
        zg = zt * ha + za

        lg = np.exp(lt) * la
        wg = np.exp(wt) * wa
        hg = np.exp(ht) * ha

        rg = rt + ra

        cgs = [t + a for t, a in zip(cts, cas)]

        return np.concatenate([xg, yg, zg, wg, lg, hg, rg, *cgs], axis=-1)



class GroundBox3dCoderTorch(GroundBox3dCoder):
    """ 
        Class that allows the decoding operation to be applied
        on a torch tensor.
        torch_decode can be used during training - inference - evaluation
    """

    def decode_torch(self, box_encodings, anchors):
        """box decode for VoxelNet in lidar
        Args:
            - boxes ([N, 7] Tensor): normal boxes: x, y, z, w, l, h, r
            - anchors ([N, 7] Tensor): anchors
        """

        box_ndim = anchors.shape[-1]

        cas, cts = [], []
        if box_ndim > 7:
            xa, ya, za, la, ha, ra, *cas = torch.split(anchors, 1, dim=-1)
            xt, yt, zt, lt, ht, rt, *cts = torch.split(box_encodings, 1, dim=-1)
        else:
            xa, ya, za, wa, la, ha, ra = torch.split(anchors, 1, dim=-1)
            xt, yt, zt, wt, lt, ht, rt = torch.split(box_encodings, 1, dim=-1)


        diagonal = torch.sqrt(la**2 + wa**2)
        xg = xt * diagonal + xa
        yg = yt * diagonal + ya
        zg = zt * ha + za

        lg = torch.exp(lt) * la
        wg = torch.exp(wt) * wa
        hg = torch.exp(ht) * ha

        rg = rt + ra
        cgs = [t + a for t, a in zip(cts, cas)]
        return torch.cat([xg, yg, zg, wg, lg, hg, rg, *cgs], dim=-1)  


