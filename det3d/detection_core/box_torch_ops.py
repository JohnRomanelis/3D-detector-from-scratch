import torch
import numpy as np

from det3d.detection_core.non_max_suppression.nms_gpu import nms_gpu_cc # also implemented on cpu
                                                                        # TODO: Move to nms_cpu
from det3d.detection_core.non_max_suppression.nms_cpu import rotate_nms_cc

def nms(bboxes, 
        scores,
        pre_max_size=None, 
        post_max_size=None, 
        iou_threshold=0.5):
    """
    Args: 
         - bboxes: detected bounding boxes
         - scores: prediction score for each bounding box
         - pre_max_size: number of bounding boxes to keep before 
                        applying the "non max suppression" algorithm
         - post_max_size: number of points to keep after 
                        applying the "non max suppression" algorithm
         - iou_threshold: the threshold to use for the 
                        "non max suppression" algorithm
    """
    
    if pre_max_size is not None:
        num_keeped_scores = scores.shape[0]
        pre_max_size = min(num_keeped_scores, pre_max_size)
        # keeping the k highest prediction scores
        scores, indices = torch.topk(scores, k=pre_max_size)
        bboxes = bboxes[indices]
    
    # adding prediction score to a matrix along with the bounding box dimensions
    dets = torch.cat([bboxes, scores.unsqueeze(-1)], dim=1)
    # moving matrix to cpu and turning to numpy
    dets_np = dets.data.cpu().numpy()

    # if number of keepers is 0 : do nothing
    if len(dets_np) == 0:
        keep = np.array([], dtype=np.int64)
    # else apply the nms algorithm
    else:
        ret = np.array(nms_gpu_cc(dets_np, iou_threshold), dtype=np.int64)
        if post_max_size is not None:
            post_max_size = min(post_max_size, ret.shape[0])
            keep = ret[:post_max_size]
        else: 
            keep = ret

    # if there are no boxes after the nms algorithm
    if keep.shape[0] == 0:
        # return a matrix of zeros
        return torch.zeros([0]).long().to(bboxes.device)
    
    if pre_max_size is not None:
        keep = torch.from_numpy(keep).long().to(bboxes.device)
        return indices[keep]
    else:
        return torch.from_numpy(keep).long().to(bboxes.device)



def rotate_nms(rbboxes, 
                scores, 
                pre_max_size=None, 
                post_max_size=None,
                iou_threshold=0.5):
    
    if pre_max_size is not None:
        num_keeped_scores = scores.shape[0]
        pre_max_size = min(num_keeped_scores, pre_max_size)
        scores, indices = torch.topk(scores, k=pre_max_size)
        rbboxes = rbboxes[indices]

    dets = torch.cat([rbboxes, scores.unsqueeze(-1)], dim=1)
    dets_np = dets.data.cpu().numpy()
    if len(dets_np) == 0:
        keep = np.array([], dtype=np.int64)
    else:
        ret = np.array(rotate_nms_cc(dets_np, iou_threshold), dtype=np.int64)
        keep = ret[:post_max_size]
    if keep.shape[0] == 0:
        return torch.zeros([0]).long().to(rbboxes.device)
    if pre_max_size is not None:
        keep = torch.from_numpy(keep).long().to(rbboxes.device)
        return indices[keep]
    else:
        return torch.from_numpy(keep).long().to(rbboxes.device)

#helper from second/torchplus
def torch_to_np_dtype(ttype):
    type_map = {
        torch.float16: np.dtype(np.float16),
        torch.float32: np.dtype(np.float32),
        torch.float16: np.dtype(np.float64),
        torch.int32: np.dtype(np.int32),
        torch.int64: np.dtype(np.int64),
        torch.uint8: np.dtype(np.uint8),
    }
    return type_map[ttype]

def corners_nd(dims, origin=0.5):
    """ generate relative box corners based on length per dim
        and origin point

    Args:
         - dims (float array, shape=[N, ndim]): array of length per dim
         - origin (list or array or float): origin point relate to smallest point.
         - dtype (output dtype, optional): Defaults to np.float32 
    
    Returns:
         - float array, shape=[N, 2 ** ndim, ndim]: returned corners. 
         - point layout example: (2d) x0y0, x0y1, x1y0, x1y1;
            (3d) x0y0z0, x0y0z1, x0y1z0, x0y1z1, x1y0z0, x1y0z1, x1y1z0, x1y1z1
            where x0 < x1, y0 < y1, z0 < z1

    """
    ndim = int(dims.shape[1])
    dtype = torch_to_np_dtype(dims.dtype)
    if isinstance(origin, float):
        origin = [origin] * ndim
    corners_norm = np.stack(
        np.unravel_index(np.arange(2**ndim), [2] * ndim), axis=1).astype(dtype)
    # now corners_norm has format: (2d) x0y0, x0y1, x1y0, x1y1
    # (3d) x0y0z0, x0y0z1, x0y1z0, x0y1z1, x1y0z0, x1y0z1, x1y1z0, x1y1z1
    # so need to convert to a format which is convenient to do other computing.
    # for 2d boxes, format is clockwise start from minimum point
    # for 3d boxes, please draw them by your hand.
    if ndim == 2:
        # generate clockwise box corners
        corners_norm = corners_norm[[0, 1, 3, 2]]
    elif ndim == 3:
        corners_norm = corners_norm[[0, 1, 3, 2, 4, 5, 7, 6]]
    corners_norm = corners_norm - np.array(origin, dtype=dtype)
    corners_norm = torch.from_numpy(corners_norm).type_as(dims)
    corners = dims.view(-1, 1, ndim) * corners_norm.view(1, 2**ndim, ndim)
    return corners

def center_to_corner_box2d(centers, dims, angles=None, origin=0.5):
    """ converts kitti locations, dimensions and angles to cornens

    Args: 
         - centers (float array, shape=[N, 2]): locations in kitti label file
         - dims (float array, shape=[N, 2]): dimensions in kitti label file
         - angles (float array, shape=[N]): rotation_y in kitti label file

    """
    # 'length' in kitti format is in x axis.
    # xyz(hwl)(kitti label file)<->xyz(lhw)(camera)<->z(-x)(-y)(wlh)(lidar)
    # center in kitti format is [0.5, 1.0, 0.5] in xyz.
    corners = corners_nd(dims, origin=origin)
    # corners: [N, 4, 2]
    if angles is not None:
        corners = rotation_2d(corners, angles)
    corners += centers.view(-1, 1, 2)
    return corners

def corners_to_standup_nd(boxes_corner):
    ndim = boxes_corner.shape[2]
    standup_boxes = []
    for i in range(ndim):
        standup_boxes.append(torch.min(boxes_corner[:, :, i], dim=1)[0])
    for i in range(ndim):
        standup_boxes.append(torch.max(boxes_corner[:, :, i], dim=1)[0])
    return torch.stack(standup_boxes, dim=1)


def limit_period(val, offset=0.5, period=np.pi):
    return val-torch.floor(val/period) * period
