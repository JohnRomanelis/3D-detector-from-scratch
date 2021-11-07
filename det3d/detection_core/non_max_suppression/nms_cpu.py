import numpy as np

from det3d.detection_core import box_np_ops

from spconv.utils import rotate_non_max_suppression_cpu 

# TODO: Study More
def rotate_nms_cc(dets, thresh):
    # encodes rotation on the bounding box
    # dets: [x, y, w, l, r, score], shape[-1, 5]

    scores = dets[:, 5]
    order = scores.argsort()[::-1].astype(np.int32)  # highest->lowest
    dets_corners = box_np_ops.center_to_corner_box2d(dets[:, :2], dets[:, 2:4],
                                                     dets[:, 4])

    dets_standup = box_np_ops.corner_to_standup_nd(dets_corners)

    standup_iou = box_np_ops.iou_jit(dets_standup, dets_standup, eps=0.0)
    # print(dets_corners.shape, order.shape, standup_iou.shape)
    return rotate_non_max_suppression_cpu(dets_corners, order, standup_iou,
                                          thresh)


