import torch
import torch.nn as nn

import numpy as np

from det3d.detection_core import box_torch_ops
from det3d.detection_core.box_coder import GroundBox3dCoderTorch


class DetectionNetwork(nn.Module):

    def predict(self, example, preds_dict):
        """
        Args: 
             - example: input to the network
             - preds_dict: network output

        Returns:
             - a list of dictionaries, for each pointcloud in the batch.

        """
        ##------------------##
        ## HardCoded Values ##
        ##------------------##
        num_class_with_bg = 1  # num classes with backround
        _encode_background_as_zeros = True  # NOT DIFINED IN THE CONFIGURATION FILE!
        use_direction_classifier = True
        num_direction_bins = True
        post_center_range = None  # [-1]
        use_sigmoid_score = True  # for classification - choosing between backround and classes
        # if false (used softmax instead)
        num_anchors_per_location = 2
        _nms_score_threshold = [0.3]  # use smaller value in early epochs
        _nms_pre_max_size = [1000]
        _nms_post_max_size = [100]
        _nms_iou_thresholds = [0.01]
        _use_rotate_nms = True
        _dir_offset = 0  # TODO resolve
        _dir_limit_offset = 1  # direction_limit_offset
        box_coder = GroundBox3dCoderTorch(linear_dim=False, vec_encode=False)


        # getting the batch_size
        batch_size = example['anchors'].shape[0]

        # we need metadata to load the original pointcloud for visualization
        if "metadata" not in example or len(example["metadata"]) == 0:
            meta_list = [None] * batch_size
        else:
            meta_list = example["metadata"]

        ##---------##
        ## ANCHORS ##
        ##---------##
        # coder size
        coder_size = example['anchors'].shape[-1]
        # getting anchors as shape [B, -1, coder_size]
        batch_anchors = example["anchors"].view(batch_size, -1, coder_size)

        ##-------------##
        ## Predictions ##
        ##-------------##
        # Regression Predictions
        batch_box_preds = preds_dict["box_preds"]
        batch_box_preds = batch_box_preds.reshape(batch_size, -1, coder_size)
                                          #view
    
        # Decoding box predictions
        batch_box_preds = box_coder.decode_torch(
            batch_box_preds, batch_anchors)

        # if backround is represented as a class and not as zeros
        # increase the number of classes by 1
        if not _encode_background_as_zeros:
            num_class_with_bg += 1

        # Classification Predictions
        batch_cls_preds = preds_dict['cls_preds']
        batch_cls_preds.reshape(batch_size, -1, num_class_with_bg)
                        #view

        # Direction Predictions
        if use_direction_classifier:
            batch_dir_preds = preds_dict["dir_cls_preds"]
            batch_dir_preds = batch_dir_preds.view(
                batch_size, -1, num_direction_bins)
        else:
            batch_dir_preds = [None] * batch_size

        ##-----------------##
        ## Prediction Core ##
        ##-----------------##
        predictions_dicts = []

        # For every Point Cloud in the batch
        for box_preds, cls_preds, dir_preds, meta in zip(
                batch_box_preds, batch_cls_preds, batch_dir_preds, meta_list):

            box_preds = box_preds.float()
            cls_preds = cls_preds.float()

            if use_direction_classifier:
                # getting direction predictions
                dir_preds = dir_preds.reshape(-1, 2)  # my addition
                dir_labels = torch.max(dir_preds, dim=-1)[1]

            # getting classification predictions
            if _encode_background_as_zeros:
                # doen't work with softmax
                assert use_sigmoid_score is True
                total_scores = torch.sigmoid(cls_preds)
            else:
                # encode backround as first element in one-hot vector
                if use_sigmoid_score:
                    total_scores = torch.sigmoid(cls_preds)[..., 1:]
                else:
                    # TODO: replace torch.nn.functional with F
                    total_scores = torch.nn.functional.softmax(
                        cls_preds)[..., 1:]

            # Apply NMS in birdeye view
            # Keeping prediction per object with the highest accuracy score
            if _use_rotate_nms: 
                nms_func = box_torch_ops.rotate_nms
            else:
                nms_func = box_torch_ops.nms

            # self.target_assigner.num_anchors_per_location
            feature_map_size_prod = batch_box_preds.shape[1] // num_anchors_per_location

            if num_class_with_bg == 1:
                total_scores = total_scores.reshape(-1, 1)  # my addition
                top_scores = total_scores.squeeze(-1)
                top_labels = torch.zeros(
                    total_scores.shape[0],
                    device=total_scores.device,
                    dtype=torch.long)
            else:
                top_scores, top_labels = torch.max(total_scores, dim=-1)

            # NOTE: Filters out predictions with score lower than the _nms_score_threshold
            # Having an initial value of 0.3 is very high for early epochs

            if _nms_score_threshold[0] > 0.0:
                top_scores_keep = top_scores > _nms_score_threshold[0]
                top_scores = top_scores.masked_select(top_scores_keep)

            if top_scores.shape[0] != 0:
                if _nms_score_threshold[0] > 0.0:
                    top_scores_keep = top_scores_keep.reshape(-1, )

                    box_preds = box_preds[top_scores_keep]
                    if use_direction_classifier:
                        dir_labels = dir_labels[top_scores_keep]
                    top_labels = top_labels[top_scores_keep]

                # turning 3d bounding box to 2d bev bounding box
                boxes_for_nms = box_preds[:, [0, 1, 3, 4, 6]]
                if not _use_rotate_nms:
                    box_preds_corners = box_torch_ops.center_to_corner_box2d(
                        boxes_for_nms[:, :2], boxes_for_nms[:, 2:4],
                        boxes_for_nms[:, 4])
                    boxes_for_nms = box_torch_ops.corner_to_standup_nd(
                        box_preds_corners)
                # the nms in 3d detection just remove overlap boxes.
                selected = nms_func(
                    boxes_for_nms,
                    top_scores,
                    pre_max_size=_nms_pre_max_size[0],
                    post_max_size=_nms_post_max_size[0],
                    iou_threshold=_nms_iou_thresholds[0],
                )
            else:
                selected = []

            selected_boxes = box_preds[selected]
            if use_direction_classifier:
                selected_dir_labels = dir_labels[selected]

            selected_labels = top_labels[selected]
            selected_scores = top_scores[selected]

            # Generate Predictions
            if selected_boxes.shape[0] != 0:
                box_preds = selected_boxes
                scores = selected_scores
                label_preds = selected_labels

                if use_direction_classifier:
                    dir_labels = selected_dir_labels
                    period = (2 * np.pi / num_direction_bins)
                    dir_rot = box_torch_ops.limit_period(
                        box_preds[..., 6] - _dir_offset,
                        _dir_limit_offset, period)
                    box_preds[
                        ...,
                        6] = dir_rot + _dir_offset + period * dir_labels.to(
                        box_preds.dtype)

                final_box_preds = box_preds
                final_scores = scores
                final_labels = label_preds

                # NOTE: Currently not using post_center_range
                if post_center_range is not None:
                    mask = (final_box_preds[:, :3] >=
                            post_center_range[:3]).all(1)
                    mask &= (final_box_preds[:, :3] <=
                             post_center_range[3:]).all(1)
                    predictions_dict = {
                        "box3d_lidar": final_box_preds[mask],
                        "scores": final_scores[mask],
                        "label_preds": label_preds[mask],
                        "metadata": meta,
                    }
                else:
                    predictions_dict = {
                        "box3d_lidar": final_box_preds,
                        "scores": final_scores,
                        "label_preds": label_preds,
                        "metadata": meta,
                    }
            else:
                dtype = batch_box_preds.dtype
                device = batch_box_preds.device
                predictions_dict = {
                    "box3d_lidar":
                    torch.zeros([0, box_preds.shape[-1]],
                                dtype=dtype,
                                device=device),
                    "scores":
                    torch.zeros([0], dtype=dtype, device=device),
                    "label_preds":
                    torch.zeros([0], dtype=top_labels.dtype, device=device),
                    "metadata":
                    meta,
                }
            predictions_dicts.append(predictions_dict)

        return predictions_dicts
