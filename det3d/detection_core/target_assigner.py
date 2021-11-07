from collections import OrderedDict
import numpy as np

from .target_ops import create_target_np


class TargetAssigner(object):

    def __init__(self, 
                 box_coder, 
                 anchor_generators, 
                 classes, 
                 feature_map_sizes, #list of feature map sizes 
                 positive_fraction=None, 
                 region_similarity_calculators=None,
                 sample_size=512, 
                 assign_per_class=True):

        if not isinstance(anchor_generators, (list, tuple)):
            anchor_generators = [anchor_generators]
        if not isinstance(feature_map_sizes, (list, tuple)):
            feature_map_sizes = [feature_map_sizes]

        self._box_coder = box_coder
        self._anchor_generators = anchor_generators

        if not isinstance(region_similarity_calculators, (list, tuple)):
            region_similarity_calculators = [region_similarity_calculators]

        self._sim_calcs = region_similarity_calculators
        box_ndims = [a.ndim for a in anchor_generators]

        assert all(e==box_ndims[0] for e in box_ndims)

        self._positive_fraction = positive_fraction
        self._sample_size = sample_size
        self._classes = classes
        self._assign_per_class = assign_per_class
        self._feature_map_sizes = feature_map_sizes


    @property
    def box_coder(self):
        return self._box_coder

    @property
    def classes(self):
        return self._classes

    @property
    def box_ndim(self):
        return self._anchor_generators[0].ndim

    def assign(self, 
               anchors, 
               anchors_dict, 
               gt_boxes, 
               anchors_mask=None,
               gt_classes=None,
               gt_names=None, 
               matched_thresholds=None, 
               unmatched_thresholds=None, 
               importance=None):
        """
            Function to assing the ground truth values to the anchors.
            Calls either:
                - assign_per_class: assigns targets individually for each class
                - assign_all: not implemented in this version yet!
        """
        if self.assign_per_class:
            return self.assign_per_class(anchors_dict, gt_boxes, anchors_mask,
                                         gt_classes, gt_names, importance=importance)
        else: 
            raise NotImplementedError #, "Check old version for implementation"


    def assign_per_class(self,
                         anchors_dict, 
                         gt_boxes, 
                         anchors_mask=None, 
                         gt_classes=None,
                         gt_names=None, 
                         importance=None):
        """
            This function assigns target individually for each class. 
            (Recommended for multiclass network)
        """

        # calling the coder through a function 
        # to reduce complexity
        def box_encoding_fn(boxes, anchors):
            return self._box_coder.encode(boxes, anchors)

        target_list = []
        anchor_loc_idx = 0
        anchor_gene_idx = 0

        # for each class and the generated anchors for this class
        # NOTE: Different classes may have different number of anchors
        #       due to differnces in anchor size, search range etc
        for class_name, anchor_dict in anchors_dict.items():

            # defining similarity function for each class
            def similarity_fn(anchors, gt_boxes):
                # anchors with rotation in bev
                anchors_rbv = anchors[:, [0, 1, 3, 4, 6]] # missing indexes: 2, 5
                # gt_boxes with rotation in bev
                gt_boxes_rbv = gt_boxes[:, [0, 1, 3, 4, 6]]
                # Comparing the anchors to gt boxes to find the activated anchors
                # (Will use nms algorithm)
                return self._sim_calcs[anchor_gene_idx].compute(
                    anchors_rbv, gt_boxes_rbv)


            # to mask the ground truths - keep only ground truths 
            # that correspond to the curent class
            mask = np.array([c == class_name for c in gt_names], dtype=np.bool_)

            # Number of anchors per axis: x, y, z
            feature_map_size = anchor_dict["anchors"].shape[:3]
            # Number of different rotation or different sizes per anchor
            num_loc = anchor_dict["anchors"].shape[-2]


            # If there is anchor mask provided, then create a prune fucntion
            # that will filter out the anchors defined by the mask
            if anchors_mask is not None:
                anchors_mask = anchors_mask.reshape(-1)
                a_range = self.anchors_range(class_name)
                anchors_mask_class = anchors_mask[a_range[0]:a_range[1]].reshape(-1)
                prune_anchor_fn = lambda _: np.where(anchors_mask_class)[0]
            else:
                prune_anchor_fn = None


            targets = create_target_np(
                anchor_dict['anchors'].reshape(-1, self.box_ndim),
                gt_boxes[mask],
                similarity_fn,
                box_encoding_fn, 
                prune_anchor_fn=prune_anchor_fn, 
                gt_classes=gt_classes[mask],
                matched_threshold=anchor_dict['matched_thresholds'],
                unmatched_threshold=anchor_dict["unmatched_thresholds"],
                positive_fraction=self._positive_fraction,
                rpn_batch_size=self._sample_size,
                norm_by_num_examples=False,
                box_code_size=self.box_coder.code_size,
                gt_importance=importance
            )
            
            anchor_loc_idx += num_loc
            target_list.append(targets)
            anchor_gene_idx += 1

        
        # ----- After the above loop has ended ----- #

        # Creating the target dict
        targets_dict = {
            'labels': [t['labels'] for t in target_list],
            'bbox_targets': [t['bbox_targets'] for t in target_list],
            'importance': [t['importance'] for t in target_list]
        }
        
        # formating bbox_targets
        targets_dict['bbox_targets'] = np.concatenate([
            v.reshape(-1, self.box_coder.code_size)
            for v in targets_dict['bbox_targets']
        ], axis=0)
        targets_dict['bbox_targets'] = targets_dict['bbox_targets'].reshape(
            -1, self.box_coder.code_size)
        
        # formating labels
        targets_dict['labels'] = np.concatenate([
            v.reshape(-1) 
            for v in targets_dict['labels']
        ], axis=0)
        targets_dict['labels'] = targets_dict['labels'].reshape(-1)

        # formating importance
        targets_dict['importance'] = np.concatenate([
            v.reshape(-1) for v in targets_dict['importance']
        ], axis=0)
        targets_dict['importance'] = targets_dict['importance'].reshape(-1)

        return targets_dict

    

    def generate_anchors(self, feature_map_size):
        
        anchor_list = []
        ndim = len(feature_map_size)
        matched_thresholds = [
            a.match_threshold for a in self._anchor_generators
        ]
        unmatched_thresholds = [
            a.unmatch_threshold for a in self._anchor_generators
        ]

        match_list, unmatched_list = [], []
        
        if self._feature_map_sizes is not None:
            feature_map_sizes = self._feature_map_sizes 
        else: 
            feature_map_sizes = [feature_map_size] * len(self._anchor_generators)

        idx=0

        for anchor_generator, match_thresh, unmatch_thresh, fsize in zip(
                self._anchor_generators, matched_thresholds,
                unmatched_thresholds, feature_map_sizes):
            if len(fsize) == 0:
                fsize = feature_map_size
                self._feature_map_sizes[idx] = feature_map_size

            # NOTE: Weird permutation -> maybe change
            # reshape is [2, 1, 200, 176, 7]
            anchors = anchor_generator.generate(fsize)
            anchors = anchors.reshape([*fsize, -1, self.box_ndim])
            anchors = anchors.transpose(ndim, *range(0, ndim), ndim+1)

            anchor_list.append(anchors.reshape(-1, self.box_ndim))
            num_anchors = np.prod(anchors.shape[:-1])
            match_list.append(
                np.full([num_anchors], match_thresh, anchors.dtype)
            )
            unmatched_list.append(
                np.full([num_anchors], unmatch_thresh, anchors.dtype)
            )
            idx += 1

        anchors = np.concatenate(anchor_list, axis=0)
        matched_thresholds = np.concatenate(match_list, axis=0)
        unmatched_thresholds = np.concatenate(unmatched_list, axis=0)
        
        return {
            "anchors": anchors,
            "matched_thresholds": matched_thresholds,
            "unmatched_thresholds": unmatched_thresholds
        }


    def generate_anchors_dict(self, feature_map_size):
        ndim = len(feature_map_size)
        anchors_list = []
        matched_thresholds = [
            a.match_threshold for a in self._anchor_generators
        ]
        unmatched_thresholds = [
            a.unmatch_threshold for a in self._anchor_generators
        ]
        
        match_list, unmatch_list = [], []
        # NOTE: OrderedDict preserves the order in which the keys are inserted.
        anchors_dict = OrderedDict()

        # creating an empty dict for each class
        # NOTE: each anchor generator 
        for a in self._anchor_generators:
            anchors_dict[a.class_name] = {}
        if self._feature_map_sizes is not None:
            feature_map_sizes = self._feature_map_sizes
        else:
            feature_map_sizes = [feature_map_size] * len(self._anchor_generators)
        
        idx = 0

        for anchor_generator, match_thresh, unmatch_thresh, fsize in zip(
                self._anchor_generators, matched_thresholds,
                unmatched_thresholds, feature_map_sizes):
            
            if len(fsize) == 0:
                fsize = feature_map_size
                self._feature_map_sizes[idx] = feature_map_size
            
            # generating anchors
            anchors = anchor_generator.generate(fsize)
            anchors = anchors.reshape([*fsize, -1, self.box_ndim])
            anchors = anchors.transpose(ndim, *range(0, ndim), ndim + 1)
            num_anchors = np.prod(anchors.shape[:-1])
            match_list.append(
                np.full([num_anchors], match_thresh, anchors.dtype))
            unmatch_list.append(
                np.full([num_anchors], unmatch_thresh, anchors.dtype))
            class_name = anchor_generator.class_name
            anchors_dict[class_name]["anchors"] = anchors.reshape(-1, self.box_ndim)
            anchors_dict[class_name]["matched_thresholds"] = match_list[-1]
            anchors_dict[class_name]["unmatched_thresholds"] = unmatch_list[-1]
            idx += 1

        return anchors_dict


    def anchors_range(self, class_name):
        # getting the class name
        if isinstance(class_name, int):
            class_name = self._classes[class_name]
        assert class_name in self._classes


        num_anchors = 0
        anchor_ranges = []
        for name in self._classes:
            anchor_ranges.append((num_anchors, num_anchors+self.num_anchors(name)))
            num_anchors += anchor_ranges[-1][1] - num_anchors
        
        return anchor_ranges[self._classes.index(class_name)]


    @property
    def num_anchors_per_location(self):
        num = 0
        for a_generator in self._anchor_generators:
            num += a_generator.num_anchors_per_localization
        return num