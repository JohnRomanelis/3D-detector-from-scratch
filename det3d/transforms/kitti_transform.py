import numpy as np
import pickle
from configs.config_utils import configure_path

# Preprocessing
from det3d.detection_core import preprocess as prep 
from det3d.detection_core import box_np_ops
from det3d.detection_core import coord_sys_utils as csu

from det3d.builders import build_target_assigner
from det3d.builders import build_db_sampler
from det3d.builders import build_voxel_generator



def drop_arrays_by_name(gt_names, used_classes):
    # returns the indexes of the objects that don't belong
    # in the used_classes
    inds = [
        i for i, x in enumerate(gt_names) if x not in used_classes
    ]
    inds = np.array(inds, dtype = np.int64)
    return inds

def _dict_select(dict_, inds):
    # masks all lists/np.arrays in a dictionary 
    # based on the given indexes
    for k, v in dict_.items():
        if isinstance(v, dict):
            _dict_select(v, inds)
        else:
            dict_[k] = v[inds]


class PrepareKitti(object):
    """
        Returns a dictionary with the following form for each example:

         - voxels
         - num_points
         - coordinates
         - num_voxels
         - calib
         - anchors
         - gt_names

         # if training
         - labels
         - reg_targets
         - importance

        # if inference
         - points
         - gt_bboxes

    """

    def __init__(self, cfg, mode="training"):

        # selecting mode for the dataset
        assert mode in ["training", "evaluation", "inference"]
        #print(mode)
        if mode == "training":
            self.training = True
            self.inference = False
        elif mode == 'inference':
            self.inference = True
            self.training = True
        else:  # mode == "evaluation"
            self.training = False
            self.inference = False

        dataset_cfg = cfg.dataset
        self.root_path = configure_path(dataset_cfg.data_path, dataset_cfg.dataset_name)
        preprocess_cfg = cfg.transforms
        # ----- Create Voxel Generator ----- #
        self.voxel_generator = build_voxel_generator(preprocess_cfg.voxel_generator)

        # ----- Create Target Assigner ----- #
        self.target_assigner = build_target_assigner(preprocess_cfg.target_assigner)

        # ----- Create DataBase Sampler ----- #
        self.db_sampler, self.dbsampler_opts = build_db_sampler(preprocess_cfg.db_sampler)

        # NOTE: TODO: 
        feature_map_size = list(preprocess_cfg.target_assigner.anchor_generator.feature_map_size)
        self.num_point_features = dataset_cfg.num_point_features

        ## ------ Generating anchor cache ------- ##
        ret = self.target_assigner.generate_anchors(feature_map_size)
        
        class_names = self.target_assigner.classes
        anchors_dict = self.target_assigner.generate_anchors_dict(
            feature_map_size)


        anchors_list = []
        for k, v in anchors_dict.items():
            # adding anchors for each category to an anchor list
            # so that the list will contain all the anchors for all the categories
            anchors_list.append(v["anchors"])

        # then we concatenate the anchors of the list to one np.array
        # should be of shape [num_anchors_per_category * num_categoris, box_code_size]
        anchors = np.concatenate(anchors_list, axis=0)
        anchors = anchors.reshape([-1, self.target_assigner.box_ndim])

        # np.allclose: Returns True if two arrays are element-wise equal within a tolerance.
        assert np.allclose(
            anchors, ret["anchors"].reshape(-1, self.target_assigner.box_ndim))

        matched_thresholds = ret["matched_thresholds"]
        unmatched_thresholds = ret["unmatched_thresholds"]

        # box_np_ops.rbbox2d_to_near_bbox: converts rotated bbox to nearest
        #                                  'standing' or 'lying' bbox.
        anchors_bv = box_np_ops.rbbox2d_to_near_bbox(
            anchors[:, [0, 1, 3, 4, 6]])

        # anchor cache - contains information about the generated anchors
        self.anchor_cache = {
            "anchors": anchors,
            "anchors_bv": anchors_bv,
            "matched_thresholds": matched_thresholds,
            "unmatched_thresholds": unmatched_thresholds,
            "anchors_dict": anchors_dict,
        }
        


        ##----------------------------------------------##
        ## Reading rest options from configuration file ##
        ##----------------------------------------------##
        preprocess_cfg = preprocess_cfg.prepare_kitti
        self.shuffle_points = preprocess_cfg.shuffle_points
        self.min_points_in_gt = preprocess_cfg.min_points_in_gt
        self.max_voxels = preprocess_cfg.max_number_of_voxels
        self.remove_unkown = preprocess_cfg.remove_unkown
        self.use_group_id = False  # TODO: Find value
        self.gt_rotation_noise = list(
            preprocess_cfg.groundtruth_rotation_uniform_noise)
        self.gt_loc_noise_std = list(
            preprocess_cfg.groundtruth_localization_noise_std)
        self.global_random_rot_range = list(
            preprocess_cfg.global_random_rotation_range_per_object)
        self.global_rotation_noise = list(
            preprocess_cfg.global_rotation_uniform_noise)
        self.global_scaling_noise = list(
            preprocess_cfg.global_scaling_uniform_noise)
        self.global_translate_noise_std = list(
            preprocess_cfg.global_translate_noise_std)

        self.random_flip_x = preprocess_cfg.random_flip_x
        self.random_flip_y = preprocess_cfg.random_flip_y

        # For cylidrical anchors-voxels
        self.frustrum_box_filtering = preprocess_cfg.frustrum_box_filtering


    def __call__(self, input_dict):
        '''
        NOTE: input_dict should be in form of KittiDataset.get_sensor_data()
        '''
        # getting class names from target assigner
        class_names = self.target_assigner.classes
        #print("class names: ",class_names)

        # getting points from pointcloud
        points = input_dict['lidar']['points']


        if self.training:
            # getting ground truth data from the lidar
            anno_dict = input_dict["lidar"]["annotations"]
            gt_dict = {
                "gt_boxes": anno_dict["boxes"],
                "gt_names": anno_dict["names"],
                "gt_importance": np.ones([anno_dict["boxes"].shape[0]], dtype=anno_dict["boxes"].dtype),
            }
            if "difficulty" not in anno_dict:
                difficulty = np.zeros([anno_dict["boxes"].shape[0]],
                                      dtype=np.int32)
                gt_dict["difficulty"] = difficulty
            else:
                gt_dict["difficulty"] = anno_dict["difficulty"]


            if self.use_group_id and "group_ids" in anno_dict:
                group_ids = anno_dict["group_ids"]
                gt_dict["group_ids"] = group_ids



        # Reading calibration file
        calib = None
        if 'calib' in input_dict:
            calib = input_dict["calib"]


        
        if self.training:
            # removing don't care from ground truths
            selected = drop_arrays_by_name(
                gt_dict["gt_names"], ["DontCare"])
            # removing DontCare parameters from every array in the gt_names dictionary
            _dict_select(gt_dict, selected)  # keeps only the selected
            
            if self.remove_unkown:
                remove_mask = gt_dict['difficulty'] == -1
                keep_mask = np.logical_not(remove_mask)
                _dict_select(gt_dict, keep_mask)
            # we no longer need the difficulty indicator
            gt_dict.pop("difficulty")

            if self.min_points_in_gt > 0:
                # removing ground truths that have less than min_points_in_gt points
                point_counts = box_np_ops.points_count_rbbox(
                    points, gt_dict["gt_boxes"])
                mask = point_counts >= self.min_points_in_gt
                _dict_select(gt_dict, mask)

            gt_boxes_mask = np.array(
                [n in class_names for n in gt_dict["gt_names"]], dtype=np.bool_)

            #print(gt_dict['gt_names'])
            ##------------------##
            ##    DBSAMPLER!    ##
            ##------------------##
            # self.db_sampler = None # -> Uncomment for fast disable of dbsampler
            if self.db_sampler is not None:
                group_ids = None
                if group_ids in gt_dict:
                    group_ids = gt_dict["group_ids"]

                sampled_dict = self.db_sampler.sample_all(self.root_path,
                                                          gt_dict["gt_boxes"],
                                                          gt_dict["gt_names"],
                                                          self.num_point_features,
                                                          self.dbsampler_opts['random_crop'],
                                                          gt_group_ids=group_ids,
                                                          calib=calib)

                if sampled_dict is not None:
                    sampled_gt_names = sampled_dict["gt_names"]
                    sampled_gt_boxes = sampled_dict["gt_boxes"]
                    sampled_points = sampled_dict["points"]
                    sampled_gt_masks = sampled_dict["gt_masks"]
                    gt_dict["gt_names"] = np.concatenate(
                        [gt_dict["gt_names"], sampled_gt_names], axis=0)
                    gt_dict["gt_boxes"] = np.concatenate(
                        [gt_dict["gt_boxes"], sampled_gt_boxes])
                    gt_boxes_mask = np.concatenate(
                        [gt_boxes_mask, sampled_gt_masks], axis=0)
                    sampled_gt_importance = np.full(
                        [sampled_gt_boxes.shape[0]], self.dbsampler_opts['sample_importance'], dtype=sampled_gt_boxes.dtype)
                    gt_dict["gt_importance"] = np.concatenate(
                        [gt_dict["gt_importance"], sampled_gt_importance])
                    
                    if group_ids is not None:
                        sampled_group_ids = sampled_dict["group_ids"]
                        gt_dict["group_ids"] = np.concatenate(
                            [gt_dict["group_ids"], sampled_group_ids])

                    if self.dbsampler_opts['remove_points_after_sample']:
                        # removing points that were at the positinion 
                        # where we added the new gt objects
                        masks = box_np_ops.points_in_rbbox(points,
                                                           sampled_gt_boxes)
                        points = points[np.logical_not(masks.any(-1))]

                    # adding new sampled points to the main pointcloud
                    points = np.concatenate([sampled_points, points], axis=0)

            #print(gt_dict['gt_names'])
            group_ids = None
            if "group_ids" in gt_dict:
                group_ids = gt_dict["group_ids"]
            
            ##--------------------------##
            ##       Adding Noise       ##
            ##--------------------------##
            # prep.noise_per_object_v3_: random rotate or remove each groundtrutn independently
            prep.noise_per_object_v3_(
                gt_dict["gt_boxes"],
                points,
                gt_boxes_mask,
                rotation_perturb=self.gt_rotation_noise,
                center_noise_std=self.gt_loc_noise_std,
                global_random_rot_range=self.global_random_rot_range,
                group_ids=group_ids,
                num_try=100)

            # should remove unrelated objects after noise per object
            _dict_select(gt_dict, gt_boxes_mask)
            #print(gt_dict["gt_names"])
            #print(class_names)
            gt_classes = np.array(
                [class_names.index(n) + 1 for n in gt_dict["gt_names"]],
                dtype=np.int32)
            #print(gt_classes)
            gt_dict["gt_classes"] = gt_classes


            # Random flip objects
            gt_dict["gt_boxes"], points = prep.random_flip(gt_dict["gt_boxes"],
                                                        points, 0.5, self.random_flip_x, self.random_flip_y)
            # Apply global rotation
            gt_dict["gt_boxes"], points = prep.global_rotation_v2(
                gt_dict["gt_boxes"], points, *self.global_rotation_noise)
            # Apply global scaling
            gt_dict["gt_boxes"], points = prep.global_scaling_v2(
                gt_dict["gt_boxes"], points, *self.global_scaling_noise)
            # global translation
            prep.global_translate_(gt_dict["gt_boxes"], points, self.global_translate_noise_std)
            bv_range = self.voxel_generator.point_cloud_range[[0, 1, 3, 4]]
            # masking boxes out of detection range
            if self.frustrum_box_filtering:
                # for cylidrical coordinates
                boxes_cyl = csu.cart2cyl_with_features_numpy(gt_dict['gt_boxes'].copy())
                mask = prep.filter_gt_box_outside_range_by_center(boxes_cyl, bv_range)
            else:
                mask = prep.filter_gt_box_outside_range_by_center(gt_dict["gt_boxes"], bv_range)
            _dict_select(gt_dict, mask)

            # limit rad to [-pi, pi]
            gt_dict["gt_boxes"][:, 6] = box_np_ops.limit_period(
                gt_dict["gt_boxes"][:, 6], offset=0.5, period=2 * np.pi)

        if self.shuffle_points:
            # shuffle is a little slow.
            np.random.shuffle(points)


        # [352, 400]
        # generating voxels
        res = self.voxel_generator.generate(
            points, self.max_voxels)
        voxels = res["voxels"]
        coordinates = res["coordinates"]
        num_points = res["num_points_per_voxel"]
        num_voxels = np.array([voxels.shape[0]], dtype=np.int64)

        example = {
            'voxels': voxels,
            'num_points': num_points,
            'coordinates': coordinates,
            "num_voxels": num_voxels,
        }
        if calib is not None:
            example["calib"] = calib

        # Is defined in __init__ so it will not be none!
        # if self.anchor_cache is not None:
        anchors = self.anchor_cache["anchors"]
        anchors_bv = self.anchor_cache["anchors_bv"]
        anchors_dict = self.anchor_cache["anchors_dict"]
        matched_thresholds = self.anchor_cache["matched_thresholds"]
        unmatched_thresholds = self.anchor_cache["unmatched_thresholds"]

        # addding anchors to example dict
        example["anchors"] = anchors
        

        if not self.training:
            return example


        example["gt_names"] = gt_dict["gt_names"]

        anchors_mask = None

        # If we were on inference of evaluation mode
        # example would have been returned
        # if create_targets: # create_targets should be true if self.training=true
        targets_dict = self.target_assigner.assign(
            anchors,
            anchors_dict,
            gt_dict["gt_boxes"],
            anchors_mask,
            gt_classes=gt_dict["gt_classes"],
            gt_names=gt_dict["gt_names"],
            matched_thresholds=matched_thresholds,
            unmatched_thresholds=unmatched_thresholds,
            importance=gt_dict["gt_importance"])

        #keeps = targets_dict['labels'] == 1
        #print(targets_dict['bbox_targets'][keeps, :])
        #print(targets_dict['labels'].shape) # (70400, )
        

        example.update({
            'labels': targets_dict['labels'],
            'reg_targets': targets_dict['bbox_targets'],
            'importance': targets_dict['importance'],
        })

        ### TESTING AREA ###

        # print('ground truth boxes : ', gt_dict["gt_boxes"].shape)
        # import open3d as o3d
        # render_list = []
        # #print(points.shape)
        # pcd = o3d.geometry.PointCloud()
        # pcd.points = o3d.utility.Vector3dVector(points[:, :3])
        # render_list.append(pcd)

        # anchors_r = anchors[..., :3].copy()
        # anchors_r[..., 2] = anchors_r[..., 2] - 2 
        # pcd2 = o3d.geometry.PointCloud()
        # pcd2.points = o3d.utility.Vector3dVector(anchors_r[:, :3])
        # render_list.append(pcd2)

        # keep_anchors = anchors[example['labels'] == 1]
        # print(keep_anchors.shape)

        # if keep_anchors.shape[0] > 0:
        #     pcd3 = o3d.geometry.PointCloud()
        #     pcd3.points = o3d.utility.Vector3dVector(keep_anchors[:, :3])
        #     pcd3.paint_uniform_color([1.0, 0.0, 0.0])
        #     render_list.append(pcd3)

        

        # o3d.visualization.draw_geometries(render_list)
     
        
        

        if self.inference:
            example['points'] = points
            example['gt_bboxes'] = gt_dict["gt_boxes"]

        return example