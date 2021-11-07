from .kitti_utils import get_path, read_split_file
from det3d.detection_core import box_np_ops as np_box_ops

import pickle
from pathlib import Path
from skimage import io
import concurrent.futures as futures
import numpy as np

def np_extend_matrix(mat):
    # given a matrix, we add a new dimension 
    # representing the homogenus coordinates

    if mat.shape==(3,4):
        mat = np.concatenate([mat, np.array([[0., 0., 0., 1.]])], axis=0)

    elif mat.shape[0]==mat.shape[1]:
        ext_dim = mat.shape[0] + 1
        mat_new = np.zeros([ext_dim, ext_dim], dtype=mat.dtype)
        mat_new[ext_dim-1, ext_dim-1] = 1
        mat_new[:ext_dim-1, :ext_dim-1] = mat
        mat = mat_new

    else: 
        print("Error in extend_matrix,\n*Case not implemented yet")
        raise NotImplementedError

    return mat

def add_difficulty_to_annos(infos):
    # minimum height for evaluated groundtruth/detections
    min_height = [40, 25, 25]
    # maximum occlusion level of the groundtruth used for evaluation
    max_occlusion = [0, 1, 2]
    # maximum truncation level of the groundtruth used for evaluation
    max_trunc = [0.15, 0.3, 0.5]

    annos = infos['annos']

    dims = annos['dimensions'] # lhw format
    bbox = annos['bbox']
    height = bbox[:, 3] - bbox[:, 1]
    occlusion = annos['occluded']
    truncation = annos['truncated']
    diff = []
    # creating masks
    easy_mask = np.ones((len(dims), ), dtype=np.bool)
    moderate_mask = np.ones((len(dims), ), dtype=np.bool)
    hard_mask = np.ones((len(dims), ), dtype=np.bool)

    i = 0
    for h, o, t in zip(height, occlusion, truncation):
        if o > max_occlusion[0] or h <= min_height[0] or t > max_trunc[0]:
            easy_mask[i] = False
        if o > max_occlusion[1] or h <= min_height[1] or t > max_trunc[1]:
            moderate_mask[i] = False
        if o > max_occlusion[2] or h <= min_height[2] or t > max_trunc[2]:
            hard_mask[i] = False
        i += 1
    is_easy = easy_mask
    is_moderate = np.logical_xor(easy_mask, moderate_mask)
    is_hard = np.logical_xor(hard_mask, moderate_mask)

    for i in range(len(dims)):
        if is_easy[i]:
            diff.append(0)
        elif is_moderate[i]:
            diff.append(1)
        elif is_hard[i]:
            diff.append(2)
        else:
            diff.append(-1)
    annos["difficulty"] = np.array(diff, np.int32)
    
    return diff
    # Done

def get_label_anno(label_path):
    annotations = {}
    annotations.update({
        'name': [],
        'truncated': [],
        'occluded': [],
        'alpha': [],
        'bbox': [],
        'dimensions': [],
        'location': [],
        'rotation_y': []
    })
    # reading the label file 
    with open(label_path, 'r') as f:
        lines = f.readlines()
    
    content = [line.strip().split(' ') for line in lines] 
    num_objects = len([x[0] for x in content if x[0] != 'DontCare'])
    annotations['name'] = np.array([x[0] for x in content]) # will include "ContCare"
    num_gt = len(annotations['name']) # number of bounding boxes including "ContCare"
    annotations['truncated'] = np.array([float(x[1]) for x in content])
    annotations['occluded'] = np.array([float(x[2]) for x in content])
    annotations['alpha'] = np.array([float(x[3]) for x in content])
    annotations['bbox'] = np.array(
        [[float(info) for info in x[4:8]] for x in content]).reshape(-1, 4)
    # dimensions will convert hwl format to standard lhw(camera) format.
    annotations['dimensions'] = np.array(
        [[float(info) for info in x[8:11]] for x in content]).reshape(
            -1, 3)[:, [2, 0, 1]]
    annotations['location'] = np.array(
        [[float(info) for info in x[11:14]] for x in content]).reshape(-1, 3)
    annotations['rotation_y'] = np.array(
        [float(x[14]) for x in content]).reshape(-1)   

    if len(content) != 0 and len(content[0]) == 16:  # have score
        annotations['score'] = np.array([float(x[15]) for x in content])
    else:
        annotations['score'] = np.zeros((annotations['bbox'].shape[0], ))
    
    # enumerating ground truth objects and adding a -1 label on DontCare
    index = list(range(num_objects)) + [-1] * (num_gt - num_objects)
    annotations['index'] = np.array(index, dtype=np.int32)
    # enumerating all objects including DontCare
    annotations['group_ids'] = np.arange(num_gt, dtype=np.int32)
    
    return annotations
    # Done

def get_kitti_image_info(path, 
                        training=True,
                        label_info=True, 
                        velodyne=False, 
                        calib=False, 
                        image_ids=7481,
                        extend_matrix=True, 
                        num_workers=8, 
                        relative_path=True, 
                        with_imageshape=True):
    """ 
    KITTI annotation format version 2:
    {
        [optional]points: [N, 3+] point cloud
        [optional, for kitti]image: {
            image_idx: ...
            image_path: ...
            image_shape: ...
        }
        point_cloud: {
            num_features: 4
            velodyne_path: ...
        }
        [optional, for kitti]calib: {
            R0_rect: ...
            Tr_velo_to_cam: ...
            P2: ...
        }
        annos: {
            location: [num_gt, 3] array
            dimensions: [num_gt, 3] array
            rotation_y: [num_gt] angle array
            name: [num_gt] ground truth name array
            [optional]difficulty: kitti difficulty
            [optional]group_ids: used for multi-part object
        }
    }
    """
    root_path = Path(path)
    if not isinstance(image_ids, list):
        image_ids = list(range(image_ids))
    
    def map_func(idx):
        info = {}
        pc_info = {'num_features': 4}
        calib_info = {}

        image_info = {'image_idx': idx}
        annotations = None

        if velodyne:
            pc_info['velodyne_path'] = get_path(idx, path, training, "velodyne", relative_path)

        image_info["image_path"] = get_path(idx, path, training, "image_2", relative_path)

        if with_imageshape:
            img_path = image_info['image_path']
            #if relative_path:
            #    img_path = str(root_path / img_path)
            image_info['image_shape'] = np.array(
                io.imread(img_path).shape[:2], dtype=np.int32
            )

        if label_info:
            label_path = get_path(idx, path, training, "label_2", relative_path)
            #if relative_path:
            #    label_path=str(root_path/label_path)
            annotations = get_label_anno(label_path)

        info['image'] = image_info
        info['point_cloud'] = pc_info

        if calib:
            calib_path = get_path(idx, path, training, "calib", relative_path=False)

            # reading calib file
            with open(calib_path, 'r') as f:
                lines = f.readlines()
            P0 = np.array(
                [float(info) for info in lines[0].split(' ')[1:13]]).reshape([3, 4])
            P1 = np.array(
                [float(info) for info in lines[1].split(' ')[1:13]]).reshape(
                    [3, 4])
            P2 = np.array(
                [float(info) for info in lines[2].split(' ')[1:13]]).reshape(
                    [3, 4])
            P3 = np.array(
                [float(info) for info in lines[3].split(' ')[1:13]]).reshape(
                    [3, 4])

            if extend_matrix:
                P0 = np_extend_matrix (P0)
                P1 = np_extend_matrix (P1)
                P2 = np_extend_matrix (P2)
                P3 = np_extend_matrix (P3)

            R0_rect = np.array([
                float(info) for info in lines[4].split(' ')[1:10]
            ]).reshape([3, 3])

            if extend_matrix:
                rect_4x4 = np.zeros([4, 4], dtype=R0_rect.dtype)
                rect_4x4[3, 3] = 1
                rect_4x4[:3, :3] = R0_rect
            else:
                rect_4x4 = R0_rect 

            Tr_velo_to_cam = np.array([
                float(info) for info in lines[5].split(' ')[1:13]
            ]).reshape([3, 4])

            Tr_imu_to_velo = np.array([
                float(info) for info in lines[6].split(' ')[1:13]
            ]).reshape([3, 4])

            if extend_matrix:
                Tr_velo_to_cam = np_extend_matrix (Tr_velo_to_cam)
                Tr_imu_to_velo = np_extend_matrix (Tr_imu_to_velo)
            
            # adding data to calib dictionary
            calib_info['P0'] = P0
            calib_info['P1'] = P1
            calib_info['P2'] = P2
            calib_info['P3'] = P3
            calib_info['R0_rect'] = rect_4x4
            calib_info['Tr_velo_to_cam'] = Tr_velo_to_cam
            calib_info['Tr_imu_to_velo'] = Tr_imu_to_velo

            # adding calib to main info dict
            info["calib"] = calib_info
            
        if annotations is not None:
            info['annos'] = annotations
            # makes adding difficulty indicators inside the info dict
            add_difficulty_to_annos(info)

        return info
    
    with futures.ThreadPoolExecutor(num_workers) as executor:
        image_infos = executor.map(map_func, image_ids)
    
    return list(image_infos)
    # Done

def _calculate_num_points_in_gt(data_path, 
                                infos, 
                                relative_path, 
                                remove_outside=True, 
                                num_features=4):

    for info in infos:
        # getting sub-dicts
        pc_info = info["point_cloud"]
        image_info = info["image"]
        calib = info["calib"]

        # resolving path
        if relative_path:
            v_path = str(Path(data_path) / pc_info["velodyne_path"])
        else:
            v_path = pc_info["velodyne_path"]
        
        # reading point cloud data
        points_v = np.fromfile(
            v_path, dtype=np.float32, count=-1).reshape([-1, num_features])
        
        rect = calib['R0_rect']
        Trv2c = calib['Tr_velo_to_cam']
        P2 = calib['P2']
        if remove_outside:
            points_v = np_box_ops.remove_outside_points(
                points_v, rect, Trv2c, P2, image_info['image_shape'])

        annos = info['annos']
        num_obj = len([n for n in annos['name'] if n != 'DontCare'])
        dims = annos['dimensions'][:num_obj]
        locs = annos['location'][:num_obj]
        rots = annos['rotation_y'][:num_obj]
        gt_boxes_camera = np.concatenate([locs, dims, rots[..., np.newaxis]], axis=1)
        gt_boxes_lidar = np_box_ops.box_camera_to_lidar(
            gt_boxes_camera, rect, Trv2c)

        indices = np_box_ops.points_in_rbbox(points_v[:,:3], gt_boxes_lidar)
        num_points_in_gt = indices.sum(0)
        num_ignored = len(annos['dimensions']) - num_obj
        num_points_in_gt = np.concatenate(
            [num_points_in_gt, -np.ones([num_ignored])])
        annos['num_points_in_gt'] = num_points_in_gt.astype(np.int32)

        # Done
 
def create_kitti_info_file(data_path, save_path=None, relative_path=True):

    # locating the folder containing the split information
    # Uncomment this lines if dataset splits are in: det3d/datasets/dataset_name 
    # split_folder = Path(__file__).resolve().parent / "kittiSplits" 
    # train_img_ids = read_split_file(split_folder / "train.txt")
    # val_img_ids = read_split_file(split_folder / "val.txt")
    # test_img_ids = read_split_file(split_folder / "test.txt")

    split_folder = Path(data_path) / "kittiSplits" 
    train_img_ids = read_split_file(split_folder / "train.txt")
    val_img_ids = read_split_file(split_folder / "val.txt")
    test_img_ids = read_split_file(split_folder / "test.txt")



    print("Generating Info. This may take sevelar minutes.")
    
    # selecting save path
    if save_path is None:
        save_path = Path(data_path)
    else: 
        save_path = Path(save_path)
    # creating info files
    # train_infos
    kitti_infos_train = get_kitti_image_info(
        data_path, 
        training=True, 
        velodyne=True,
        calib=True, 
        image_ids=train_img_ids, 
        relative_path=relative_path
    )
    _calculate_num_points_in_gt(data_path, kitti_infos_train, relative_path)
    filename = save_path / 'kitti_infos_train.pkl'
    # saving file
    with open(filename, 'wb') as f:
        pickle.dump(kitti_infos_train, f)
    print(f"Kitti info train file is saved to {filename}")

    # validation_infos
    kitti_infos_val = get_kitti_image_info(
        data_path,
        training=True,
        velodyne=True,
        calib=True,
        image_ids=val_img_ids,
        relative_path=relative_path)
    _calculate_num_points_in_gt(data_path, kitti_infos_val, relative_path)
    filename = save_path / 'kitti_infos_val.pkl'
    with open(filename, 'wb') as f:
        pickle.dump(kitti_infos_val, f)
    print(f"Kitti info val file is saved to {filename}")

    # trainval_infos
    filename = save_path / 'kitti_infos_trainval.pkl'
    with open(filename, 'wb') as f:
        pickle.dump(kitti_infos_train + kitti_infos_val, f)
    print(f"Kitti info trainval file is saved to {filename}")

    # test_infos
    kitti_infos_test = get_kitti_image_info(
        data_path,
        training=False,
        label_info=False,
        velodyne=True,
        calib=True,
        image_ids=test_img_ids,
        relative_path=relative_path)
    filename = save_path / 'kitti_infos_test.pkl'
    with open(filename, 'wb') as f:
        pickle.dump(kitti_infos_test, f)
    print(f"Kitti info test file is saved to {filename}")
    
    # Done


def _create_reduced_point_cloud(data_path, 
                                info_path, 
                                save_path=None,
                                back=False):
    
    with open(info_path, 'rb') as f:
        kitti_infos = pickle.load(f)
    
    for info in kitti_infos:
        pc_info = info["point_cloud"]
        image_info = info["image"]
        calib = info["calib"]

        v_path = Path(pc_info['velodyne_path'])
        # v_path = Path(data_path) / v_path # in this version full path is saved as the velodyne path
        
        # reading the point cloud
        points_v = np.fromfile(
            str(v_path), dtype=np.float32, count=-1).reshape([-1, 4])
        
        rect = calib['R0_rect']
        P2 = calib['P2']
        Trv2c = calib['Tr_velo_to_cam']

        # first remove points where z < 0
        # keep = points_v[:, -1] > 0
        # points_v = points_v[keep]
        # then remove outside

        if back:
            points[:, 0] = -points_v[:, 0]

        # cropping point cloud
        points_v = np_box_ops.remove_outside_points(points_v, rect,Trv2c, P2,
                                                    image_info["image_shape"]) 
        
        if save_path is None:
            save_filename = v_path.parent.parent / (
                v_path.parent.stem + "_reduced") / v_path.name
            if back:
                save_filename += "_back"
        else:
            save_filename = str(Path(save_path) / v_path.name)
            if back:
                save_filename += "_back"

        with open(save_filename, 'w') as f:
            points_v.tofile(f)
        

def create_reduced_point_cloud(data_path, 
                                train_info_path=None,
                                val_info_path=None,
                                test_info_path=None,
                                save_path=None,
                                with_back=False):
    # setting defaults
    if train_info_path is None: 
        train_info_path = Path(data_path) / "kitti_infos_train.pkl"
    if val_info_path is None:
        val_info_path = Path(data_path) / 'kitti_infos_val.pkl'
    if test_info_path is None:
        test_info_path = Path(data_path) / 'kitti_infos_test.pkl'


    _create_reduced_point_cloud(data_path, train_info_path, save_path)
    _create_reduced_point_cloud(data_path, val_info_path, save_path)
    _create_reduced_point_cloud(data_path, test_info_path, save_path)

    if with_back:
        _create_reduced_point_cloud(
            data_path, train_info_path, save_path, back=True)
        _create_reduced_point_cloud(
            data_path, val_info_path, save_path, back=True)
        _create_reduced_point_cloud(
            data_path, test_info_path, save_path, back=True)
