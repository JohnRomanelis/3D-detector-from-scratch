from pathlib import Path
from tqdm import tqdm
import pickle
import numpy as np

from det3d.detection_core import box_np_ops
from det3d.builders import dataset_builder

def create_groundtruth_database(dataset_name,
                                data_path, 
                                info_path, 
                                used_classes=None,
                                database_save_path=None, 
                                db_info_save_path=None,
                                relative_path=True):

    # getting dataset
    dataset = dataset_builder.build_dataset(dataset_name, data_path, info_path)

    # configuring path to store the gt_database
    root_path = Path(data_path)
    if database_save_path is None:
        database_save_path = root_path / 'gt_database'
    else:
        database_save_path = Path(database_save_path)

    if db_info_save_path is None:
        db_info_save_path = root_path / "kitti_dbinfos_train.pkl"

    # Creating a directory to save the database
    # if it doesn't already exists
    database_save_path.mkdir(parents=True, exist_ok=True)

    all_db_infos = {}

    group_counter = 0

    # for every frame of the dataset
    for j in tqdm(range(len(dataset))):
        # getting the frame idx
        image_idx = j
        # reading the sensor data for that frame
        sensor_data = dataset.get_sensor_data(j)

        # getting the actual frame name if provided
        if 'img_idx' in sensor_data['metadata']:
            image_idx = sensor_data["metadata"]["image_idx"]

        # reading frame data
        points = sensor_data['lidar']['points']
        annos = sensor_data['lidar']['annotations']
        gt_boxes = annos['boxes']
        names = annos['names']

        # number of ground truth objects in the scene
        num_obj = gt_boxes.shape[0]

        group_dict = {}

        # creating a np.array, every cell of which
        # corresponds to a ground truth bounding box
        group_ids = np.full([num_obj], -1, dtype=np.int64)
        if "group_ids" in annos:
            group_ids = annos["group_ids"]
        else:
            group_ids = np.arange(num_obj, dtype=np.int64)

        # getting the difficulty for every ground truth bounding box
        # if provided, else difficulty is 0
        difficulty = np.zeros(num_obj, dtype=np.int32)
        if "difficulty" in annos:
            difficulty = annos["difficulty"]
        
        # getting the point indices of the points residing inside
        # the ground truth bounding boxes
        point_indices = box_np_ops.points_in_rbbox(points, gt_boxes)

        # for every ground truth box
        for i in range(num_obj):
            # get the filename to store the gt_database binary
            filename = f"{image_idx}_{names[i]}_{i}.bin"
            filepath = database_save_path / filename

            # getting the points inside the gt_box
            gt_points = points[point_indices[:, i]]

            # transforming points in local bbox coordinates
            gt_points[:,:3] -= gt_boxes[i, :3]

            # writting the points in the binary file
            with open(filepath, 'w') as f:
                gt_points.tofile(f)

            if (used_classes is None) or names[i] in used_classes:

                if relative_path:
                    db_path = str(database_save_path.stem + "/" + filename)
                else:
                    db_path = str(filepath)
                
                db_info = {
                    "name": names[i],
                    "path": db_path,
                    "image_idx": image_idx,
                    "gt_idx": i, 
                    "box3d_lidar": gt_boxes[i],
                    "num_points_in_gt": gt_points.shape[0],
                    "difficulty": difficulty[i]
                }

                local_group_id = group_ids[i]
                if local_group_id not in group_dict:
                    group_dict[local_group_id] = group_counter
                    group_counter += 1
                db_info['group_id'] = group_dict[local_group_id]
                if "score" in annos:
                    db_info["score"] = annos["score"][i]
                
                # adding info file to info_files list
                if names[i] in all_db_infos:
                    all_db_infos[names[i]].append(db_info)
                else:
                    all_db_infos[names[i]] = [db_info]

    for k, v in all_db_infos.items():
        print(f"load {len(v)} {k} database infos")

    with open(db_info_save_path, 'wb') as f:
        pickle.dump(all_db_infos, f)   