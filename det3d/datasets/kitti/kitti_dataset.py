from ..base_dataset import DatasetTemplate
from .kitti_utils import remove_dontcare

from det3d.detection_core import box_np_ops as np_box_ops

import numpy as np
from pathlib import Path


class KittiDataset(DatasetTemplate):

    ## TODO: Read parameters from configs
    num_point_features = 4

    def __init__(self, 
                 root_path, 
                 info_path,
                 mode = 'train', 
                 class_names=None,
                 transforms=[],
                 num_point_features=4):
        
        # configuring info path based on the mode of the dataset
        # Options for mode: - train
        #                   - val
        #                   - trainval
        #                   - test
        self.mode = mode
        info_path = self.configure_info_path(info_path)
        print(info_path)
        super().__init__(root_path, info_path, 
                        class_names=class_names, 
                        transforms=transforms,
                        num_point_features=num_point_features)


    def get_sensor_data(self, query):  
        # reading raw sensor data
        read_image = False
        idx=query
        if isinstance(query, dict):
            read_image = "cam" in query
            assert "lidar" in query
            idx = query['lidar']['idx']


        info = self._infos[idx]
        res = {
            'lidar':{
                'type': 'lidar',
                'points': None
            },
            'metadata':{
                'image_idx': info['image']['image_idx'],
                'image_shape': info['image']['image_shape']
            },
            'calib': None,
            'cam':{}
        }

        pc_info = info['point_cloud']
        velo_path = Path(pc_info['velodyne_path'])

        if not velo_path.is_absolute():
            velo_path = Path(self._root_path) / pc_info['velodyne_path']
        velo_reduced_path = velo_path.parent.parent / (
            velo_path.parent.stem + '_reduced') / velo_path.name
        if velo_reduced_path.exists():
            velo_path = velo_reduced_path

        # reading point cloud data
        points = np.fromfile(
            str(velo_path), dtype=np.float32, count=-1
            ).reshape([-1, self.num_point_features])

        res["lidar"]["points"]=points
        
        if read_image:
            image_info=info["image"]
            image_path=image_info["image_path"]
            #image_path = self._root_path / image_path # full path is stored
            with open(str(image_path), 'rb') as f:
                image_str = f.read()
            res["cam"] = {
                "type" : "camera",
                "data" : image_str,
                "datatype" : image_path.suffix[1:]
            } 

        calib = info['calib']
        calib_dict = {
            'rect' : calib['R0_rect'],
            'Trv2c': calib['Tr_velo_to_cam'],
            'P2': calib['P2'],
        }
        res['calib'] = calib_dict

        if 'annos' in info:
            annos = info["annos"]
            # we need other object to avoid collision when sample
            #print(annos)
            annos = remove_dontcare(annos)
            #print(annos)
            locs = annos["location"]
            dims = annos["dimensions"]
            rots = annos["rotation_y"]
            gt_names = annos["name"]
            gt_boxes = np.concatenate([locs, dims, rots[..., np.newaxis]],
                                      axis=1).astype(np.float32)

            calib = info["calib"]
            gt_boxes = np_box_ops.box_camera_to_lidar(
                gt_boxes, calib["R0_rect"], calib["Tr_velo_to_cam"])

            # only center format is allowed. so we need to convert
            # kitti [0.5, 0.5, 0] center to [0.5, 0.5, 0.5]
            np_box_ops.change_box3d_center_(gt_boxes, [0.5, 0.5, 0],
                                            [0.5, 0.5, 0.5])

            res["lidar"]["annotations"] = {
                'boxes': gt_boxes,
                'names': gt_names,
            }
            res["cam"]["annotations"] = {
                'boxes': annos["bbox"],
                'names': gt_names,
            }

        return res


    def add_metadata_to_example(self, example, input_dict):
        # copies the specified metadata from the input_dict to the example
        # ByDefault: copies the metada if image_idx is incuded
        # Overwrite this function to change condition
        if 'image_idx' in input_dict['metadata']:
            example['metadata'] = input_dict['metadata']



    def configure_info_path(self, info_path):
        msg = '''Unknown Mode. Please provide one of the following:
            - train
            - trainval
            - val
            - test
        '''
        assert self.mode in ['train', 'trainval', 'val', 'test'], msg

        info_path = Path(info_path)
        file_name =  'kitti_infos_' + self.mode + '.pkl'
        return info_path / file_name

        
            