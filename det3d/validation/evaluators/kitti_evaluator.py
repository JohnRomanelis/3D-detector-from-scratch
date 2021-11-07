from det3d.detection_core import box_np_ops

from ..eval_utils import save_anno_to_labelfile
from tqdm import tqdm
import numpy as np

# for KittiEvaluator.evaluate
# using code from https://github.com/traveller59/kitti-object-eval-python.git
from det3d.datasets.kitti.kitti_object_eval_python.eval import get_official_eval_result
from det3d.datasets.kitti.kitti_object_eval_python import kitti_common as kitti


def _read_imageset_file(path):
    with open(path, 'r') as f:
        lines = f.readlines()
    return [int(line) for line in lines]


def get_start_result_anno():
    annotations = {}
    annotations.update({
        'name': [],
        'truncated': [],
        'occluded': [],
        'alpha': [],
        'bbox': [],
        'dimensions': [],
        'location': [],
        'rotation_y': [],
        'score': [],
    })
    return annotations

def empty_result_anno():
    annotations = {}
    annotations.update({
        'name': np.array([]),
        'truncated': np.array([]),
        'occluded': np.array([]),
        'alpha': np.array([]),
        'bbox': np.zeros([0, 4]),
        'dimensions': np.zeros([0, 3]),
        'location': np.zeros([0, 3]),
        'rotation_y': np.array([]),
        'score': np.array([]),
    })
    return annotations


class KittiEvaluator():

    def __init__(self, dataset_info_files, class_names):

        # reading info files from dataset
        self._kitti_infos = dataset_info_files
        # getting evalating class names
        if not isinstance(class_names, (list, tuple)):
            class_names = [class_names]
        self._class_names = class_names
        self.detections = []

    def track_detections(self):

        self.detections = []

    def add_detections(self, detections):

        if not isinstance(detections, list):
            detections = [detections]
        
        self.detections += detections

    def save_detections_for_evaluation(self, save_path):

        annos = self.convert_detection_to_kitti_annos(self.detections)

        print("Saving KITTI predictions")
        #print(annos[0])
        # saving annos to file
        for anno in tqdm(annos):
            save_anno_to_labelfile(save_path, anno)
    

    def evaluate(self, det_path, gt_path, gt_split_file, return_metrics=False):
        # calculates the AP 
        dt_annos = kitti.get_label_annos(det_path)
        val_image_ids = _read_imageset_file(gt_split_file)
        gt_annos = kitti.get_label_annos(gt_path, val_image_ids)
        if return_metrics: 
            return get_official_eval_result(gt_annos, dt_annos, 0, 
                    return_metrics=return_metrics)
        else: 
            print(get_official_eval_result(gt_annos, dt_annos, 0, 
                    return_metrics=return_metrics)) # 6s in my computer






    def convert_detection_to_kitti_annos(self, detection):

        class_names = self._class_names

        det_image_idxes = [det["metadata"]["image_idx"] for det in detection]
        gt_image_idxes = [info["image"]["image_idx"]
                          for info in self._kitti_infos]

        annos = []

        for i in range(len(detection)):
            # seperating detection results for frame i
            det_idx = det_image_idxes[i]
            det = detection[i]

            info = self._kitti_infos[i]
            calib = info["calib"]
            rect = calib["R0_rect"]
            Trv2c = calib["Tr_velo_to_cam"]
            P2 = calib["P2"]

            final_box_preds = det["box3d_lidar"].detach().cpu().numpy()
            label_preds = det["label_preds"].detach().cpu().numpy()
            scores = det["scores"].detach().cpu().numpy()

            # if there are available predictions
            if final_box_preds.shape[0] != 0:
                final_box_preds[:, 2] -= final_box_preds[:, 5] / 2
                # projecting points to cameraspace
                box3d_camera = box_np_ops.box_lidar_to_camera(
                    final_box_preds, rect, Trv2c)
                # getting box attributes
                locs = box3d_camera[:, :3]
                dims = box3d_camera[:, 3:6]
                angles = box3d_camera[:, 6]
                #
                camera_box_origin = [0.5, 1.0, 0.5]
                # changing box representation -> using box corners to represent it
                box_corners = box_np_ops.center_to_corner_box3d(
                    locs, dims, angles, camera_box_origin, axis=1)
                # projecting the box on the image plane
                box_corners_in_image = box_np_ops.project_to_image(
                    box_corners, P2)

                minxy = np.min(box_corners_in_image, axis=1)
                maxxy = np.max(box_corners_in_image, axis=1)
                # 2d box encoding in kitti format
                bbox = np.concatenate([minxy, maxxy], axis=1)


            # Create a dictionary with empty lists to store the detection results
            anno = get_start_result_anno()
            num_example = 0
            box3d_lidar = final_box_preds

            # for every detection
            for j in range(box3d_lidar.shape[0]):
                image_shape = info["image"]["image_shape"]
                # if box is out of image plane, go to next prediction
                if bbox[j, 0] > image_shape[1] or bbox[j, 1] > image_shape[0]:
                    continue
                if bbox[j, 2] < 0 or bbox[j, 3] < 0:
                    continue
                #else:
                # croping bbox edges that may lay outside the image frame
                bbox[j, 2:] = np.minimum(bbox[j, 2:], image_shape[::-1])
                bbox[j, :2] = np.maximum(bbox[j, :2], [0, 0])
                # adding bounding box to anno
                anno["bbox"].append(bbox[j])

                #alpha
                anno['alpha'].append(
                    -np.arctan2(-box3d_lidar[j, 1], box3d_lidar[j, 0]) +
                    box3d_camera[j, 6])
                anno["dimensions"].append(box3d_camera[j, 3:6])
                anno["location"].append(box3d_camera[j, :3])
                anno["rotation_y"].append(box3d_camera[j, 6])
                anno["name"].append(class_names[int(label_preds[j])])
                anno["truncated"].append(0.0)
                anno["occluded"].append(0)
                anno["score"].append(scores[j])

                num_example += 1

            if num_example != 0:
                anno = {n : np.stack(v) for n, v in anno.items()}
                annos.append(anno)
            else:
                annos.append(empty_result_anno())
            # adding metadata info 
            annos[-1]["metadata"] = det["metadata"]
        
        return annos



