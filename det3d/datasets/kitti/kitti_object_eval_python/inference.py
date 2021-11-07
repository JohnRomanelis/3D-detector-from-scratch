import kitti_common as kitti
from eval import get_official_eval_result, get_coco_eval_result
def _read_imageset_file(path):
    with open(path, 'r') as f:
        lines = f.readlines()
    return [int(line) for line in lines]
det_path = "/home/tourloid120/Desktop/programming/thesis/detection3d_v1-master/data/kitti/detection_results/data"
dt_annos = kitti.get_label_annos(det_path)
gt_path = "/home/tourloid120/Desktop/programming/thesis/detection3d_v1-master/data/kitti/training/label_2"
gt_split_file = "/home/tourloid120/Desktop/programming/thesis/detection3d_v1-master/v3d/datasets/kitti/kittiSplits/val.txt" # from https://xiaozhichen.github.io/files/mv3d/imagesets.tar.gz
val_image_ids = _read_imageset_file(gt_split_file)
gt_annos = kitti.get_label_annos(gt_path, val_image_ids)
print(get_official_eval_result(gt_annos, dt_annos, 0)) # 6s in my computer
#print(get_coco_eval_result(gt_annos, dt_annos, 0)) # 18s in my computer