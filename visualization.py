from tools import visualizer
from tools.visualizer import KittiVisualizer
import numpy as np
import pickle

from PIL import Image

from det3d.detection_core import box_np_ops

import matplotlib.pyplot as plt
import matplotlib.lines as lines

from det3d.datasets.kitti.kitti_preprocess import get_label_anno

def int2constString(num, string_len=6):
    '''
        Input : num (intiger)

        Output : a string with a constant length of string_len
        
            e.g  num=1, string_len=6   =>   "000001"
    '''
    
    return str(num).zfill(string_len) 



calib_data = '/home/ioannis/Desktop/programming/thesis/detection3d_new/data/kitti/training/calib/'
lidar_path = '/home/ioannis/Desktop/programming/thesis/detection3d_new/data/kitti/training/velodyne_reduced/'
label_path = '/home/ioannis/Desktop/programming/thesis/detection3d_new/data/kitti/training/label_2/'
detection_path = '/home/ioannis/Desktop/programming/thesis/detection3d_new/data/kitti/detection_results/data/'
info_path = '/home/ioannis/Desktop/programming/thesis/detection3d_new/data/kitti/kitti_infos_val.pkl'

# reading val dataset indexes
val_split_idxs = '/home/ioannis/Desktop/programming/thesis/detection3d_new/data/kitti/kittiSplits/val.txt'
with open(val_split_idxs, 'r') as f:
    idx = f.readlines()
idxs = [i[:-1] for i in idx[:-1]] + [idx[-1]] # removing '\n' character

with open(info_path, 'rb') as f:
    infos = pickle.load(f)


def get_velo_path(idx):
    assert len(idx) == 6
    return lidar_path + idx + '.bin'

def get_label_path(idx):
    assert len(idx) == 6
    return label_path + idx + '.txt'

def get_detection_path(idx):
    assert len(idx) == 6
    return detection_path + idx + '.txt'

def get_gt_boxes(idx):
    filename = get_label_path(idx)
    boxes = get_label_anno(filename)
    return boxes

def get_det_boxes(idx):
    filename = get_detection_path(idx)
    boxes = get_label_anno(filename)
    return boxes

def read_pointcloud(velo_path):
    points = np.fromfile(
            str(velo_path), dtype=np.float32, count=-1
            ).reshape([-1, 4])[..., :3]
    return points

def filter_boxes(boxes, cat='Car'):
    mask = boxes['name'] == cat
    for k, v in boxes.items():
        boxes[k] = v[mask] 
    
    num_boxes = len(boxes['name'])
    code_lenght = 7

    res = np.zeros((num_boxes, code_lenght))

    for i in range(num_boxes):
        res[i, :3] = boxes['location'][i]
        res[i, 3:-1] = boxes['dimensions'][i]
        res[i, -1] = boxes['rotation_y'][i]

    return res

def crop_dim(pc, lims, axis):
    pc = pc[pc[..., axis] > lims[0]]
    pc = pc[pc[..., axis] < lims[1]]
    return pc

def crop_pc(pc, ranges=[(0, 70.4), (-40, 40), (-3, 1)]):

    for axis, lims in enumerate(ranges):
        pc = crop_dim(pc, lims, axis)
    
    return pc
    
def drawBB3D(ax, corners3d, P, color ):
    
    # the face indexes are the same and are computed based on the way we create 
    # the bounding boxes in function - boundingBoxToGlobal -
    face_idx = np.array([[0,1,5,4],   # front face
                         [1,2,6,5],   # left face
                         [2,3,7,6],   # back face
                         [3,0,4,7]])  # right face
    
    # Projecting 3d bounding box to plane
    corners2d = box_np_ops.project_to_image(corners3d, P)
    
    # Drawing the lines connecting the edge poitns
    # and adding them to ax
    
    # for each face of the cube
    for box in corners2d:
        for i in range(4):
            # for each corner
            for j in range(4):
                l = lines.Line2D([box[face_idx[i][j], 0],
                                        box[face_idx[i][(j+1)%4], 0]
                                    ],  # x coordinates
                                [box[face_idx[i][j], 1],
                                        box[face_idx[i][(j+1)%4], 1]
                                    ],  # y coordinates
                                color = color)  
                # adding line to axes
                ax.add_line(l)

def rbbox3d_to_corners(rbboxes, origin=[0.5, 0.5, 0.5], axis=2):
    return box_np_ops.center_to_corner_box3d(
        rbboxes[..., :3],
        rbboxes[..., 3:6],
        rbboxes[..., 6],
        origin,
        axis=axis)



start = 4
for info in infos[start:]:

    idx = int2constString(info['image']['image_idx'])
    print("scene : ", idx)

    frame = get_velo_path(idx)
    points = read_pointcloud(frame)
    #points = crop_pc(points)


    calib = info['calib']
    calib_dict = {
        'rect' : calib['R0_rect'],
        'Trv2c': calib['Tr_velo_to_cam'],
        'P2': calib['P2'],
    }


    annos = info['annos']
    dimensions = annos['dimensions']
    locations = annos['location']
    rotations = annos['rotation_y'][..., np.newaxis]
    gt_boxes = np.concatenate([locations, dimensions, rotations],
                             axis=-1).astype(np.float32)
    gt_boxes = gt_boxes[annos['name'] == 'Car']

    gt_boxes_lidar = gt_boxes.copy()
    gt_boxes_lidar = box_np_ops.box_camera_to_lidar(
               gt_boxes_lidar, calib["R0_rect"], calib["Tr_velo_to_cam"])
    box_np_ops.change_box3d_center_(gt_boxes_lidar, [0.5, 0.5, 0],
                                            [0.5, 0.5, 0.5])
    

    # getting detections
    det_boxes = get_det_boxes(idx)
    det_boxes = filter_boxes(det_boxes)
    det_boxes = box_np_ops.box_camera_to_lidar(
                det_boxes, calib["R0_rect"], calib["Tr_velo_to_cam"])
    box_np_ops.change_box3d_center_(det_boxes, [0.5, 0.5, 0],
                                            [0.5, 0.5, 0.5])
    


    # Visualizer for 3D scene
    visualizer = KittiVisualizer()
    visualizer.add_lidar_pointcloud(points)#, np.array([0, 0, 0]))
    #visualizer.add_multiple_boxes(gt_boxes_lidar, np.array([0, 1, 0]))
    visualizer.add_multiple_boxes(det_boxes, np.array([0, 0, 0]))
    visualizer.draw()
    
## =============================================================== ##
## =============== Visualization on image frame ================== ##
## =============================================================== ##

    image = Image.open(info['image']['image_path']).convert ('RGB')
    image = np.array(image, dtype=np.uint8)
    projection_matrix = calib['P0']


    #fig,ax = plt.subplots(1, figsize=(18, 16))
    #ax.imshow(image)

    
    # reading bounding box parameters
    # (keeping original dimensions)
    dimensions = annos['dimensions']#[..., [1, 2, 0]]
    locations = annos['location']
    rotations = annos['rotation_y'][..., np.newaxis]

    gt_boxes = np.concatenate([locations, dimensions, rotations],
                             axis=-1).astype(np.float32)
    gt_boxes = gt_boxes[annos['name'] == 'Car']

    origin = [0.5, -0.5, 0.5]
    box_np_ops.change_box3d_center_(gt_boxes, [0.5,  0. , 0.5],
                                              origin)
     
    gt_camera = rbbox3d_to_corners(gt_boxes, axis = 1)
    #drawBB3D(ax, gt_camera, projection_matrix, np.array([1, 0, 0]))
    

    det_boxes = get_det_boxes(idx)
    det_boxes = filter_boxes(det_boxes)
    #det_boxes_camera = det_boxes[..., [0, 1, 2, 4, 5, 3, 6]]
    det_boxes_camera = det_boxes
    box_np_ops.change_box3d_center_(det_boxes_camera, [0.5,  0. , 0.5],
                                                        origin)
    det_camera = rbbox3d_to_corners(det_boxes_camera, axis = 1)
    #drawBB3D(ax, det_camera, projection_matrix, np.array([1, 0, 0]))
    #plt.show()     

    

'''
for idx in idxs[2:]:
    # reading point cloud
    frame = get_velo_path(idx)
    points = read_pointcloud(frame)

    # getting gt boxes
    gt_boxes = get_gt_boxes(idx)
    gt_boxes = filter_boxes(gt_boxes)

    
    break
'''


'''
{
	"class_name" : "ViewTrajectory",
	"interval" : 29,
	"is_loop" : false,
	"trajectory" : 
	[
		{
			"boundingbox_max" : [ 36.906352353053933, -2.3435024608366186, 0.023128226259399476 ],
			"boundingbox_min" : [ 7.5863610232275711, -4.0694122952445761, -2.0163112631695905 ],
			"field_of_view" : 60.0,
			"front" : [ 0.0, 0.0, 1.0 ],
			"lookat" : [ 22.24635668814075, -3.2064573780405974, -0.99659151845509553 ],
			"up" : [ 0.0, 1.0, 0.0 ],
			"zoom" : 0.69999999999999996
		}
	],
	"version_major" : 1,
	"version_minor" : 0
}
'''
