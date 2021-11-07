from itertools import starmap
import torch
from det3d.builders import anchor_generator_builder
from det3d.builders import voxel_generator_builder
import yaml
from omegaconf import DictConfig
import open3d as o3d
import numpy as np

filepath = 'configs/transforms/target_assigner/anchor_generator/cyl_kitti_car_fhd.yaml'
scenepath = 'data/kitti/training/velodyne_reduced/000004.bin'
cyl_voxel_gen_cfg_path = 'configs/transforms/voxel_generator/cyl_kitti_car_git.yaml'
voxel_gen_cfg_path = 'configs/transforms/voxel_generator/kitti_car_ld.yaml'

def cyl2cart(pc_cyl):
    pc_cyl = torch.tensor(pc_cyl)
    x = pc_cyl[:,0] * torch.cos(pc_cyl[:,1])
    x.unsqueeze_(-1)
    y = pc_cyl[:,0] * torch.sin(pc_cyl[:,1])
    y = y.unsqueeze_(-1)
    z = pc_cyl[:, 2]
    z.unsqueeze_(-1)

    return torch.cat([x, y, z], dim=-1)

def cyl2cart_torch(pc_cyl):
    
    x = pc_cyl[..., 0] * torch.cos(pc_cyl[..., 1])
    y = pc_cyl[..., 0] * torch.sin(pc_cyl[..., 1])
    z = pc_cyl[:, 2]
    x.unsqueeze_(-1)
    y.unsqueeze_(-1)
    z.unsqueeze_(-1)

    return torch.cat([x, y, z], dim=-1).reshape_as(pc_cyl)


def cyl2cart_numpy(pc_cyl):
    
    pc_cyl_shape = pc_cyl.shape

    x = pc_cyl[..., 0] * np.cos(pc_cyl[..., 1])
    y = pc_cyl[..., 0] * np.sin(pc_cyl[..., 1])
    z = pc_cyl[:, 2]
    x = x[..., np.newaxis]
    y = y[..., np.newaxis]
    z = z[..., np.newaxis]

    return np.concatenate([x, y, z], axis=-1).reshape(pc_cyl_shape)

def cart2cyl(pc_cyl):
    r = np.sqrt(np.power(pc_cyl[:, 0], 2) + np.power(pc_cyl[:, 1], 2))
    r = r[..., np.newaxis]
    theta = np.arctan2(pc_cyl[:, 1], pc_cyl[:, 0])
    theta = theta[..., np.newaxis]
    z = pc_cyl[:, 2]
    z = z[..., np.newaxis]

    return np.concatenate([r, theta, z], axis=-1)


def read_yaml(filepath):
    with open(filepath, 'r') as stream:
        try: 
            cfg = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
        
        cfg = DictConfig(cfg)
    
    return cfg

# loading the yaml file
with open(filepath, 'r') as stream:
    try:
        cfg = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        print(exc)
        
cfg.update({'class_name':'Car'})
# Turning to dict_config to add .key refference to keys
cfg = DictConfig(cfg)

# Creating the anchor generator
anchor_generator = anchor_generator_builder.build_anchor_generator(cfg)

print(anchor_generator)
# Generate anchors
anchors = anchor_generator.generate(cfg.feature_map_size)

# Keeping only anchor centers
anchors = anchors[..., :3]#.reshape(-1, 3)
# shape = anchors.shape
# print(anchors.shape)
# #anchors = torch.tensor(anchors)
# anchors = anchors.reshape(-1, 3)
# anchors = cyl2cart_numpy(anchors) #cyl2cart(anchors)
# anchors = anchors.reshape(shape)
anchors = anchors.reshape(-1, 3)


#load an actual scene
scene = np.fromfile(str(scenepath), dtype=np.float32, count=-1
            ).reshape([-1, 4])[:, :3]

scene_o3d = o3d.geometry.PointCloud()
scene_o3d.points = o3d.utility.Vector3dVector(scene)

# Drawing anchors using open3d
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(anchors)

#o3d.visualization.draw_geometries([scene_o3d, pcd])

## --------------- Point Cloud Cylidrical Voxelization --------------- ##


from spconv.utils import VoxelGeneratorV2, points_to_voxel

def numpyfy(array):
    if not isinstance(array, np.ndarray):
        array = np.array(array)

    return array

class CylVoxelGenerator():

    def __init__(self, 
                 grid_size, 
                 point_cloud_range, 
                 max_num_points):

        # getting point cloud range to match the grid dimensions
        grid_size = numpyfy(grid_size)
        point_cloud_range = numpyfy(point_cloud_range)

        print(point_cloud_range)
        ranges = numpyfy([point_cloud_range[i+3] - point_cloud_range[i] for i in range(3)])
        print(ranges)
        self.voxel_size = ranges / grid_size


        self.voxel_generator = VoxelGeneratorV2(self.voxel_size, 
                                                point_cloud_range, 
                                                max_num_points)


    def generate(self, points, cylidrical_input=False, max_voxels=None):
        print('Points shape')
        print(points.shape)
        if not cylidrical_input:
            # transforming point cloud to cylidrical
            points = cart2cyl(points)

        return self.voxel_generator.generate(points, max_voxels)

# grid_size = [351, 399, 10] # using voxel size = N-1
# point_cloud_range = [0, -1.0471975511965976, -3.0, 80.0, 1.0471975511965976, 1.0]
# voxel_generator = CylVoxelGenerator(grid_size=grid_size, 
#                                     point_cloud_range=point_cloud_range,
#                                     max_num_points = 35)

voxel_generator_cfg = read_yaml(cyl_voxel_gen_cfg_path)
print(voxel_generator_cfg)

voxel_generator = voxel_generator_builder.build_voxel_generator(voxel_generator_cfg)

res = voxel_generator.generate(
            scene,
            max_voxels=17000)


#print(res['coordinates'])
#print("max")
#print(np.max(res['coordinates']))
coords = res['coordinates'][:, [2, 1, 0]]
points_new = coords * voxel_generator.voxel_size
#print(points_new)
points_new = points_new + voxel_generator.point_cloud_range[:3]
#print(points_new)


points_new = cyl2cart(points_new)
scene_o3d = o3d.geometry.PointCloud()
scene_o3d.points = o3d.utility.Vector3dVector(points_new)
o3d.visualization.draw_geometries([scene_o3d ]) #,  pcd])
# NOTE: voxel coordinates are in a cylidrical grid -> Proof that it works

