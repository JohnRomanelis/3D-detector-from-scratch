from spconv.utils import VoxelGeneratorV2
from det3d.detection_core.voxel_generator import CylidricalVoxelGenerator


def build_voxel_generator(cfg):

    if cfg.voxel_generator_class == 'VoxelGeneratorV2':
        return VoxelGeneratorV2(
                    voxel_size=cfg.voxel_size, 
                    point_cloud_range=cfg.point_cloud_range, 
                    max_num_points=cfg.max_number_of_points_per_voxel)
    
    elif cfg.voxel_generator_class == 'CylidricalVoxelGenerator':
        return CylidricalVoxelGenerator(
                grid_size = cfg.grid_size, 
                point_cloud_range = cfg.point_cloud_range, 
                max_num_points = cfg.max_number_of_points_per_voxel)
    
    else:
        raise NotImplementedError