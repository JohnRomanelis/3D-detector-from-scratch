import hydra 
from omegaconf import DictConfig

# Import building functions
from det3d.builders import build_dataset, build_transforms

# Dataloader & collate_fn
from torch.utils.data import DataLoader
from tools.collate_fns import sparse_collate_fn

# Trainer
from det3d.training_utils.trainers import SingleModelTrainer

# Network modules
from det3d.models.networks.second import VanillaSecond
from det3d.models.networks.cyl_second import CylidricalSecond
from det3d.models.networks.second_github import SECONDGithub

# Remove numba warnings
import warnings
warnings.filterwarnings('ignore')  # to remove warnings from numba

from tools.batch_transforms import example_convert_to_torch

from tools.visualizer import KittiVisualizer

import numpy as np

@hydra.main(config_path='configs', config_name="car_fhd_git")
def main(cfg: DictConfig) -> None:

    # Building Dataset and Transforms
    transform = build_transforms(cfg, mode='inference')
    train_dataset = build_dataset(cfg = cfg.dataset, transforms=transform)

    # Creating the training dataloader
    batch_size = 1
    num_workers = 1

    train_loader = DataLoader(train_dataset, 
                              batch_size=batch_size,
                              shuffle=False,
                              collate_fn=sparse_collate_fn,
                              num_workers=num_workers,
                              drop_last=False)

    # Creating the validation dataloader
    transform = build_transforms(cfg, mode='evaluation')
    valid_dataset = build_dataset(cfg=cfg.dataset, transforms=transform, mode='val')

    batch_size = 1
    num_workers = 1

    # Creating the network
    #net = VanillaSecond(cfg)
    #net = CylidricalSecond(cfg)
    net = SECONDGithub(cfg)

    valid_loader = DataLoader(valid_dataset,  
                              batch_size=batch_size,
                              shuffle=False, 
                              collate_fn=sparse_collate_fn, 
                              num_workers=num_workers,
                              drop_last=False)



    trainer = SingleModelTrainer(net, cfg, train_loader, valid_loader)
    

    # getting the trained model from the trainer
    net = trainer.model
    net.eval()

    batch = next(iter(train_loader))
    example = example_convert_to_torch(batch, device='cuda:0')

    model_out = net(example)

    preds = net.predict(example, model_out)



    visualizer = KittiVisualizer(bounding_boxes=preds[0]['box3d_lidar'].detach().cpu().numpy())
    
    points = batch['points'] #.squeeze(0)
    points = points[:, :3]
    visualizer.add_lidar_pointcloud(points, pc_color=None)
    visualizer.add_multiple_boxes(batch['gt_bboxes'], np.array([1, 0, 0]))
    visualizer.draw()

































    '''
    pathfile = '/home/ioannis/Desktop/programming/thesis/detection3d_v1-master/data/kitti/training/velodyne/000000.bin'
    points = np.fromfile(
            str(pathfile), dtype=np.float32, count=-1
            ).reshape([-1, 4])[:, :3]

    v2 = KittiVisualizer()
    v2.add_lidar_pointcloud(points)
    v2.draw()
    '''

if __name__=="__main__":
    main()

