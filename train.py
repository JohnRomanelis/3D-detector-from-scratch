# Configuration files
import hydra
from omegaconf import DictConfig
from configs.config_utils import configure_path
from pathlib import Path

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
from det3d.models.networks.second_github import SECONDGithub, SECONDGithub2, SECONDGithubOccupancy

# Optimizer and Scheduler
import torch

# Remove numba warnings
import warnings
warnings.filterwarnings('ignore')  # to remove warnings from numba


@hydra.main(config_path='configs', config_name="car_fhd_git") #_git
def main(cfg: DictConfig) -> None:

    # Building Dataset and Transforms
    transform = build_transforms(cfg, mode='training')
    train_dataset = build_dataset(cfg = cfg.dataset, transforms=transform)

    # Creating the training dataloader
    batch_size = 8
    num_workers = 8

    train_loader = DataLoader(train_dataset, 
                              batch_size=batch_size,
                              shuffle=True,
                              collate_fn=sparse_collate_fn,
                              num_workers=num_workers,
                              drop_last=True)

    # Creating the validation dataloader
    transform = build_transforms(cfg, mode='evaluation')
    valid_dataset = build_dataset(cfg=cfg.dataset, transforms=transform, mode='val')

    batch_size = 16
    num_workers = 8

    # Creating the network
    #net = VanillaSecond(cfg)
    #net = CylidricalSecond(cfg)
    #net = SECONDGithub(cfg)
    #net = SECONDGithub2(cfg)
    net = SECONDGithubOccupancy(cfg)

    # Creating optimizer and scheduler
    # Creating optimizer
    #optimizer = torch.optim.Adam(
    #    net.parameters(), lr=0.0002, weight_decay=0.0001)
    # Creating scheduler
    #scheduler = torch.optim.lr_scheduler.StepLR(
    #    optimizer, step_size=15, gamma=0.8)

    valid_loader = DataLoader(valid_dataset,  
                              batch_size=batch_size,
                              shuffle=False, 
                              collate_fn=sparse_collate_fn, 
                              num_workers=num_workers,
                              drop_last=False)

    trainer = SingleModelTrainer(net, cfg, train_loader, valid_loader)
    
    trainer.train()

if __name__ == "__main__":
    main()
