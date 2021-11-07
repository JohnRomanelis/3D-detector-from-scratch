import omegaconf
import torch.nn as nn
from torch import optim

from functools import partial

from torch.optim import lr_scheduler

from tools.optimization import OptimWrapper
from tools.optimization.learning_schedules_fastai import OneCycle, NoSched

def build_optimizer(model, optim_cfg):

    if optim_cfg.OPTIMIZER == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=optim_cfg.LR, weight_decay=optim_cfg.WEIGHT_DECAY)
    elif optim_cfg.OPTIMIZER == 'adam_onecycle':

        def children(m: nn.Module):
            return list(m.children())
        
        def num_children(m: nn.Module) -> int:
            return len(children(m))

        
        flatten_model = lambda m: sum(map(flatten_model, m.children()), []) if num_children(m) else [m]
        get_layer_groups = lambda m: [nn.Sequential(*flatten_model(m))]
        optimizer_func = partial(optim.Adam, betas=(0.9, 0.99))
        optimizer = OptimWrapper.create(
            optimizer_func, 3e-3, get_layer_groups(model), wd=optim_cfg.WEIGHT_DECAY, true_wd=True, bn_wd=True
        )

    else: 
        raise NotImplementedError

    return optimizer

def build_scheduler(optimizer, total_iters_each_epoch, total_epochs, last_epoch, optim_cfg):
    # decay_steps = [x * total_iters_each_epoch for x in list(optim_cfg.DECAY_STEP_LIST)]
    # def lr_lbmd(cur_epoch):
    #     cur_decay = 1
    #     for decay_step in decay_steps:
    #         if cur_epoch >= decay_step:
    #             cur_decay = cur_decay * optim_cfg.LR_DECAY
    #     return max(cur_decay, optim_cfg.LR_CLIP / optim_cfg.LR)


    lr_warmup_scheduler = None
    total_steps = total_iters_each_epoch * total_epochs
    

    step_during_epoch = False
    
    if optim_cfg.SCHEDULER == 'adam_onecycle': # this scheduler is paired with the optimizer
        lr_scheduler = OneCycle(
            optimizer, total_steps, optim_cfg.LR, list(optim_cfg.MOMS), optim_cfg.DIV_FACTOR, optim_cfg.PCT_START
        )
        step_during_epoch = True
    elif optim_cfg.SCHEDULER == 'StepLR':
        lr_scheduler = optim.lr_scheduler.StepLR(
            optimizer, step_size=optim_cfg.STEP_SIZE, gamma=optim_cfg.GAMMA
        )
    else:
        lr_scheduler = NoSched() 


    return lr_scheduler, step_during_epoch #, lr_warmup_scheduler

