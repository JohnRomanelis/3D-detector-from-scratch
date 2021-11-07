from det3d.transforms import PrepareKitti


def build_transforms(cfg, mode):
    
    # Should give the parent cfg of transforms, 
    # containing also information about the used dataset

    transform_list = []

    transform_cfg = cfg.transforms

    if 'prepare_kitti' in transform_cfg.keys():
        transform_list.append(
            PrepareKitti(cfg, mode=mode)
        )
    

    return transform_list

