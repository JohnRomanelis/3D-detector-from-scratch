from ..datasets import KittiDataset

from configs.config_utils import configure_path


def build_dataset(dataset_name=None, data_path=None, info_path=None, mode='train', transforms=[], cfg=None):

    # getting dataset name
    if dataset_name is None:
        assert 'dataset_name' in cfg.keys(), 'No dataset name is provides and "dataset_name" not in cfg file'
        dataset_name = cfg.dataset_name

    # configuring mode
    if mode is None:
        assert 'mode' in cfg.keys(), 'No mode provided'
        mode = cfg.mode

    # getting data path
    if data_path is None:
        assert 'data_path' in cfg.keys(), 'No data path is provided and "data_path" not in cfg file'
        data_path = configure_path(cfg.data_path, dataset_name)

    # getting info path
    if info_path is None:
        assert 'info_path' in cfg.keys(), 'No info path provided and "info_path" not in cfg file'
        info_path = configure_path(cfg.info_path, dataset_name)

    # ----- BUILDING DATASET ----- #
    if dataset_name == "kitti":
        if cfg is None:
            dataset = KittiDataset(data_path,
                                   info_path=info_path, 
                                   mode=mode, 
                                   transforms=transforms)
        else:
            dataset = KittiDataset(data_path, 
                                   info_path, 
                                   mode=mode,
                                   class_names=cfg.class_names,
                                   transforms=transforms, 
                                   num_point_features=cfg.num_point_features)
    else: 
        raise NotImplementedError


    return dataset






