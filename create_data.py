# configuration imports
import hydra
from omegaconf import DictConfig
from configs.config_utils import configure_path


def prepare_kitti(dataset_name, data_path, info_path, cfg):
    from det3d.datasets.kitti import kitti_preprocess
    from det3d.datasets.gt_database import create_groundtruth_database

    # create info files
    kitti_preprocess.create_kitti_info_file(data_path)

    # create reduced version of the pointcloud
    kitti_preprocess.create_reduced_point_cloud(data_path)

    # create ground truth dataset
    create_groundtruth_database("kitti", data_path, info_path)


@hydra.main(config_path='configs/dataset', config_name='kitti_all')
def main(cfg: DictConfig) -> None:

    # loading the name of the dataset to process
    dataset_name = cfg.dataset_name
    assert dataset_name is not None, 'No dataset provided'

    # configuring the path to the dataset
    data_path = configure_path(cfg.data_path, dataset_name)
    info_path = configure_path(cfg.info_path, dataset_name)

    if dataset_name == 'kitti':
        prepare_kitti(dataset_name, data_path, info_path, cfg)



if __name__ == "__main__":
    main()