from det3d.detection_core import preprocess as prep 
from det3d.detection_core.sample_ops import DataBaseSampler

import pickle


def build_preprocessor(cfg):


    preprocesses = []

    if 'filter_by_difficulty' in cfg.keys():
        preproc = prep.DBFilterByDifficulty([
            cfg.filter_by_difficulty.removed_difficulties
        ])
        preprocesses.append(preproc)
    if 'filter_by_min_num_points' in cfg.keys():
        preproc = prep.DBFilterByMinNumPoint(
            cfg.filter_by_min_num_points.min_num_point_pairs
        )
        preprocesses.append(preproc)

    database_preprocessor = prep.DataBasePreprocessor(preprocesses)

    return database_preprocessor
    


def build_db_sampler(cfg):

    # Create Database preprocessors
    preproc_cfg = cfg.database_prep_steps
    database_preprocessor = build_preprocessor(preproc_cfg)

    # loading database info path
    info_path = cfg.database_info_path
    with open(info_path, 'rb') as f:
        db_infos = pickle.load(f)

    # loading rest info from the cfg file
    rate = cfg.rate
    grot_range = list(cfg.global_random_rotation_range_per_object)
    if len(grot_range) == 0:
        grot_range=None

    # TODO: NEEDS FIX for multiple classes
    groups = cfg.sample_groups.name_to_max_num 
    print(groups)
    groups = [groups]

    # Creating the db sampler 
    db_sampler = DataBaseSampler(
        db_infos, groups, database_preprocessor, rate, grot_range
    )

    sampler_opts = {}
    sampler_opts['random_crop'] = cfg.random_crop
    sampler_opts['sample_importance'] = cfg.sample_importance
    sampler_opts['remove_points_after_sample'] = cfg.remove_points_after_sample

    return db_sampler, sampler_opts