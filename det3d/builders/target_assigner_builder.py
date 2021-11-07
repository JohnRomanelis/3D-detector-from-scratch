from det3d.detection_core.region_similarity import NearestIouSimilarity
from det3d.detection_core.target_assigner import TargetAssigner

from .box_encoder_builder import build_box_coder
from .anchor_generator_builder import build_anchor_generator

def get_region_similarity_function(sim_func_name):

    if sim_func_name == 'nearest_iou_similarity':
        return NearestIouSimilarity()

    

def build_target_assigner(cfg):

    # NOTE: Add support for multiple anchor generators

    # Getting boc_coder and anchor_generator cfgs
    box_coder_cfg = cfg.box_coder
    anchor_generator_cfg = cfg.anchor_generator
    
    # Building box_coder and anchor_generator
    box_coder = build_box_coder(box_coder_cfg)
    anchor_generator = build_anchor_generator(cfg) # needs target_assigner cfg


    # Getting the similarity function that we will use
    sim_func_name = cfg.region_similarity_calculator
    similarity_calc = get_region_similarity_function(sim_func_name)

    target_assigner = TargetAssigner(box_coder=box_coder, 
                                     anchor_generators=anchor_generator,
                                     classes=cfg.class_name,
                                     feature_map_sizes=cfg.anchor_generator.feature_map_size,
                                     region_similarity_calculators=similarity_calc)

    return target_assigner


    