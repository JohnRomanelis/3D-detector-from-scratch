from det3d.detection_core.anchor_generator import AnchorGeneratorRange, CylidricalAnchorGeneratorRange

def build_anchor_generator(cfg):
    # NOTE: give target assigner cfg
    class_name = cfg.class_name
    cfg = cfg.anchor_generator #uncomment this line
    if cfg.anchor_generator_class == 'AnchorGeneratorRange':
        anchor_generator = AnchorGeneratorRange(
                                    cfg.anchor_ranges, 
                                    cfg.sizes, 
                                    cfg.rotations, 
                                    match_threshold=cfg.matched_threshold,
                                    unmatch_threshold=cfg.unmatched_threshold,
                                    class_name=class_name)
    
    elif cfg.anchor_generator_class == 'CylidricalAnchorGeneratorRange':
        anchor_generator = CylidricalAnchorGeneratorRange(
                                    cfg.anchor_ranges, 
                                    cfg.sizes, 
                                    cfg.rotations, 
                                    match_threshold=cfg.matched_threshold,
                                    unmatch_threshold=cfg.unmatched_threshold,
                                    class_name=class_name)
    else:
        raise NotImplementedError


    return anchor_generator