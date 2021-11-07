from det3d.detection_core.box_coder import GroundBox3dCoder



def build_box_coder(cfg):

    if cfg.box_coder_class == 'GroundBox3dCoder':
        box_coder = GroundBox3dCoder(cfg.linear_dim,
                                     cfg.encode_angle_to_vector)

    else:
        raise NotImplementedError


    return box_coder


