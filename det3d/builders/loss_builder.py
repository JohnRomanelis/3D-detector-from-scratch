from det3d.losses.loss_functions import SigmoidFocalClassificationLoss, \
                                        WeightedSmoothL1LocalizationLoss, \
                                        WeightedSoftmaxClassificationLoss


def build_loss(cfg):

    if cfg.loss=='weighted_sigmoid_focal':
        alpha = cfg.alpha
        gamma = cfg.gamma

        loss = SigmoidFocalClassificationLoss(alpha=alpha, gamma=gamma)
        print("- Classification Loss Initialized")
        print("  Loss: Focal Loss")
        print(f"    with Parameters: alpha:{alpha}  gamma:{gamma} ")

    elif cfg.loss=='weighted_smooth_l1':
        sigma = cfg.sigma
        code_weights = None #cfg.code_weights #TODO
        codewise = cfg.codewise
        loss = WeightedSmoothL1LocalizationLoss(sigma=sigma,
                                                code_weights=code_weights,
                                                codewise=codewise)
        print("- Localization Loss Initialized")
        print("  Loss: Smooth L1")
        print(f"     with Parameters: sigma:{sigma}")

    elif cfg.loss=="weighted_softmax":
        loss = WeightedSoftmaxClassificationLoss()
        print("- Direction Classification Loss Initialized")
        print("  Loss: Softmax")
    
    # NOTE: add new losses here! 
    # elif cfg.loss == "new loss name":
    #    get loss parameters
    #    loss = LossClass(parameters)
    #       NOTE: Dont forget to import new Loss class
    #    print initialization message
    # TODO: add the initialization message inside the Loss class

    else:
        raise NotImplementedError


    return loss
