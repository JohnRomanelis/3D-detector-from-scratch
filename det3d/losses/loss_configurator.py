import torch
from det3d.builders import build_loss

from .loss_utils import create_loss, prepare_loss_weights, LossNormType, \
                        get_pos_neg_loss, get_direction_target

class LossConfigurator:
    """ 
    Desc:
        A LossFunction is a structure that will calculate multiple 
        losses for a network and accumulate the final loss. 
        Will also provide information in dict format for all
        sublosses that can be used for tracking the detection results.
    """

    def __init__(self, loss_cfg, 
                       return_results_for_tracker=True, 
                       dtype=torch.float32):
        
        self.return_results_for_tracker = return_results_for_tracker
        self.dtype = dtype

        # Configuring Normalization type
        if loss_cfg.loss_norm_type == "NormByNumPositives":
            self.loss_norm_type= LossNormType.NormByNumPositives
        else:
            raise NotImplementedError

        #self.use_sigmoid_score = loss_cfg.use_sigmoid_score # not used yet
        self.encode_background_as_zeros = loss_cfg.encode_background_as_zeros
        self.encode_rad_error_by_sin = loss_cfg.encode_rad_error_by_sin
        #self.sin_error_factor = loss_cfg.sin_error_factor # not used yet

        # Relative to Directional Classifier
        self.use_direction_classifier = loss_cfg.use_direction_classifier
        self.num_direction_bins = loss_cfg.num_direction_bins
        self.direction_limit_offset = loss_cfg.direction_limit_offset 

        # weights: 
        self.cls_loss_weight = loss_cfg.classification_weight
        self.loc_loss_weight = loss_cfg.localization_weight
        self.direction_loss_weight = loss_cfg.direction_loss_weight

        # pos/neg cls weights
        self.pos_cls_weight = loss_cfg.pos_cls_weight
        self.neg_cls_weight = loss_cfg.neg_cls_weight

        # list of loss names to return to the tracker 
        self._keys = ['loss']

        
        # ----- Constructing Actual Losses ----- #
        
        if 'cls_loss' in loss_cfg.keys():
            # taking the sub_loss dict
            cls_loss_cfg = loss_cfg.cls_loss
            # building loss
            self.cls_loss_ftor = build_loss(cls_loss_cfg)

            # getting num classes
            self.num_classes = cls_loss_cfg.num_classes
            # adding classification loss keys to self._keys
            self._keys.extend(['cls_loss', 'cls_pos_loss', 'cls_neg_loss'])

        if 'loc_loss' in loss_cfg.keys():
            # taking the sub_loss dict
            loc_loss_cfg = loss_cfg.loc_loss
            # building loss
            self.loc_loss_ftor = build_loss(loc_loss_cfg)

            # adding localization loss keys to self._keys
            self._keys.append('loc_loss')

        if "dir_cls_loss" in loss_cfg.keys() and \
                loss_cfg.use_direction_classifier==True:

            # taking the sub_loss dict
            dir_cls_loss_cfg = loss_cfg.dir_cls_loss
            # building loss
            self.dir_loss_ftor = build_loss(dir_cls_loss_cfg)

            # adding localization loss keys to self._keys
            self._keys.append('dir_loss')

        
            
    @property
    def keys(self):
        return self._keys
    
    def compute_losses(self, network_out, example):

        # computing weights 
        cls_weights, reg_weights, cared = prepare_loss_weights(
            example['labels'],
            pos_cls_weight=self.pos_cls_weight,
            neg_cls_weight=self.neg_cls_weight,
            loss_norm_type=self.loss_norm_type,
            dtype=self.dtype
        )

        # Getting ground truths and predictions
        box_preds = network_out['box_preds']
        cls_preds = network_out['cls_preds']
        #print("cls_preds", cls_preds.shape)
        batch_size_dev = cls_preds.shape[0]
        labels = example['labels']
        reg_targets = example['reg_targets']
        importance = example['importance']
        cls_targets = labels * cared.type_as(labels)
        cls_targets = cls_targets.unsqueeze(-1)

        # Calculating per (batch or anchor) loss
        loc_loss, cls_loss = create_loss(
                    self.loc_loss_ftor,
                    self.cls_loss_ftor,
                    box_preds=box_preds,
                    cls_preds=cls_preds,
                    cls_targets=cls_targets,
                    cls_weights=cls_weights * importance,
                    reg_targets=reg_targets,
                    reg_weights=reg_weights * importance,
                    num_class=self.num_classes,
                    encode_background_as_zeros=self.encode_background_as_zeros,
                    encode_rad_error_by_sin=self.encode_rad_error_by_sin   
                    ### MISSING ARGUMENTS!!! ###                
        )

        #print("cls_loss.shape ", cls_loss.shape)
        #print("max val ", torch.max(cls_loss, 1))
        #print(cls_weights)
        
        #print("batch_size_dev ", cls_loss.sum())
        
        # Accumulate localization loss
        loc_loss_reduced = loc_loss.sum() / batch_size_dev
        loc_loss_reduced *= self.loc_loss_weight
        
        # Accumulate classification loss
        cls_pos_loss, cls_neg_loss = get_pos_neg_loss(cls_loss, labels)
        cls_pos_loss /= self.pos_cls_weight
        cls_neg_loss /= self.neg_cls_weight
        cls_loss_reduced = cls_loss.sum() / batch_size_dev
        cls_loss_reduced *= self.cls_loss_weight
        # constructing a dictionary for the tracker
        if self.return_results_for_tracker:
            losses = {}
            losses['loc_loss'] = loc_loss_reduced.item()
            losses['cls_loss'] = cls_loss_reduced.item()
            losses['cls_pos_loss'] = cls_pos_loss.item()
            losses['cls_neg_loss'] = cls_neg_loss.item()
        
        # total loss (without direction classifier)
        loss = loc_loss_reduced + cls_loss_reduced
        
        if self.use_direction_classifier:
            dir_targets = get_direction_target(
                example['anchors'],
                reg_targets, 
                dir_offset=self.direction_limit_offset,
                num_bins=self.num_direction_bins
            )
            dir_logits = network_out['dir_cls_preds'].view(
                batch_size_dev, -1, self.num_direction_bins)
            weights = (labels > 0).type_as(dir_logits) * importance
            weights /= torch.clamp(weights.sum(-1, keepdim=True), min=1.0)
            dir_loss = self.dir_loss_ftor(
            dir_logits, dir_targets, weights=weights)
            dir_loss = dir_loss.sum() / batch_size_dev

            #adding direction loss to losses
            if self.return_results_for_tracker:
                losses['dir_loss'] = dir_loss.item()

            # adding direction loss to total loss
            loss += dir_loss * self.direction_loss_weight

        # finally adding the total loss
        if self.return_results_for_tracker:
            losses['loss'] = loss.item()

            # returning both the loss and the losses for the tracker
            return loss, losses
        
        # returning only the loss
        return loss




        
        