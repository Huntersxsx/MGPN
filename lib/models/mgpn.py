from torch import nn
import torch
import math
import numpy as np

from core.config import config
import models.feature_encoder as feature_encoder
import models.choice_generator as choice_generator
import models.finegrained_encoder as finegrained_encoder
import models.modality_interactor as modality_interactor
import models.relation_constructor as relation_constructor


class MGPN(nn.Module):
    def __init__(self):
        super(MGPN, self).__init__()

        self.encoder_layer = getattr(feature_encoder, config.MGPN.COARSE_GRAINED_ENCODER.NAME)(config.MGPN.COARSE_GRAINED_ENCODER.PARAMS)
        self.interactor_layer1 = getattr(modality_interactor, config.MGPN.COATTENTION_MODULE.NAME)(config.MGPN.COATTENTION_MODULE.PARAMS)
        self.generator_layer = getattr(choice_generator, config.MGPN.CHOICE_GENERATOR.NAME)(config.MGPN.CHOICE_GENERATOR.PARAMS)
        self.finegrained_layer = getattr(finegrained_encoder, config.MGPN.FINE_GRAINED_ENCODER.NAME)(config.MGPN.FINE_GRAINED_ENCODER.PARAMS)
        self.interactor_layer2 = getattr(modality_interactor, config.MGPN.CONDITIONED_INTERACTION_MODULE.NAME)(config.MGPN.CONDITIONED_INTERACTION_MODULE.PARAMS)
        self.relation_layer = getattr(relation_constructor, config.MGPN.CHOICE_COMPARISON_MODULE.NAME)(config.MGPN.CHOICE_COMPARISON_MODULE.PARAMS)
        self.pred_layer = nn.Conv2d(config.MGPN.PRED_INPUT_SIZE, 1, 1, 1)

        # self.pred_layer2 = nn.Conv2d(config.RANET.PRED_INPUT_SIZE, 1, 1, 1)

    def forward(self, textual_input, textual_mask, visual_input):

        vis_encoded, txt_encoded = self.encoder_layer(visual_input, textual_input, textual_mask) 
        vis_fused, txt_fused = self.interactor_layer1(vis_encoded, txt_encoded)
        boundary_map, content_map, map_mask = self.generator_layer(vis_fused) 
        vis_finegrained, txt_finegrained = self.finegrained_layer(vis_fused, txt_fused)  
        # boundary_map, content_map, map_mask = self.generator_layer(vis_finegrained) 
        fused_map = self.interactor_layer2(boundary_map, content_map, map_mask, vis_finegrained, txt_finegrained) 
        # fused_map = torch.cat((boundary_map, content_map), dim=1) * map_mask.float()
        relation_map = self.relation_layer(fused_map, boundary_map, content_map, map_mask) 
        # relation_map = self.relation_layer(fused_map, map_mask)   
        score_map = self.pred_layer(relation_map) * map_mask.float() 

        # score_map = self.pred_layer(content_map) * map_mask.float()

        # coarse_score = self.pred_layer2(boundary_map + content_map) * map_mask.float() 
        # score_map = torch.sigmoid(coarse_score) * score_map * map_mask.float() 
        # score_map = 2 * score_map + coarse_score

        return score_map, map_mask


