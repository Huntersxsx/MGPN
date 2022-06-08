import torch
from torch import nn
import numpy as np
import math, copy, time
import torch.nn.functional as F
from torch.autograd import Variable
from .transformer_enc import TransformerBlock


class LearnPositionalEncoding(nn.Module):

    def __init__(self, d_model, max_len=64, dropout=0.1):
        super(LearnPositionalEncoding, self).__init__()
        self.pos_embed = nn.Embedding(max_len, d_model)

        nn.init.uniform_(self.pos_embed.weight)

        self.dropout = nn.Dropout(p=dropout)


    def forward(self, q):
        bsz_q, d_model, q_frm = q.shape
        assert q_frm == self.pos_embed.weight.shape[0], (q_frm,self.pos_embed.weight.shape)
        q_pos = self.pos_embed.weight.clone()
        q_pos = q_pos.unsqueeze(0)
        q_pos = q_pos.expand(bsz_q, q_frm, d_model).transpose(1,2)
        # q_pos = q_pos.contiguous().view(bsz_q, q_frm, n_head, d_k)
        q = q + q_pos
        return self.dropout(q)


class FrameAvgPool(nn.Module):

    def __init__(self, input_size, hidden_size, kernel_size, stride, use_position, num_clips):
        super(FrameAvgPool, self).__init__()
        self.vis_conv = nn.Conv1d(input_size, hidden_size, 1, 1)
        self.avg_pool = nn.AvgPool1d(kernel_size, stride)

        if use_position:
            self.pos_embed = LearnPositionalEncoding(d_model=hidden_size, max_len=num_clips)
        else:
            self.pos_embed = None

    def forward(self, visual_input):
        vis_h = torch.relu(self.vis_conv(visual_input))
        vis_h = self.avg_pool(vis_h)
        if self.pos_embed:
            vis_h = self.pos_embed(vis_h) 
        return vis_h


class FeatureEncoder(nn.Module):

    def __init__(self, cfg):
        super(FeatureEncoder, self).__init__()
        self.frame_encoder = FrameAvgPool(cfg.FRAME.INPUT_SIZE, cfg.FRAME.HIDDEN_SIZE,cfg.FRAME.KERNEL_SIZE,cfg.FRAME.STRIDE,
                                        cfg.FRAME.USE_POSITION,cfg.FRAME.NUM_CLIPS)
        self.vis_encoder = nn.GRU(cfg.GRU.VIS_INPUT_SIZE, cfg.GRU.VIS_HIDDEN_SIZE//2 if cfg.GRU.BIDIRECTIONAL else cfg.GRU.VIS_HIDDEN_SIZE,
                                       num_layers=cfg.GRU.NUM_LAYERS, bidirectional=cfg.GRU.BIDIRECTIONAL, batch_first=True)
        self.txt_encoder = nn.GRU(cfg.GRU.TXT_INPUT_SIZE, cfg.GRU.TXT_HIDDEN_SIZE//2 if cfg.GRU.BIDIRECTIONAL else cfg.GRU.TXT_HIDDEN_SIZE,
                                       num_layers=cfg.GRU.NUM_LAYERS, bidirectional=cfg.GRU.BIDIRECTIONAL, batch_first=True)
        # self.vis_encoder = nn.LSTM(cfg.GRU.VIS_INPUT_SIZE, cfg.GRU.VIS_HIDDEN_SIZE//2 if cfg.GRU.BIDIRECTIONAL else cfg.GRU.VIS_HIDDEN_SIZE,
        #                                num_layers=cfg.GRU.NUM_LAYERS, bidirectional=cfg.GRU.BIDIRECTIONAL, batch_first=True)
        # self.txt_encoder = nn.LSTM(cfg.GRU.TXT_INPUT_SIZE, cfg.GRU.TXT_HIDDEN_SIZE//2 if cfg.GRU.BIDIRECTIONAL else cfg.GRU.TXT_HIDDEN_SIZE,
        #                                num_layers=cfg.GRU.NUM_LAYERS, bidirectional=cfg.GRU.BIDIRECTIONAL, batch_first=True)


    def forward(self, visual_input, textual_input, textual_mask):
        visual_input = visual_input.transpose(-1, -2)  
        vis_out = self.frame_encoder(visual_input)  # B, C, T
        bs, dimc, _ = vis_out.size()
        vis_out = vis_out.transpose(-1, -2).view(bs, -1, dimc)   # B, T, C
        self.vis_encoder.flatten_parameters()
        vis_out = self.vis_encoder(vis_out)[0].transpose(-1, -2)
        self.txt_encoder.flatten_parameters()
        txt_out = self.txt_encoder(textual_input)[0] * textual_mask  # B, L, C
        
        return vis_out, txt_out
