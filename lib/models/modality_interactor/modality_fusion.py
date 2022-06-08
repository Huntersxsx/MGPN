import torch
from torch import nn
import torch.nn.functional as F
import math
import numpy as np
import pdb
from torch.autograd import Variable


class FirstFuse(nn.Module):
    def __init__(self, cfg):
        super(FirstFuse, self).__init__()
        self.cfg = cfg
        self.txt_softmax = nn.Softmax(1)
        self.vis_softmax = nn.Softmax(2)
        self.txt_linear1 = nn.Linear(cfg.TXT_INPUT_SIZE, 1)
        self.vis_conv = nn.Conv1d(cfg.VIS_INPUT_SIZE, 1, 1, 1)
        self.avg_pool = nn.AvgPool1d(kernel_size=cfg.NUM_CLIPS, stride=cfg.NUM_CLIPS)

    def forward(self, vis_encoded, txt_encoded):
        # vis_encoded: B, C, T
        # txt_encoded: B, L, C
        txt_attn = self.txt_softmax(self.txt_linear1(txt_encoded))  # B, L, 1
        txt_pool = torch.sum(txt_attn * txt_encoded, dim=1)[:,:,None]  # B, C, 1
        vis_fused = F.normalize(txt_pool * vis_encoded) # B, C, T
        vis_attn = self.vis_softmax(self.vis_conv(vis_encoded))  # B, C, T
        vis_pool = torch.sum(vis_attn * vis_encoded, dim=2)[:,None,:]
        txt_fused = F.normalize(vis_pool * txt_encoded)
        return vis_fused, txt_fused


class SecondFuse(nn.Module):
    def __init__(self, cfg):
        super(SecondFuse, self).__init__()

        self.cfg = cfg
        self.txt_linear_b1 = nn.Linear(cfg.TXT_INPUT_SIZE, cfg.HIDDEN_SIZE)
        self.vis_linear_b2 = nn.Linear(cfg.VIS_INPUT_SIZE, cfg.HIDDEN_SIZE)
        self.softmax = nn.Softmax(dim=1)
        self.relu = nn.ReLU(True)
        self.avg_pool = nn.AvgPool1d(kernel_size=cfg.NUM_CLIPS, stride=cfg.NUM_CLIPS)

    def forward(self, boundary_map, content_map, map_mask, vis_h, txt_h):
        map_mask = map_mask.float()
        txt_pool = torch.max(txt_h, dim=1)[0]  # B, C
        txt_h_b1 = self.txt_linear_b1(txt_pool)[:,:,None,None]  # f_s: B, C, 1, 1
        gate_b1 = torch.sigmoid(boundary_map * txt_h_b1)  # gate_b1: B, C, T, T
        fused_b1 = gate_b1 * content_map  # query-related content representation: B, C, T, T

        vis_pool = self.avg_pool(vis_h)
        vis_pool = vis_pool.squeeze(-1)
        vis_h_b2 = self.vis_linear_b2(vis_pool)[:,:,None,None]  # f_s: B, C, 1, 1
        gate_b2 = torch.sigmoid(boundary_map * vis_h_b2)  # gate_b3: B, C, T, T
        fused_b2 = gate_b2 * content_map
        fused_h = torch.cat((fused_b1, fused_b2), dim=1) * map_mask
        return fused_h