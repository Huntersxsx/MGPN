import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
import sys, os

from models.relation_constructor import get_padded_mask_and_weight
from .non_local import NONLocalBlock2D

class MapConv(nn.Module):

    def __init__(self, cfg):
        super(MapConv, self).__init__()
        input_size = cfg.INPUT_SIZE
        hidden_sizes = cfg.HIDDEN_SIZES
        kernel_sizes = cfg.KERNEL_SIZES
        strides = cfg.STRIDES
        paddings = cfg.PADDINGS
        dilations = cfg.DILATIONS
        self.convs = nn.ModuleList()
        assert len(hidden_sizes) == len(kernel_sizes) \
               and len(hidden_sizes) == len(strides) \
               and len(hidden_sizes) == len(paddings) \
               and len(hidden_sizes) == len(dilations)
        channel_sizes = [input_size * 2]+hidden_sizes
        for i, (k, s, p, d) in enumerate(zip(kernel_sizes, strides, paddings, dilations)):
            self.convs.append(nn.Conv2d(channel_sizes[i], channel_sizes[i+1], k, s, p, d))

        self.conv_layer = nn.Conv2d(cfg.INPUT_SIZE * 4, cfg.INPUT_SIZE * 2, 1, 1)
        self.relu = nn.ReLU(True)

    def forward(self, map_h, boundary_map, content_map, map_mask):
        fused_map = torch.cat((map_h, boundary_map, content_map), dim=1) * map_mask.float() 
        x = (self.relu(self.conv_layer(fused_map)) + map_h) * map_mask.float() 
        padded_mask = map_mask
        for i, pred in enumerate(self.convs):
            x = F.relu(pred(x))
            padded_mask, masked_weight = get_padded_mask_and_weight(padded_mask, pred)
            x = x * masked_weight
        return x


class ConvBlock(nn.Module):
    def __init__(self, input_size, output_size, kernel_size, paddings, groups):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(input_size, output_size, kernel_size, 1, paddings, groups=groups)
        self.bn = nn.BatchNorm2d(output_size)
        self.relu = nn.ReLU()
        # self.dropout = nn.Dropout(p=0.2)
    def forward(self, x):
        x = self.relu(self.bn(self.conv(x)))
        # x = self.dropout(x)
        return x



class ConvModule(nn.Module):
    def __init__(self, cfg):
        super(ConvModule, self).__init__()
        self.block_num = cfg.BLOCK_NUM
        self.conv_layer = nn.Conv2d(cfg.INPUT_SIZE * 4, cfg.INPUT_SIZE * 2, 1, 1)
        # self.conv_block = nn.ModuleList([ConvBlock(cfg) for _ in range(self.block_num)])
        self.conv_block1 = ConvBlock(cfg.INPUT_SIZE * 2, cfg.INPUT_SIZE * 2, 7, 3, 32)
        self.conv_block2 = ConvBlock(cfg.INPUT_SIZE * 2, cfg.OUTPUT_SIZE, 7, 3, 32)
        self.conv_block3 = ConvBlock(cfg.OUTPUT_SIZE, cfg.OUTPUT_SIZE, 7, 3, 32)
        self.conv_block4 = ConvBlock(cfg.OUTPUT_SIZE, cfg.OUTPUT_SIZE, 7, 3, 32)
        self.relu = nn.ReLU(True)

    def forward(self, map_h, boundary_map, content_map, map_mask):
        fused_map = torch.cat((map_h, boundary_map, content_map), dim=1) * map_mask.float() 
        x = (self.relu(self.conv_layer(fused_map)) + map_h) * map_mask.float() 
        # for i in range(len(self.conv_block)):
        #     x = self.conv_block[i](x) * map_mask.float()
        x = self.conv_block1(x) * map_mask.float()
        x = self.conv_block2(x) * map_mask.float()
        x = self.conv_block3(x) * map_mask.float() 
        x = self.conv_block4(x) * map_mask.float()
        return x 



class NoNLocalModule(nn.Module):
    def __init__(self, cfg):
        super(NoNLocalModule, self).__init__()
        self.block_num = cfg.BLOCK_NUM
        # self.NL_block  = NONLocalBlock2D(256, sub_sample=False, bn_layer=True)
        self.NL_block = nn.ModuleList([NONLocalBlock2D(256, sub_sample=True, bn_layer=True) for _ in range(self.block_num)])
        self.conv_layer = nn.Conv2d(cfg.INPUT_SIZE * 4, cfg.INPUT_SIZE * 2, 1, 1)
        self.relu = nn.ReLU(True)
        self.conv_layer2 = nn.Conv2d(cfg.INPUT_SIZE * 2, cfg.INPUT_SIZE, 1, 1)

    def forward(self, map_h, boundary_map, content_map, map_mask):
        fused_map = torch.cat((map_h, boundary_map, content_map), dim=1) * map_mask.float() 
        x = (self.relu(self.conv_layer(fused_map)) + map_h) * map_mask.float() 
        x = self.conv_layer2(x) * map_mask.float() 
        for i in range(len(self.NL_block)):
            x = self.NL_block[i](x, map_mask)
        # x = self.NL_block(x) * map_mask.float() 
        return x