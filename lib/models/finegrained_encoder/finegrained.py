import torch
import torch.nn as nn
import numpy as np
import math, copy, time
import torch.nn.functional as F
from torch.autograd import Variable


class LayerNorm(nn.Module):

    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class SublayerConnection(nn.Module):

    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        return x + self.dropout(sublayer(self.norm(x)))


class FeedForwardNet(nn.Module):
    "Implements FFN equation."

    def __init__(self, d_in, d_ff, d_out, dropout=0.1):
        super(FeedForwardNet, self).__init__()
        self.w_1 = nn.Linear(d_in, d_ff)
        self.w_2 = nn.Linear(d_ff, d_out)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU()

    def forward(self, x):
        return self.w_2(self.dropout(self.activation(self.w_1(x))))



class PhraseEncodeNet(nn.Module):

    def __init__(self, d_in, d_out):
        super(PhraseEncodeNet, self).__init__()
        self.unigram_conv = nn.Conv1d(d_in, d_out, 1, stride=1, padding=0)
        self.bigram_conv  = nn.Conv1d(d_in, d_out, 2, stride=1, padding=1, dilation=2)
        self.trigram_conv = nn.Conv1d(d_in, d_out, 3, stride=1, padding=2, dilation=2)

    def forward(self, x):
        bs, _, dimc = x.size()
        words = x.transpose(-1, -2)  # B, C, L
        unigrams = self.unigram_conv(words)
        bigrams  = self.bigram_conv(words)  # B, C, L
        trigrams = self.trigram_conv(words)
        phrase = torch.cat((unigrams, bigrams, trigrams), dim=1)
        return phrase.transpose(-1, -2).view(bs, -1, dimc * 3)


class FineGrainedFeature(nn.Module):
    def __init__(self, cfg):
        super(FineGrainedFeature, self).__init__()
        self.phrase_encode = PhraseEncodeNet(d_in=cfg.TXT_INPUT_SIZE, d_out=cfg.TXT_OUTPUT_SIZE)
        self.feed_forward = FeedForwardNet(d_in=cfg.VIS_INPUT_SIZE, d_ff=cfg.VIS_HIDDEN_SIZE, d_out=cfg.VIS_OUTPUT_SIZE, dropout=0.1)
        self.vis_sublayer = SublayerConnection(size=cfg.VIS_INPUT_SIZE, dropout=0.1)
        self.txt_linear = nn.Linear(cfg.TXT_OUTPUT_SIZE * 3, cfg.TXT_OUTPUT_SIZE)

    def forward(self, vis_h, txt_h):
        bs, dimc, _ = vis_h.size()
        phrase = self.phrase_encode(txt_h)
        txt_h =self.txt_linear(phrase)
        vis_h = vis_h.transpose(-1, -2)  # B, T, C
        vis_h = self.vis_sublayer(vis_h, self.feed_forward).transpose(-1, -2).view(bs, dimc, -1)  # B, C, T
        return vis_h, txt_h
