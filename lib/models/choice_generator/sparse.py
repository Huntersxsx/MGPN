from torch import nn
import torch

class SparseBoundaryContent(nn.Module):
    def __init__(self, cfg):
        super(SparseBoundaryContent, self).__init__()
        pooling_counts = cfg.NUM_SCALE_LAYERS   
        N = cfg.NUM_CLIPS
        mask2d = torch.zeros(N, N, dtype=torch.bool) 
        mask2d[range(N), range(N)] = 1 
        # self.linear = nn.Linear(1024, 512)
        self.conv1d = nn.Conv1d(1024, 512, 3, 1, 1)

        poolers = [nn.MaxPool1d(2,1) for _ in range(pooling_counts[0])]
        for c in pooling_counts[1:]:
            poolers.extend(
                [nn.MaxPool1d(3,2)] + [nn.MaxPool1d(2,1) for _ in range(c - 1)]
            )

        stride, offset = 1, 0
        maskij = []
        for c in pooling_counts:
            for _ in range(c): 
                offset += stride
                i, j = range(0, N - offset, stride), range(offset, N, stride)
                mask2d[i, j] = 1
                maskij.append((i, j))
            stride *= 2

        self.mask2d = mask2d
        self.maskij = maskij
        self.poolers = poolers

    def forward(self, x):
        B, D, N = x.shape
        boundary_map2d = x.new_zeros(B, D, N, N).cuda()
        content_map2d = x.new_zeros(B, D, N, N).cuda()
        mask2d = self.mask2d.unsqueeze(0).unsqueeze(0).repeat(B, 1, 1, 1).cuda()
        # boundary_map2d[:, :, range(N), range(N)] = x.repeat(1, 2, 1)
        boundary_map2d[:, :, range(N), range(N)] = x
        content_map2d[:, :, range(N), range(N)] = x

        for (i, j) in self.maskij:
            boundary_map2d[:, :, i, j] = (x[:, :, i] + x[:, :, j]) / 2
            # boundary_map2d[:, :, i, j] = torch.cat((x[:, :, i], x[:, :, j]), dim=1)

        for pooler, (i, j) in zip(self.poolers, self.maskij):
            x = pooler(x)
            content_map2d[:, :, i, j] = x
        return boundary_map2d, content_map2d, mask2d
