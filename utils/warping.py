import torch
import torch.nn as nn
import torch.nn.functional as F


def get_grid(x):
    torchHorizontal = (
        torch.linspace(-1.0, 1.0, x.size(3)).view(1, 1, 1, x.size(3)).expand(x.size(0), 1, x.size(2), x.size(3))
    )
    torchVertical = (
        torch.linspace(-1.0, 1.0, x.size(2)).view(1, 1, x.size(2), 1).expand(x.size(0), 1, x.size(2), x.size(3))
    )
    return torch.cat([torchHorizontal, torchVertical], 1).cuda()


class WarpingLayer(nn.Module):
    def __init__(self, device):
        super(WarpingLayer, self).__init__()
        self.device = device

    def forward(self, x, flow):
        # WarpingLayer uses F.grid_sample, which expects normalized grid
        # we still output unnormalized flow for the convenience of comparing EPEs with FlowNet2 and original code
        # so here we need to denormalize the flow
        flow_for_grip = torch.zeros_like(flow).cuda()
        flow_for_grip[:, 0, :, :] = flow[:, 0, :, :] / ((flow.size(3) - 1.0) / 2.0)
        flow_for_grip[:, 1, :, :] = flow[:, 1, :, :] / ((flow.size(2) - 1.0) / 2.0)

        grid = (get_grid(x) + flow_for_grip).permute(0, 2, 3, 1)
        return F.grid_sample(x, grid, align_corners=True)
