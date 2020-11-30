""""xyz2rgb(lab2xyz(lab, illuminant, observer))"""
import numpy as np
import torch
from scipy import linalg
from torch.autograd import Variable

xyz_from_rgb = np.array(
    [[0.412453, 0.357580, 0.180423], [0.212671, 0.715160, 0.072169], [0.019334, 0.119193, 0.950227]]
)

# rgb_from_xyz = linalg.inv(xyz_from_rgb).T
rgb_from_xyz = np.array(
    [[3.24048134, -0.96925495, 0.05564664], [-1.53715152, 1.87599, -0.20404134], [-0.49853633, 0.04155593, 1.05731107]]
)


def tensor_lab2rgb(input):
    """
    n * 3* h *w
    """
    input_trans = input.transpose(1, 2).transpose(2, 3)  # n * h * w * 3
    L, a, b = input_trans[:, :, :, 0:1], input_trans[:, :, :, 1:2], input_trans[:, :, :, 2:]
    y = (L + 16.0) / 116.0
    x = (a / 500.0) + y
    z = y - (b / 200.0)

    neg_mask = z.data < 0
    z[neg_mask] = 0
    # if neg_mask.any():
    #     z[neg_mask] = 0
    xyz = torch.cat((x, y, z), dim=3)

    mask = xyz.data > 0.2068966
    mask_xyz = xyz.clone()
    mask_xyz[mask] = torch.pow(xyz[mask], 3.0)
    mask_xyz[~mask] = (xyz[~mask] - 16.0 / 116.0) / 7.787
    # if mask.any():
    #     mask_xyz[mask] = torch.pow(xyz[mask], 3.)
    # if not mask.all():
    #     mask_xyz[~mask] = (xyz[~mask] - 16.0 / 116.) / 7.787
    mask_xyz[:, :, :, 0] = mask_xyz[:, :, :, 0] * 0.95047
    mask_xyz[:, :, :, 2] = mask_xyz[:, :, :, 2] * 1.08883

    rgb_trans = torch.mm(mask_xyz.view(-1, 3), torch.from_numpy(rgb_from_xyz).type_as(xyz)).view(
        input.size(0), input.size(2), input.size(3), 3
    )
    rgb = rgb_trans.transpose(2, 3).transpose(1, 2)

    mask = rgb > 0.0031308
    mask_rgb = rgb.clone()
    mask_rgb[mask] = 1.055 * torch.pow(rgb[mask], 1 / 2.4) - 0.055
    mask_rgb[~mask] = rgb[~mask] * 12.92
    # if mask.any():
    #     mask_rgb[mask] = 1.055 * torch.pow(rgb[mask], 1 / 2.4) - 0.055
    # if not mask.all():
    #     mask_rgb[~mask] = rgb[~mask] * 12.92

    neg_mask = mask_rgb.data < 0
    large_mask = mask_rgb.data > 1
    mask_rgb[neg_mask] = 0
    mask_rgb[large_mask] = 1
    # if neg_mask.any():
    #     mask_rgb[neg_mask] = 0
    # if large_mask.any():
    #     mask_rgb[large_mask] = 1

    return mask_rgb
