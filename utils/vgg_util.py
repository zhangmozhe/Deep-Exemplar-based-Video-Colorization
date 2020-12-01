import collections
import os

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image


def preprocess(img, scale_size=None):
    """PILimg: RGB: HxWxC"""
    if scale_size is not None:
        scale_transforms = MaxScale(scale_size)
        img = scale_transforms(img)

    prep_transforms = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x[torch.LongTensor([2, 1, 0])]),  # turn to BGR
            transforms.Normalize(mean=[0.40760392, 0.45795686, 0.48501961], std=[1, 1, 1]),  # subtract imagenet mean
            transforms.Lambda(lambda x: x.mul_(255)),
        ]
    )
    img = prep_transforms(img)
    return img


def deprocess(img):
    post_transforms_a = transforms.Compose(
        [
            transforms.Lambda(lambda x: x.mul_(1.0 / 255)),
            transforms.Normalize(mean=[-0.40760392, -0.45795686, -0.48501961], std=[1, 1, 1]),  # add imagenet mean
            transforms.Lambda(lambda x: x[torch.LongTensor([2, 1, 0])]),  # turn to RGB
        ]
    )
    post_transforms_b = transforms.Compose([transforms.ToPILImage()])
    img = post_transforms_a(img)
    img[img > 1] = 1
    img[img < 0] = 0
    img = post_transforms_b(img)
    return img


def get_renamed_vgg():
    cache_file = "data/vgg19_conv.pth"
    vgg = models.vgg19().features
    renamed_vgg = nn.Sequential()
    part_idx, layer_idx = 1, 1
    for layer in list(vgg):
        if isinstance(layer, nn.Conv2d):
            name = "conv{}_{}".format(part_idx, layer_idx)
            renamed_vgg.add_module(name, layer)
        elif isinstance(layer, nn.ReLU):
            name = "relu{}_{}".format(part_idx, layer_idx)
            renamed_vgg.add_module(name, layer)
            layer_idx += 1
        elif isinstance(layer, nn.MaxPool2d):
            name = "pool{}".format(part_idx)
            renamed_vgg.add_module(name, layer)
            part_idx += 1
            layer_idx = 1
    renamed_vgg.load_state_dict(torch.load(cache_file))

    return renamed_vgg


def get_renamed_vgg_johnson():
    cache_file = "data/vgg19-d01eb7cb.pth"
    vgg = models.vgg19()
    model_dict = vgg.state_dict()
    pretrained_dict = torch.load(cache_file)
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    vgg.load_state_dict(model_dict)
    vgg = vgg.features
    renamed_vgg = nn.Sequential()
    part_idx, layer_idx = 1, 1
    for layer in list(vgg):
        if isinstance(layer, nn.Conv2d):
            name = "conv{}_{}".format(part_idx, layer_idx)
            renamed_vgg.add_module(name, layer)
        elif isinstance(layer, nn.ReLU):
            name = "relu{}_{}".format(part_idx, layer_idx)
            renamed_vgg.add_module(name, layer)
            layer_idx += 1
        elif isinstance(layer, nn.MaxPool2d):
            name = "pool{}".format(part_idx)
            renamed_vgg.add_module(name, layer)
            part_idx += 1
            layer_idx = 1

    return renamed_vgg


class MaxScale(object):
    """Rescale the input PIL.Image to the given size.

    Args:
        size (sequence or int): Desired output size. If size is a sequence like
            (w, h), output size will be matched to this. If size is an int,
            smaller edge of the image will be matched to this number.
            i.e, if height > width, then image will be rescaled to
            (size * height / width, size)
        interpolation (int, optional): Desired interpolation. Default is
            ``PIL.Image.BILINEAR``
    """

    def __init__(self, size, interpolation=Image.BILINEAR):
        assert isinstance(size, int) or (isinstance(size, collections.Iterable) and len(size) == 2)
        self.size = size
        self.interpolation = interpolation

    def __call__(self, img):
        """
        Args:
            img (PIL.Image): Image to be scaled.

        Returns:
            PIL.Image: Rescaled image.
        """
        if not isinstance(self.size, int):
            return img.resize(self.size, self.interpolation)
        w, h = img.size
        if (w <= h == self.size) or (h <= w == self.size):
            return img
        if w < h:
            oh = self.size
            ow = int(self.size * w / h)
        else:
            ow = self.size
            oh = int(self.size * h / w)
        return img.resize((ow, oh), self.interpolation)
