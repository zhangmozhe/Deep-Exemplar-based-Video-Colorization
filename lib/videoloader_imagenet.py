import os
import os.path as osp
import pdb
import struct
import time

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image
from scipy.misc import imread, imresize, imsave
from scipy.ndimage import geometric_transform, map_coordinates
from scipy.ndimage.filters import gaussian_filter
from skimage import color
from utils.util_distortion import (
    CenterPad,
    CenterPadCrop_numpy,
    Distortion,
    Distortion_with_flow,
    Normalize,
    RGB2Lab,
    ToTensor,
)

import lib.functional as F
from lib.RefPairDatasetWithGlobalOrigRGB_optimize import ToTensor, combo5_loader, image_loader, parse_images

cv2.setNumThreads(0)


class VideosDataset_ImageNet(data.Dataset):
    def __init__(
        self,
        data_root,
        epoch,
        image_size,
        with_bad=False,
        with_mid=False,
        transforms_imagenet=None,
        distortion_level=3,
        brightnessjitter=0,
        nonzero_placeholder_probability=0.5,
        extra_reference_transform=None,
        real_reference_probability=1,
    ):
        image_pairs = []
        curr_image_pairs = parse_images(data_root, with_bad, with_mid)
        image_pairs += curr_image_pairs
        print("##### parsing image_a pairs in %s: %d pairs #####" % (data_root, len(curr_image_pairs)))
        if len(image_pairs) == 0:
            raise RuntimeError("Found 0 image_a pairs in all the data_roots")

        self.image_pairs = image_pairs
        self.transforms_imagenet_raw = transforms_imagenet
        self.extra_reference_transform = transforms.Compose(extra_reference_transform)
        self.real_reference_probability = real_reference_probability
        self.transforms_imagenet = transforms.Compose(transforms_imagenet)
        self.epoch = epoch
        self.image_size = image_size
        self.real_len = len(self.image_pairs)
        self.image_pairs *= epoch
        self.distortion_level = distortion_level
        self.distortion_transform = Distortion_with_flow()
        self.brightnessjitter = brightnessjitter
        self.flow_transform = transforms.Compose([CenterPadCrop_numpy(self.image_size), ToTensor()])
        self.nonzero_placeholder_probability = nonzero_placeholder_probability
        self.ToTensor = ToTensor()
        self.Normalize = Normalize()

    def __getitem__(self, index):
        try:
            # print('imagenet loader')

            image_id = 0
            pair_id = index

            combo_path = None
            image_a_path = None
            image_b_path = None

            image_names = ["", ""]
            dir_root, cls_dir, image_names[0], image_names[1], is_good = self.image_pairs[pair_id]
            sub_dir = osp.join(dir_root, cls_dir)
            if is_good >= 1:
                image_a_path = osp.join(sub_dir, "input", "%s.JPEG" % image_names[image_id])
                image_b_path = osp.join(sub_dir, "input", "%s.JPEG" % image_names[1 - image_id])
            elif is_good == 0:
                image_a_path = osp.join(sub_dir, "input_mid", "%s.JPEG" % image_names[image_id])
                image_b_path = osp.join(sub_dir, "input_mid", "%s.JPEG" % image_names[1 - image_id])
            else:
                image_a_path = osp.join(sub_dir, "input_bad", "%s.JPEG" % image_names[image_id])
                image_b_path = osp.join(sub_dir, "input_bad", "%s.JPEG" % image_names[1 - image_id])

            if np.random.random() > 0.5:
                image_a_path, image_b_path = image_b_path, image_a_path

            I1 = image_loader(image_a_path)
            I2 = I1
            I_reference_video = I1
            I_reference_video_real = image_loader(image_b_path)

            ## generate the flow
            height, width = np.array(I2).shape[0], np.array(I2).shape[1]
            alpha = np.random.rand() * self.distortion_level
            distortion_range = 50
            random_state = np.random.RandomState(None)
            shape = self.image_size[0], self.image_size[1]
            # dx: flow on the vertical direction; dy: flow on the horizontal direction
            forward_dx = (
                gaussian_filter((random_state.rand(*shape) * 2 - 1), distortion_range, mode="constant", cval=0)
                * alpha
                * 1000
            )
            forward_dy = (
                gaussian_filter((random_state.rand(*shape) * 2 - 1), distortion_range, mode="constant", cval=0)
                * alpha
                * 1000
            )

            for transform in self.transforms_imagenet_raw:
                if type(transform) is RGB2Lab:
                    I1_raw = I1
                    I1 = transform(I1)
                else:
                    I1 = transform(I1)

            for transform in self.transforms_imagenet_raw:
                if type(transform) is RGB2Lab:
                    I2 = self.distortion_transform(I2, forward_dx, forward_dy)
                    I2_raw = I2
                I2 = transform(I2)
            I2[0:1, :, :] = I2[0:1, :, :] + torch.randn(1) * self.brightnessjitter

            I_reference_video = self.extra_reference_transform(I_reference_video)
            for transform in self.transforms_imagenet_raw:
                # if type(transform) is RGB2Lab and self.extra_reference_transform is not None:
                # I_reference_video = self.extra_reference_transform(I_reference_video)
                I_reference_video = transform(I_reference_video)

            I_reference_video_real = self.transforms_imagenet(I_reference_video_real)

            flow_forward_raw = np.stack((forward_dy, forward_dx), axis=-1)
            flow_backward_raw = np.zeros_like(flow_forward_raw)
            flow_forward = self.flow_transform(flow_forward_raw)
            flow_backward = self.flow_transform(flow_backward_raw)

            # update the mask for the pixels on the border
            grid_x, grid_y = np.meshgrid(np.arange(self.image_size[0]), np.arange(self.image_size[1]), indexing="ij")
            grid = np.stack((grid_y, grid_x), axis=-1)
            grid_warp = grid + flow_forward_raw
            location_y = grid_warp[:, :, 0].flatten()
            location_x = grid_warp[:, :, 1].flatten()
            I2_raw = np.array(I2_raw).astype(float)
            I21_r = map_coordinates(I2_raw[:, :, 0], np.stack((location_x, location_y)), cval=-1).reshape(
                (self.image_size[0], self.image_size[1])
            )
            I21_g = map_coordinates(I2_raw[:, :, 1], np.stack((location_x, location_y)), cval=-1).reshape(
                (self.image_size[0], self.image_size[1])
            )
            I21_b = map_coordinates(I2_raw[:, :, 2], np.stack((location_x, location_y)), cval=-1).reshape(
                (self.image_size[0], self.image_size[1])
            )
            I21_raw = np.stack((I21_r, I21_g, I21_b), axis=2)
            mask = np.ones((self.image_size[0], self.image_size[1]))
            mask[(I21_raw[:, :, 0] == -1) & (I21_raw[:, :, 1] == -1) & (I21_raw[:, :, 2] == -1)] = 0
            mask[abs(I21_raw - I1_raw).sum(axis=-1) > 50] = 0
            mask = self.ToTensor(mask)

            if np.random.random() < self.real_reference_probability:
                I_reference_output = I_reference_video_real
                placeholder = torch.zeros_like(I1)
                self_ref_flag = torch.zeros_like(I1)
            else:
                I_reference_output = I_reference_video
                placeholder = I2 if np.random.random() < self.nonzero_placeholder_probability else torch.zeros_like(I1)
                self_ref_flag = torch.ones_like(I1)

            outputs = []
            outputs.append(I1)
            outputs.append(I2)
            outputs.append(I_reference_output)
            # outputs.append(I_reference_video_real)
            outputs.append(flow_forward)
            outputs.append(flow_backward)
            outputs.append(mask)
            outputs.append(placeholder)
            outputs.append(self_ref_flag)

        except Exception as e:
            if combo_path is not None:
                print("problem in ", combo_path)
            print("problem in, ", image_a_path)
            print(e)
            return self.__getitem__(np.random.randint(0, len(self.image_pairs)))

        return outputs

    def __len__(self):
        return len(self.image_pairs)
        # return len(self.image_pairs) * 2
