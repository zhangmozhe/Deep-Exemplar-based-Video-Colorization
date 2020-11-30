import sys

sys.path.insert(0, "..")
import os
import struct

import cv2
import numpy as np
import torch.utils.data as data
from PIL import Image

cv2.setNumThreads(0)
import pdb
import random

import matplotlib.pyplot as plt
import torch
import torchvision
import torchvision.transforms as transforms
from PIL import Image
from skimage import color
from utils.flowlib import read_flow
from utils.util_distortion import CenterPad, CenterPadCrop_numpy
from utils.warping import WarpingLayer

import lib.functional as F


class RGB2Lab(object):
    def __init__(self):
        pass

    def __call__(self, inputs):
        image_lab = color.rgb2lab(inputs)
        return image_lab


class Normalize(object):
    def __init__(self):
        pass

    def __call__(self, inputs):
        inputs[0:1, :, :] = F.normalize(inputs[0:1, :, :], 50, 1)
        inputs[1:3, :, :] = F.normalize(inputs[1:3, :, :], (0, 0), (1, 1))
        return inputs


class ToTensor(object):
    def __init__(self):
        pass

    def __call__(self, inputs):
        outputs = F.to_mytensor(inputs)  # permute channel and transform to tensor
        return outputs


class RandomErasing(object):
    def __init__(self, probability=0.6, sl=0.05, sh=0.6):
        self.probability = probability
        self.sl = sl
        self.sh = sh

    def __call__(self, img):
        img = np.array(img)
        if random.uniform(0, 1) > self.probability:
            return Image.fromarray(img)

        area = img.shape[0] * img.shape[1]
        h0 = img.shape[0]
        w0 = img.shape[1]
        channel = img.shape[2]

        h = int(round(random.uniform(self.sl, self.sh) * h0))
        w = int(round(random.uniform(self.sl, self.sh) * w0))

        if w < img.shape[1] and h < img.shape[0]:
            x1 = random.randint(0, img.shape[0] - h)
            y1 = random.randint(0, img.shape[1] - w)
            img[x1 : x1 + h, y1 : y1 + w, :] = np.random.rand(h, w, channel) * 255
            return Image.fromarray(img)

        return Image.fromarray(img)


class CenterCrop(object):
    """
    center crop the numpy array
    """

    def __init__(self, image_size):
        # target size
        self.h0, self.w0 = image_size

    def __call__(self, input_numpy):
        if input_numpy.ndim == 3:
            h, w, channel = input_numpy.shape
            output_numpy = np.zeros((self.h0, self.w0, channel))
            output_numpy = input_numpy[
                (h - self.h0) // 2 : (h - self.h0) // 2 + self.h0, (w - self.w0) // 2 : (w - self.w0) // 2 + self.w0, :
            ]
        else:
            h, w = input_numpy.shape
            output_numpy = np.zeros((self.h0, self.w0))
            output_numpy = input_numpy[
                (h - self.h0) // 2 : (h - self.h0) // 2 + self.h0, (w - self.w0) // 2 : (w - self.w0) // 2 + self.w0
            ]
        return output_numpy


def parse_images(data_root):
    image_pairs = []
    subdirs = sorted(os.listdir(data_root))
    for subdir in subdirs:
        path = os.path.join(data_root, subdir)
        if not os.path.isdir(path):
            continue

        # parse_file = os.path.join(path, 'pairs_output.txt')
        parse_file = os.path.join(path, "pairs_output_new.txt")
        if os.path.exists(parse_file):
            f = open(parse_file, "r")
            lines = f.readlines()
            for line in lines:
                line = line.replace("\n", "")
                (
                    image1_name,
                    image2_name,
                    reference_video_name,
                    reference_video_name1,
                    reference_name1,
                    reference_name2,
                    reference_name3,
                    reference_name4,
                    reference_name5,
                    reference_gt1,
                    reference_gt2,
                    reference_gt3,
                ) = line.split()
                image1_name = image1_name.split(".")[0]
                image2_name = image2_name.split(".")[0]
                reference_video_name = reference_video_name.split(".")[0]
                reference_video_name1 = reference_video_name1.split(".")[0]
                reference_name1 = reference_name1.split(".")[0]
                reference_name2 = reference_name2.split(".")[0]
                reference_name3 = reference_name3.split(".")[0]
                reference_name4 = reference_name4.split(".")[0]
                reference_name5 = reference_name5.split(".")[0]

                reference_gt1 = reference_gt1.split(".")[0]
                reference_gt2 = reference_gt2.split(".")[0]
                reference_gt3 = reference_gt3.split(".")[0]

                flow_forward_name = image1_name + "_forward"
                flow_backward_name = image1_name + "_backward"
                mask_name = image1_name + "_mask"

                item = (
                    image1_name + ".jpg",
                    image2_name + ".jpg",
                    reference_video_name + ".jpg",
                    reference_name1 + ".JPEG",
                    reference_name2 + ".JPEG",
                    reference_name3 + ".JPEG",
                    reference_name4 + ".JPEG",
                    reference_name5 + ".JPEG",
                    flow_forward_name + ".flo",
                    flow_backward_name + ".flo",
                    mask_name + ".pgm",
                    reference_gt1 + ".jpg",
                    reference_gt2 + ".jpg",
                    reference_gt3 + ".jpg",
                    path,
                )
                image_pairs.append(item)

            f.close()

        else:
            raise (RuntimeError("Error when parsing pair_output_count.txt in subfolders of: " + path + "\n"))

    return image_pairs


class VideosDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        data_root,
        epoch,
        image_size,
        image_transform=None,
        use_google_reference=False,
        real_reference_probability=1,
        nonzero_placeholder_probability=0.5,
    ):
        self.data_root = data_root
        self.image_transform = image_transform
        self.CenterPad = CenterPad(image_size)
        self.ToTensor = ToTensor()
        self.CenterCrop = CenterCrop(image_size)

        assert len(self.data_root) > 0, "find no dataroot"
        self.epoch = epoch
        self.image_pairs = parse_images(self.data_root)
        self.real_len = len(self.image_pairs)
        print("##### parsing image pairs in %s: %d pairs #####" % (data_root, self.real_len))
        self.image_pairs *= epoch
        self.use_google_reference = use_google_reference
        self.real_reference_probability = real_reference_probability
        self.nonzero_placeholder_probability = nonzero_placeholder_probability

    def __getitem__(self, index):
        (
            image1_name,
            image2_name,
            reference_video_name,
            reference_name1,
            reference_name2,
            reference_name3,
            reference_name4,
            reference_name5,
            flow_forward_name,
            flow_backward_name,
            mask_name,
            reference_gt1,
            reference_gt2,
            reference_gt3,
            path,
        ) = self.image_pairs[index]
        try:
            # print('video loader')
            I1 = Image.open(os.path.join(path, "input_pad", image1_name))
            I2 = Image.open(os.path.join(path, "input_pad", image2_name))

            # randomly choose a reference frame from the video
            I_reference_video = Image.open(
                os.path.join(path, "reference_gt", random.choice([reference_gt1, reference_gt2, reference_gt3]))
            )
            I_reference_video_real = Image.open(
                os.path.join(
                    path,
                    "reference",
                    random.choice(
                        [reference_name1, reference_name2, reference_name3, reference_name4, reference_name5]
                    ),
                )
            )

            flow_forward = read_flow(os.path.join(path, "flow", flow_forward_name))  # numpy
            flow_backward = read_flow(os.path.join(path, "flow", flow_backward_name))  # numpy
            mask = Image.open(os.path.join(path, "mask", mask_name))

            # binary mask
            mask = np.array(mask)
            mask[mask < 240] = 0
            mask[mask >= 240] = 1

            # transform
            I1 = self.image_transform(I1)
            I2 = self.image_transform(I2)
            I_reference_video = self.image_transform(self.CenterPad(I_reference_video))
            I_reference_video_real = self.image_transform(self.CenterPad(I_reference_video_real))
            flow_forward = self.ToTensor(self.CenterCrop(flow_forward))
            flow_backward = self.ToTensor(self.CenterCrop(flow_backward))
            mask = self.ToTensor(self.CenterCrop(mask))

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
            print("problem in, ", path)
            print(e)
            return self.__getitem__(np.random.randint(0, len(self.image_pairs)))

        return outputs

    def __len__(self):
        return len(self.image_pairs)


import torchvision.utils as vutils
from torch.autograd import Variable


def batch_lab2rgb_transpose_mc(img_l_mc, img_ab_mc, nrow=8):
    if isinstance(img_l_mc, Variable):
        img_l_mc = img_l_mc.data.cpu()
    if isinstance(img_ab_mc, Variable):
        img_ab_mc = img_ab_mc.data.cpu()

    if img_l_mc.is_cuda:
        img_l_mc = img_l_mc.cpu()
    if img_ab_mc.is_cuda:
        img_ab_mc = img_ab_mc.cpu()

    assert img_l_mc.dim() == 4 and img_ab_mc.dim() == 4, "only for batch input"

    l_norm, ab_norm = 1.0, 1.0
    l_mean, ab_mean = 50.0, 0
    img_l = img_l_mc * l_norm + l_mean
    img_ab = img_ab_mc * ab_norm + ab_mean
    pred_lab = torch.cat((img_l, img_ab), dim=1)
    grid_lab = vutils.make_grid(pred_lab, nrow=nrow).numpy().astype("float64")
    grid_rgb = (np.clip(color.lab2rgb(grid_lab.transpose((1, 2, 0))), 0, 1) * 255).astype("uint8")
    return grid_rgb


# if __name__ == "__main__":
def test_videoloader():
    # dataset initI1lization
    # image size = 256
    train_dataset = VideosDataset(data_root="../../Dataset/video/video_pair", epoch=10)

    data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=0)

    iter_instance = iter(data_loader)
    sample = iter_instance.next()
    I1_lab, I2_lab, I_reference_lab, flow_forward, flow_backward, mask = sample

    # l: [-50,50], ab: [-128, 128]
    l_norm, ab_norm = 1.0, 1.0
    l_mean, ab_mean = 50.0, 0
    I1_l = I1_lab[:, 0:1, :, :]
    I1_ab = I1_lab[:, 1:3, :, :]
    I2_l = I2_lab[:, 0:1, :, :]
    I2_ab = I2_lab[:, 1:3, :, :]
    I_reference_l = I_reference_lab[:, 0:1, :, :]
    I_reference_ab = I_reference_lab[:, 1:3, :, :]

    # test image samples
    image_A_grid = batch_lab2rgb_transpose_mc(I1_l, I1_ab, nrow=4).astype(np.uint8)
    image_B_grid = batch_lab2rgb_transpose_mc(I2_l, I2_ab, nrow=4).astype(np.uint8)
    image_ref_grid = batch_lab2rgb_transpose_mc(I_reference_l, I_reference_ab, nrow=4).astype(np.uint8)
    plt.imsave("A.png", image_A_grid)
    plt.imsave("B.png", image_B_grid)
    plt.figure()
    plt.imshow(image_A_grid)
    plt.title("image A")
    plt.figure()
    plt.imshow(image_B_grid)
    plt.title("image B")
    plt.figure()
    plt.imshow(image_ref_grid)
    plt.title("image ref")
    # plt.show()

    # test warping
    print(I2_lab.shape)
    print(flow_forward.shape)
    warping_layer = WarpingLayer("gpu")
    I21_lab = warping_layer(I2_lab.cuda(), flow_forward.cuda())
    I21_lab = I21_lab * mask.cuda()

    I21_l = I21_lab[:, 0:1, :, :]
    I21_ab = I21_lab[:, 1:3, :, :]
    image_warp_grid = batch_lab2rgb_transpose_mc(I21_l, I21_ab, nrow=4).astype(np.uint8)
    plt.imsave("BA.png", image_warp_grid)
    plt.figure()
    plt.imshow(image_warp_grid)
    plt.title("image warp")
    # plt.show()

    # image difference
    plt.figure()
    plt.imshow(np.abs(image_warp_grid - image_A_grid).astype(np.uint8))
    plt.title("warp-A")
    plt.figure()
    plt.imshow(np.abs(image_B_grid - image_A_grid).astype(np.uint8))
    plt.title("B-A")
    # plt.show()
    plt.imsave("difference_warp.png", np.abs(image_warp_grid - image_A_grid).astype(np.uint8))
    plt.imsave("difference_BA.png", np.abs(image_B_grid - image_A_grid).astype(np.uint8))

    return None
