import os.path as osp

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image
from scipy.ndimage.filters import gaussian_filter
from utils.util_distortion import CenterPadCrop_numpy, Distortion_with_flow, RGB2Lab, ToTensor, random_mask

import lib.functional as F
from lib.RefPairDatasetWithGlobalOrigRGB_optimize import ToTensor, combo5_loader, image_loader, parse_images

cv2.setNumThreads(0)


class ImageNetDataset(data.Dataset):
    def __init__(
        self,
        data_root,
        epoch,
        image_size,
        with_bad=False,
        with_mid=False,
        transforms_imagenet=None,
        distortion_level=3,
        brightnessjitter=5,
        extra_reference_transform=None,
        real_reference_probability=1,
        mask_reference=False,
        no_reference_probability=0,
    ):
        image_pairs = []
        curr_image_pairs = parse_images(data_root, with_bad, with_mid)
        image_pairs += curr_image_pairs
        print("##### parsing image_a pairs in %s: %d pairs #####" % (data_root, len(curr_image_pairs)))
        if len(image_pairs) == 0:
            raise RuntimeError("Found 0 image_a pairs in all the data_roots")

        self.image_pairs = image_pairs
        self.epoch = epoch
        self.image_size = image_size
        self.real_len = len(self.image_pairs)
        self.image_pairs *= epoch

        self.real_reference_probability = real_reference_probability
        self.no_reference_probability = no_reference_probability
        self.mask_reference = mask_reference

        self.transforms_imagenet_raw = transforms_imagenet
        self.transforms_imagenet = transforms.Compose(transforms_imagenet)
        self.extra_reference_transform = transforms.Compose(extra_reference_transform)
        self.flow_transform = transforms.Compose([CenterPadCrop_numpy(self.image_size), ToTensor()])

        self.distortion_level = distortion_level
        self.distortion_transform = Distortion_with_flow()
        self.brightnessjitter = brightnessjitter

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
            if np.random.random() < self.no_reference_probability:
                no_reference_flag = 1
                I_reference_fake = Image.fromarray(np.array(I1) * 0)
                I_reference_real = I_reference_fake
            else:
                no_reference_flag = 0
                I_reference_fake = I1
                I_reference_real = image_loader(image_b_path)

            ## generate the flow
            height, width = np.array(I1).shape[0], np.array(I1).shape[1]
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

            # transform the input image for colorization
            I1 = self.transforms_imagenet(I1)

            # DEBUG
            # from utils.util import batch_lab2rgb_transpose_mc
            # plt.imshow(image_loader(image_a_path))
            # plt.figure()
            # I1 = I1.unsqueeze(0)
            # plt.imshow(batch_lab2rgb_transpose_mc(I1[0:32, 0:1, :, :], I1[0:32, 1:3, :, :]))

            # transform the fake reference
            I_reference_fake = self.extra_reference_transform(I_reference_fake)
            for transform in self.transforms_imagenet_raw:
                if type(transform) is RGB2Lab and self.extra_reference_transform is not None:
                    I_reference_fake = self.distortion_transform(I_reference_fake, forward_dx, forward_dy)
                I_reference_fake = transform(I_reference_fake)
            I_reference_fake[0:1, :, :] = I_reference_fake[0:1, :, :] + torch.randn(1) * self.brightnessjitter

            # transform the real reference
            I_reference_real = self.transforms_imagenet(I_reference_real)

            # generate the masked image
            if self.mask_reference:
                mask = torch.Tensor(random_mask(self.image_size[0], self.image_size[1]))
                I_reference_fake[1:2, :, :] = (
                    I_reference_fake[1:2, :, :] * (1 - mask) + I_reference_real[1:2, :, :] * mask
                )
                I_reference_fake[2:3, :, :] = (
                    I_reference_fake[2:3, :, :] * (1 - mask) + I_reference_real[2:3, :, :] * mask
                )

            # choose the reference
            if no_reference_flag:
                I_reference_output, fake_refernce_flag, no_reference_flag = (
                    I_reference_real,
                    torch.Tensor([0]),
                    torch.Tensor([1]).unsqueeze(-1).unsqueeze(-1),
                )
            else:
                no_reference_flag = torch.Tensor([0]).unsqueeze(-1).unsqueeze(-1)
                I_reference_output, fake_refernce_flag = (
                    (I_reference_real, torch.Tensor([0]))
                    if np.random.random() < self.real_reference_probability
                    else (I_reference_fake, torch.Tensor([1]))
                )

            outputs = []
            outputs.append(I1)
            outputs.append(I_reference_output)
            outputs.append(fake_refernce_flag)
            outputs.append(no_reference_flag)

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
