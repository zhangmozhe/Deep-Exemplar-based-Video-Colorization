import glob
import os
import os.path as osp
import struct

import cv2
import numpy as np
import torch.utils.data as data
from PIL import Image

cv2.setNumThreads(0)
import pdb

import lib.functional as F
import matplotlib.pyplot as plt
import torch
import torchvision.transforms as transforms
from PIL import Image
from skimage import color

from utils.util import (
    MovingAvg,
    batch_lab2rgb_transpose_mc,
    calc_ab_gradient,
    calc_cosine_dist_loss,
    calc_tv_loss,
    clean_tensorboard,
    feature_normalize,
    imshow,
    imshow_lab,
    mkdir_if_not,
    to_np,
    uncenter_l,
    vgg_preprocess,
    weighted_l1_loss,
)


def combo5_loader_nonoptimize(path):
    f = open(path, "rb")

    # width, height
    d = f.read(4)
    im_sz = struct.unpack("i", d)
    h = im_sz[0]

    d = f.read(4)
    im_sz = struct.unpack("i", d)
    w = im_sz[0]

    # warp_ba_layer 4
    d = f.read(4)
    im_sz = struct.unpack("i", d)
    d = f.read(im_sz[0])
    file_bytes = np.asarray(bytearray(d), dtype=np.uint8)
    img_data_ndarray = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    img_data_ndarray = cv2.cvtColor(img_data_ndarray, cv2.COLOR_BGR2RGB)
    warp_ba = Image.fromarray(img_data_ndarray)
    # warp_ba.save('/home/v-mingmh/Documents/gray/test_input/warp_ba.PNG')

    # warp_aba_layer 4
    d = f.read(4)
    im_sz = struct.unpack("i", d)
    d = f.read(im_sz[0])
    file_bytes = np.asarray(bytearray(d), dtype=np.uint8)
    img_data_ndarray = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    img_data_ndarray = cv2.cvtColor(img_data_ndarray, cv2.COLOR_BGR2RGB)
    warp_aba = Image.fromarray(img_data_ndarray)

    # 5 layers: err_aba, err_ba, err_ab
    errs = []
    for l in range(5):
        d = f.read(4)
        im_sz = struct.unpack("i", d)
        d = f.read(im_sz[0])
        file_bytes = np.asarray(bytearray(d), dtype=np.uint8)
        img_data_ndarray = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
        err_ba = Image.fromarray(img_data_ndarray)

        # err_ba.save('/home/v-mingmh/Documents/gray/test_input/input_err_ba' + str(l) + '.PNG')

        d = f.read(4)
        im_sz = struct.unpack("i", d)
        d = f.read(im_sz[0])
        file_bytes = np.asarray(bytearray(d), dtype=np.uint8)
        img_data_ndarray = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
        err_ab = Image.fromarray(img_data_ndarray)

        # err_ab.save('/home/v-mingmh/Documents/gray/test_input/input_err_ab' + str(l) + '.PNG')

        errs.append([err_ba, err_ab])

    f.close()

    return errs, warp_ba, warp_aba


# combo_folder = '../../movie/seq1/combo_new/'
combo_folder = "../../movie/seq2/combo_new"
output_folder = "analogy_output2/"
path, dirs, filenames = os.walk(combo_folder).__next__()

for filename in filenames:
    errs, warp_ba, warp_aba = combo5_loader_nonoptimize(combo_folder + "/" + filename)
    warp_ba.save(output_folder + filename.split(".")[0] + ".jpg")
