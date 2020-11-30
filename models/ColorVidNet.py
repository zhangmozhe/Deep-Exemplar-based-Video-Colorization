from os import path as osp

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel


def type_name(m):
    name = type(m).__name__
    if name == "TorchObject":
        name = m._typename.replace("cudnn.", "")
    return name


def model_tm2pm(model):
    layer_map = {}
    for m in model.modules:
        layer_map[m.name] = tm2pm(m)
    return layer_map


"""' 
removed layers: [[ 'ab_conv1_1', 'bw_conv1_1' ]]  
"""

ref_color_layernames = [
    "relu1_1",
    "conv1_2",
    "relu1_2",
    "conv1_2norm",
    "conv1_2norm_ss",
    "conv2_1",
    "relu2_1",
    "conv2_2",
    "relu2_2",
    "conv2_2norm",
    "conv2_2norm_ss",
    "conv3_1",
    "relu3_1",
    "conv3_2",
    "relu3_2",
    "conv3_3",
    "relu3_3",
    "conv3_3norm",
    "conv3_3norm_ss",
    "conv4_1",
    "relu4_1",
    "conv4_2",
    "relu4_2",
    "conv4_3",
    "relu4_3",
    "conv4_3norm",
    "conv5_1",
    "relu5_1",
    "conv5_2",
    "relu5_2",
    "conv5_3",
    "relu5_3",
    "conv5_3norm",
    "conv6_1",
    "relu6_1",
    "conv6_2",
    "relu6_2",
    "conv6_3",
    "relu6_3",
    "conv6_3norm",
    "conv7_1",
    "relu7_1",
    "conv7_2",
    "relu7_2",
    "conv7_3",
    "relu7_3",
    "conv7_3norm",
    "conv8_1",
    "conv3_3_short",
    "relu8_1_comb",
    "conv8_2",
    "relu8_2",
    "conv8_3",
    "relu8_3",
    "conv8_3norm",
    "conv9_1",
    "conv2_2_short",
    "relu9_1_comb",
    "conv9_2",
    "relu9_2",
    "conv9_2norm",
    "conv10_1",
    "conv1_2_short",
    "relu10_1_comb",
    "conv10_2",
    "relu10_2",
    "conv10_ab",
]


class ColorVidNet(nn.Module):
    def __init__(
        self, ic, ref_color_torch_model="convert_caffe_model/torch_model/reference_model/color_nodist_model.t7"
    ):
        super(ColorVidNet, self).__init__()
        """torch_model = load_lua(ref_color_torch_model, unknown_classes=True)
        if type(torch_model).__name__ == 'hashable_uniq_dict':
            torch_model = torch_model.model

        # fix the ugly typo error
        for m in torch_model.modules:
            if m.name == 'conv8_1_comb' and type_name(m) == 'ReLU':
                m.name = 'relu8_1_comb'

        pytorch_model = model_tm2pm(torch_model)

        self.add_module('conv1_1', nn.Sequential(nn.Conv2d(ic, 32, 3, 1, 1), nn.ReLU(), nn.Conv2d(32, 64, 3, 1, 1), nn.ReLU()))

        for l_name in ref_color_layernames:
            self.add_module(l_name, pytorch_model[l_name])

        ### replace all deconvolution with neareast upsample + convolution
        assert type(self.conv8_1).__name__=='ConvTranspose2d' and type(self.conv9_1).__name__=='ConvTranspose2d' and \
               type(self.conv10_1).__name__=='ConvTranspose2d', 'conv8_1, conv9_1, conv10_1 should be all ConvTranspose2d'
        """

        ##### Written by Ak. #####
        self.conv1_1 = nn.Sequential(nn.Conv2d(ic, 32, 3, 1, 1), nn.ReLU(), nn.Conv2d(32, 64, 3, 1, 1))
        self.conv1_2 = nn.Conv2d(64, 64, 3, 1, 1)
        self.conv1_2norm = nn.BatchNorm2d(64, affine=False)
        self.conv1_2norm_ss = nn.Conv2d(64, 64, 1, 2, bias=False, groups=64)
        self.conv2_1 = nn.Conv2d(64, 128, 3, 1, 1)
        self.conv2_2 = nn.Conv2d(128, 128, 3, 1, 1)
        self.conv2_2norm = nn.BatchNorm2d(128, affine=False)
        self.conv2_2norm_ss = nn.Conv2d(128, 128, 1, 2, bias=False, groups=128)
        self.conv3_1 = nn.Conv2d(128, 256, 3, 1, 1)
        self.conv3_2 = nn.Conv2d(256, 256, 3, 1, 1)
        self.conv3_3 = nn.Conv2d(256, 256, 3, 1, 1)
        self.conv3_3norm = nn.BatchNorm2d(256, affine=False)
        self.conv3_3norm_ss = nn.Conv2d(256, 256, 1, 2, bias=False, groups=256)
        self.conv4_1 = nn.Conv2d(256, 512, 3, 1, 1)
        self.conv4_2 = nn.Conv2d(512, 512, 3, 1, 1)
        self.conv4_3 = nn.Conv2d(512, 512, 3, 1, 1)
        self.conv4_3norm = nn.BatchNorm2d(512, affine=False)
        self.conv5_1 = nn.Conv2d(512, 512, 3, 1, 2, 2)
        self.conv5_2 = nn.Conv2d(512, 512, 3, 1, 2, 2)
        self.conv5_3 = nn.Conv2d(512, 512, 3, 1, 2, 2)
        self.conv5_3norm = nn.BatchNorm2d(512, affine=False)
        self.conv6_1 = nn.Conv2d(512, 512, 3, 1, 2, 2)
        self.conv6_2 = nn.Conv2d(512, 512, 3, 1, 2, 2)
        self.conv6_3 = nn.Conv2d(512, 512, 3, 1, 2, 2)
        self.conv6_3norm = nn.BatchNorm2d(512, affine=False)
        self.conv7_1 = nn.Conv2d(512, 512, 3, 1, 1)
        self.conv7_2 = nn.Conv2d(512, 512, 3, 1, 1)
        self.conv7_3 = nn.Conv2d(512, 512, 3, 1, 1)
        self.conv7_3norm = nn.BatchNorm2d(512, affine=False)
        self.conv8_1 = nn.ConvTranspose2d(512, 256, 4, 2, 1)
        self.conv3_3_short = nn.Conv2d(256, 256, 3, 1, 1)
        self.conv8_2 = nn.Conv2d(256, 256, 3, 1, 1)
        self.conv8_3 = nn.Conv2d(256, 256, 3, 1, 1)
        self.conv8_3norm = nn.BatchNorm2d(256, affine=False)
        self.conv9_1 = nn.ConvTranspose2d(256, 128, 4, 2, 1)
        self.conv2_2_short = nn.Conv2d(128, 128, 3, 1, 1)
        self.conv9_2 = nn.Conv2d(128, 128, 3, 1, 1)
        self.conv9_2norm = nn.BatchNorm2d(128, affine=False)
        self.conv10_1 = nn.ConvTranspose2d(128, 128, 4, 2, 1)
        self.conv1_2_short = nn.Conv2d(64, 128, 3, 1, 1)
        self.conv10_2 = nn.Conv2d(128, 128, 3, 1, 1)
        self.conv10_ab = nn.Conv2d(128, 2, 1, 1)

        # add self.relux_x
        self.relu1_1 = nn.ReLU()
        self.relu1_2 = nn.ReLU()
        self.relu2_1 = nn.ReLU()
        self.relu2_2 = nn.ReLU()
        self.relu3_1 = nn.ReLU()
        self.relu3_2 = nn.ReLU()
        self.relu3_3 = nn.ReLU()
        self.relu4_1 = nn.ReLU()
        self.relu4_2 = nn.ReLU()
        self.relu4_3 = nn.ReLU()
        self.relu5_1 = nn.ReLU()
        self.relu5_2 = nn.ReLU()
        self.relu5_3 = nn.ReLU()
        self.relu6_1 = nn.ReLU()
        self.relu6_2 = nn.ReLU()
        self.relu6_3 = nn.ReLU()
        self.relu7_1 = nn.ReLU()
        self.relu7_2 = nn.ReLU()
        self.relu7_3 = nn.ReLU()
        self.relu8_1_comb = nn.ReLU()
        self.relu8_2 = nn.ReLU()
        self.relu8_3 = nn.ReLU()
        self.relu9_1_comb = nn.ReLU()
        self.relu9_2 = nn.ReLU()
        self.relu10_1_comb = nn.ReLU()
        # self.relu10_2 = nn.ReLU()
        self.relu10_2 = nn.LeakyReLU(0.2, True)

        ##### Above #####

        print("replace all deconv with [nearest + conv]")
        self.conv8_1 = nn.Sequential(nn.Upsample(scale_factor=2, mode="nearest"), nn.Conv2d(512, 256, 3, 1, 1))
        self.conv9_1 = nn.Sequential(nn.Upsample(scale_factor=2, mode="nearest"), nn.Conv2d(256, 128, 3, 1, 1))
        self.conv10_1 = nn.Sequential(nn.Upsample(scale_factor=2, mode="nearest"), nn.Conv2d(128, 128, 3, 1, 1))

        print("replace all batchnorm with instancenorm")
        self.conv1_2norm = nn.InstanceNorm2d(64)
        self.conv2_2norm = nn.InstanceNorm2d(128)
        self.conv3_3norm = nn.InstanceNorm2d(256)
        self.conv4_3norm = nn.InstanceNorm2d(512)
        self.conv5_3norm = nn.InstanceNorm2d(512)
        self.conv6_3norm = nn.InstanceNorm2d(512)
        self.conv7_3norm = nn.InstanceNorm2d(512)
        self.conv8_3norm = nn.InstanceNorm2d(256)
        self.conv9_2norm = nn.InstanceNorm2d(128)
        # print(self)

    def forward(self, x):
        """ x: gray image (1 channel), ab(2 channel), ab_err, ba_err"""
        conv1_1 = self.relu1_1(self.conv1_1(x))
        conv1_2 = self.relu1_2(self.conv1_2(conv1_1))
        conv1_2norm = self.conv1_2norm(conv1_2)
        conv1_2norm_ss = self.conv1_2norm_ss(conv1_2norm)
        conv2_1 = self.relu2_1(self.conv2_1(conv1_2norm_ss))
        conv2_2 = self.relu2_2(self.conv2_2(conv2_1))
        conv2_2norm = self.conv2_2norm(conv2_2)
        conv2_2norm_ss = self.conv2_2norm_ss(conv2_2norm)
        conv3_1 = self.relu3_1(self.conv3_1(conv2_2norm_ss))
        conv3_2 = self.relu3_2(self.conv3_2(conv3_1))
        conv3_3 = self.relu3_3(self.conv3_3(conv3_2))
        conv3_3norm = self.conv3_3norm(conv3_3)
        conv3_3norm_ss = self.conv3_3norm_ss(conv3_3norm)
        conv4_1 = self.relu4_1(self.conv4_1(conv3_3norm_ss))
        conv4_2 = self.relu4_2(self.conv4_2(conv4_1))
        conv4_3 = self.relu4_3(self.conv4_3(conv4_2))
        conv4_3norm = self.conv4_3norm(conv4_3)
        conv5_1 = self.relu5_1(self.conv5_1(conv4_3norm))
        conv5_2 = self.relu5_2(self.conv5_2(conv5_1))
        conv5_3 = self.relu5_3(self.conv5_3(conv5_2))
        conv5_3norm = self.conv5_3norm(conv5_3)
        conv6_1 = self.relu6_1(self.conv6_1(conv5_3norm))
        conv6_2 = self.relu6_2(self.conv6_2(conv6_1))
        conv6_3 = self.relu6_3(self.conv6_3(conv6_2))
        conv6_3norm = self.conv6_3norm(conv6_3)
        conv7_1 = self.relu7_1(self.conv7_1(conv6_3norm))
        conv7_2 = self.relu7_2(self.conv7_2(conv7_1))
        conv7_3 = self.relu7_3(self.conv7_3(conv7_2))
        conv7_3norm = self.conv7_3norm(conv7_3)
        conv8_1 = self.conv8_1(conv7_3norm)
        conv3_3_short = self.conv3_3_short(conv3_3norm)
        conv8_1_comb = self.relu8_1_comb(conv8_1 + conv3_3_short)
        conv8_2 = self.relu8_2(self.conv8_2(conv8_1_comb))
        conv8_3 = self.relu8_3(self.conv8_3(conv8_2))
        conv8_3norm = self.conv8_3norm(conv8_3)
        conv9_1 = self.conv9_1(conv8_3norm)
        conv2_2_short = self.conv2_2_short(conv2_2norm)
        conv9_1_comb = self.relu9_1_comb(conv9_1 + conv2_2_short)
        conv9_2 = self.relu9_2(self.conv9_2(conv9_1_comb))
        conv9_2norm = self.conv9_2norm(conv9_2)
        conv10_1 = self.conv10_1(conv9_2norm)
        conv1_2_short = self.conv1_2_short(conv1_2norm)
        conv10_1_comb = self.relu10_1_comb(conv10_1 + conv1_2_short)
        conv10_2 = self.relu10_2(self.conv10_2(conv10_1_comb))
        conv10_ab = self.conv10_ab(conv10_2)
        pred_ab = torch.tanh(conv10_ab) * 128

        return pred_ab
