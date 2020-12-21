import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models as torch_models
from utils.util import uncenter_l, vgg_preprocess

from models.vgg19_gray import vgg19_gray, vgg19_gray_new


def find_local_patch(x, patch_size):
    N, C, H, W = x.shape
    x_unfold = F.unfold(
        x, kernel_size=(patch_size, patch_size), padding=(patch_size // 2, patch_size // 2), stride=(1, 1)
    )

    return x_unfold.view(N, x_unfold.shape[1], H, W)


class WeightedAverage(nn.Module):
    def __init__(
        self,
    ):
        super(WeightedAverage, self).__init__()

    def forward(self, x_lab, patch_size=3, alpha=1, scale_factor=1):
        # alpha=0: less smooth; alpha=inf: smoother
        x_lab = F.interpolate(x_lab, scale_factor=scale_factor)
        l = x_lab[:, 0:1, :, :]
        a = x_lab[:, 1:2, :, :]
        b = x_lab[:, 2:3, :, :]
        local_l = find_local_patch(l, patch_size)
        local_a = find_local_patch(a, patch_size)
        local_b = find_local_patch(b, patch_size)
        local_difference_l = (local_l - l) ** 2
        correlation = nn.functional.softmax(-1 * local_difference_l / alpha, dim=1)

        return torch.cat(
            (
                torch.sum(correlation * local_a, dim=1, keepdim=True),
                torch.sum(correlation * local_b, dim=1, keepdim=True),
            ),
            1,
        )


class WeightedAverage_color(nn.Module):
    """
    smooth the image according to the color distance in the LAB space
    """

    def __init__(
        self,
    ):
        super(WeightedAverage_color, self).__init__()

    def forward(self, x_lab, x_lab_predict, patch_size=3, alpha=1, scale_factor=1):
        """ alpha=0: less smooth; alpha=inf: smoother """
        x_lab = F.interpolate(x_lab, scale_factor=scale_factor)
        l = uncenter_l(x_lab[:, 0:1, :, :])
        a = x_lab[:, 1:2, :, :]
        b = x_lab[:, 2:3, :, :]
        a_predict = x_lab_predict[:, 1:2, :, :]
        b_predict = x_lab_predict[:, 2:3, :, :]
        local_l = find_local_patch(l, patch_size)
        local_a = find_local_patch(a, patch_size)
        local_b = find_local_patch(b, patch_size)
        local_a_predict = find_local_patch(a_predict, patch_size)
        local_b_predict = find_local_patch(b_predict, patch_size)

        local_color_difference = (local_l - l) ** 2 + (local_a - a) ** 2 + (local_b - b) ** 2
        correlation = nn.functional.softmax(
            -1 * local_color_difference / alpha, dim=1
        )  # so that sum of weights equal to 1

        return torch.cat(
            (
                torch.sum(correlation * local_a_predict, dim=1, keepdim=True),
                torch.sum(correlation * local_b_predict, dim=1, keepdim=True),
            ),
            1,
        )


class NonlocalWeightedAverage(nn.Module):
    def __init__(
        self,
    ):
        super(NonlocalWeightedAverage, self).__init__()

    def forward(self, x_lab, feature, patch_size=3, alpha=0.1, scale_factor=1):
        # alpha=0: less smooth; alpha=inf: smoother
        # input feature is normalized feature
        x_lab = F.interpolate(x_lab, scale_factor=scale_factor)
        batch_size, channel, height, width = x_lab.shape
        feature = F.interpolate(feature, size=(height, width))
        batch_size = x_lab.shape[0]
        x_ab = x_lab[:, 1:3, :, :].view(batch_size, 2, -1)
        x_ab = x_ab.permute(0, 2, 1)

        local_feature = find_local_patch(feature, patch_size)
        local_feature = local_feature.view(batch_size, local_feature.shape[1], -1)

        correlation_matrix = torch.matmul(local_feature.permute(0, 2, 1), local_feature)
        correlation_matrix = nn.functional.softmax(correlation_matrix / alpha, dim=-1)

        weighted_ab = torch.matmul(correlation_matrix, x_ab)
        weighted_ab = weighted_ab.permute(0, 2, 1).contiguous()
        weighted_ab = weighted_ab.view(batch_size, 2, height, width)
        return weighted_ab


class CorrelationLayer(nn.Module):
    def __init__(self, search_range):
        super(CorrelationLayer, self).__init__()
        self.search_range = search_range

    def forward(self, x1, x2, alpha=1, raw_output=False, metric="similarity"):
        shape = list(x1.size())
        shape[1] = (self.search_range * 2 + 1) ** 2
        cv = torch.zeros(shape).to(torch.device("cuda"))

        for i in range(-self.search_range, self.search_range + 1):
            for j in range(-self.search_range, self.search_range + 1):
                if i < 0:
                    slice_h, slice_h_r = slice(None, i), slice(-i, None)
                elif i > 0:
                    slice_h, slice_h_r = slice(i, None), slice(None, -i)
                else:
                    slice_h, slice_h_r = slice(None), slice(None)

                if j < 0:
                    slice_w, slice_w_r = slice(None, j), slice(-j, None)
                elif j > 0:
                    slice_w, slice_w_r = slice(j, None), slice(None, -j)
                else:
                    slice_w, slice_w_r = slice(None), slice(None)

                if metric == "similarity":
                    cv[:, (self.search_range * 2 + 1) * i + j, slice_h, slice_w] = (
                        x1[:, :, slice_h, slice_w] * x2[:, :, slice_h_r, slice_w_r]
                    ).sum(1)
                else:  # patchwise subtraction
                    cv[:, (self.search_range * 2 + 1) * i + j, slice_h, slice_w] = -(
                        (x1[:, :, slice_h, slice_w] - x2[:, :, slice_h_r, slice_w_r]) ** 2
                    ).sum(1)

        # TODO sigmoid?
        if raw_output:
            return cv
        else:
            return nn.functional.softmax(cv / alpha, dim=1)


class Self_Attn(nn.Module):
    """ Self attention Layer"""

    def __init__(self, in_dim, activation):
        super(Self_Attn, self).__init__()
        self.chanel_in = in_dim
        self.activation = activation

        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 4, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 4, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
        inputs :
            x : input feature maps(B X C X W X H)
        returns :
            out : self attention value + input feature
            attention: B X N X N (N is Width*Height)
        """
        m_batchsize, C, width, height = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width * height).permute(0, 2, 1)  # B X N X C
        proj_key = self.key_conv(x).view(m_batchsize, -1, width * height)  # B X C x N
        energy = torch.bmm(proj_query, proj_key)  # transpose check
        attention = self.softmax(energy)  # B X (N) X (N)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width * height)  # B X C X N

        out = torch.bmm(proj_value, attention.permute(0, 1, 2))
        out = out.view(m_batchsize, C, width, height)

        out = self.gamma * out + x
        return out


class VGG19_pytorch(nn.Module):
    """
    NOTE: no need to pre-process the input; input tensor should range in [0,1]
    """

    def __init__(self, pool="max"):
        super(VGG19_pytorch, self).__init__()
        self.conv1_1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv3_3 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv3_4 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv4_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv4_4 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_1 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_4 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        if pool == "max":
            self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
            self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
            self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
            self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
            self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2)
        elif pool == "avg":
            self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)
            self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)
            self.pool3 = nn.AvgPool2d(kernel_size=2, stride=2)
            self.pool4 = nn.AvgPool2d(kernel_size=2, stride=2)
            self.pool5 = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x, out_keys, preprocess=True):
        """
        NOTE: input tensor should range in [0,1]
        """
        out = {}
        if preprocess:
            x = vgg_preprocess(x)
        out["r11"] = F.relu(self.conv1_1(x))
        out["r12"] = F.relu(self.conv1_2(out["r11"]))
        out["p1"] = self.pool1(out["r12"])
        out["r21"] = F.relu(self.conv2_1(out["p1"]))
        out["r22"] = F.relu(self.conv2_2(out["r21"]))
        out["p2"] = self.pool2(out["r22"])
        out["r31"] = F.relu(self.conv3_1(out["p2"]))
        out["r32"] = F.relu(self.conv3_2(out["r31"]))
        out["r33"] = F.relu(self.conv3_3(out["r32"]))
        out["r34"] = F.relu(self.conv3_4(out["r33"]))
        out["p3"] = self.pool3(out["r34"])
        out["r41"] = F.relu(self.conv4_1(out["p3"]))
        out["r42"] = F.relu(self.conv4_2(out["r41"]))
        out["r43"] = F.relu(self.conv4_3(out["r42"]))
        out["r44"] = F.relu(self.conv4_4(out["r43"]))
        out["p4"] = self.pool4(out["r44"])
        out["r51"] = F.relu(self.conv5_1(out["p4"]))
        out["r52"] = F.relu(self.conv5_2(out["r51"]))
        out["r53"] = F.relu(self.conv5_3(out["r52"]))
        out["r54"] = F.relu(self.conv5_4(out["r53"]))
        out["p5"] = self.pool5(out["r54"])
        return [out[key] for key in out_keys]


class VGG19_feature_color(nn.Module):
    def __init__(self):
        super(VGG19_feature_color, self).__init__()
        # self.select = ['0', '5', '10', '19', '28']  # Select conv1_1 ~ conv5_1 activation maps.
        self.select = ["1", "6", "11", "20", "29"]  # Select relu1_1 ~ relu5_1 activation maps.
        self.vgg = torch_models.vgg19(pretrained=True).features

    def forward(self, x):
        """Extract multiple convolutional feature maps."""
        features = []
        for name, layer in self.vgg._modules.items():
            if name in self.select:
                x = layer(x)
                features.append(x)
        return features


class VGG19_feature(nn.Module):
    # input: [LLL] channels, range=[0,255]
    def __init__(self, gpu_ids):
        super(VGG19_feature, self).__init__()
        self.vgg19_gray = vgg19_gray().cuda()

    def forward(self, A_l, B_l):
        A_relu3_1, A_relu4_1, A_relu5_1 = self.vgg19_gray(A_l)
        B_relu3_1, B_relu4_1, B_relu5_1 = self.vgg19_gray(B_l)
        return A_relu3_1, A_relu4_1, A_relu5_1, B_relu3_1, B_relu4_1, B_relu5_1


class WTA_scale(torch.autograd.Function):
    """
    We can implement our own custom autograd Functions by subclassing
    torch.autograd.Function and implementing the forward and backward passes
    which operate on Tensors.
    """

    @staticmethod
    def forward(ctx, input, scale=1e-4):
        """
        In the forward pass we receive a Tensor containing the input and return a
        Tensor containing the output. You can cache arbitrary Tensors for use in the
        backward pass using the save_for_backward method.
        """
        activation_max, index_max = torch.max(input, -1, keepdim=True)
        input_scale = input * scale  # default: 1e-4
        # input_scale = input * scale  # default: 1e-4
        output_max_scale = torch.where(input == activation_max, input, input_scale)

        mask = (input == activation_max).type(torch.float)
        ctx.save_for_backward(input, mask)
        return output_max_scale

    @staticmethod
    def backward(ctx, grad_output):
        """
        In the backward pass we receive a Tensor containing the gradient of the loss
        with respect to the output, and we need to compute the gradient of the loss
        with respect to the input.
        """
        # import pdb
        # pdb.set_trace()
        input, mask = ctx.saved_tensors
        mask_ones = torch.ones_like(mask)
        mask_small_ones = torch.ones_like(mask) * 1e-4
        # mask_small_ones = torch.ones_like(mask) * 1e-4

        grad_scale = torch.where(mask == 1, mask_ones, mask_small_ones)
        grad_input = grad_output.clone() * grad_scale
        return grad_input, None


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, stride=1):
        super(ResidualBlock, self).__init__()
        self.padding1 = nn.ReflectionPad2d(padding)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=0, stride=stride)
        self.bn1 = nn.InstanceNorm2d(out_channels)
        self.prelu = nn.PReLU()
        self.padding2 = nn.ReflectionPad2d(padding)
        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=0, stride=stride)
        self.bn2 = nn.InstanceNorm2d(out_channels)

    def forward(self, x):
        residual = x
        out = self.padding1(x)
        out = self.conv1(out)
        out = self.bn1(out)
        out = self.prelu(out)
        out = self.padding2(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        out = self.prelu(out)
        return out


class WarpNet(nn.Module):
    """ input is Al, Bl, channel = 1, range~[0,255] """

    def __init__(self, batch_size):
        super(WarpNet, self).__init__()
        self.feature_channel = 64
        self.in_channels = self.feature_channel * 4
        self.inter_channels = 256
        # 44*44
        self.layer2_1 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(128, 128, kernel_size=3, padding=0, stride=1),
            nn.InstanceNorm2d(128),
            nn.PReLU(),
            nn.ReflectionPad2d(1),
            nn.Conv2d(128, self.feature_channel, kernel_size=3, padding=0, stride=2),
            nn.InstanceNorm2d(self.feature_channel),
            nn.PReLU(),
        )
        self.layer3_1 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(256, 128, kernel_size=3, padding=0, stride=1),
            nn.InstanceNorm2d(128),
            nn.PReLU(),
            nn.ReflectionPad2d(1),
            nn.Conv2d(128, self.feature_channel, kernel_size=3, padding=0, stride=1),
            nn.InstanceNorm2d(self.feature_channel),
            nn.PReLU(),
        )

        # 22*22->44*44
        self.layer4_1 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(512, 256, kernel_size=3, padding=0, stride=1),
            nn.InstanceNorm2d(256),
            nn.PReLU(),
            nn.ReflectionPad2d(1),
            nn.Conv2d(256, self.feature_channel, kernel_size=3, padding=0, stride=1),
            nn.InstanceNorm2d(self.feature_channel),
            nn.PReLU(),
            nn.Upsample(scale_factor=2),
        )

        # 11*11->44*44
        self.layer5_1 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(512, 256, kernel_size=3, padding=0, stride=1),
            nn.InstanceNorm2d(256),
            nn.PReLU(),
            nn.Upsample(scale_factor=2),
            nn.ReflectionPad2d(1),
            nn.Conv2d(256, self.feature_channel, kernel_size=3, padding=0, stride=1),
            nn.InstanceNorm2d(self.feature_channel),
            nn.PReLU(),
            nn.Upsample(scale_factor=2),
        )

        self.layer = nn.Sequential(
            ResidualBlock(self.feature_channel * 4, self.feature_channel * 4, kernel_size=3, padding=1, stride=1),
            ResidualBlock(self.feature_channel * 4, self.feature_channel * 4, kernel_size=3, padding=1, stride=1),
            ResidualBlock(self.feature_channel * 4, self.feature_channel * 4, kernel_size=3, padding=1, stride=1),
        )

        self.theta = nn.Conv2d(
            in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1, stride=1, padding=0
        )
        self.phi = nn.Conv2d(
            in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1, stride=1, padding=0
        )

        self.upsampling = nn.Upsample(scale_factor=4)

    def forward(
        self,
        B_lab_map,
        A_relu2_1,
        A_relu3_1,
        A_relu4_1,
        A_relu5_1,
        B_relu2_1,
        B_relu3_1,
        B_relu4_1,
        B_relu5_1,
        temperature=0.001 * 5,
        detach_flag=False,
        WTA_scale_weight=1,
        feature_noise=0,
    ):
        batch_size = B_lab_map.shape[0]
        channel = B_lab_map.shape[1]
        image_height = B_lab_map.shape[2]
        image_width = B_lab_map.shape[3]
        feature_height = int(image_height / 4)
        feature_width = int(image_width / 4)

        # scale feature size to 44*44
        A_feature2_1 = self.layer2_1(A_relu2_1)
        B_feature2_1 = self.layer2_1(B_relu2_1)
        A_feature3_1 = self.layer3_1(A_relu3_1)
        B_feature3_1 = self.layer3_1(B_relu3_1)
        A_feature4_1 = self.layer4_1(A_relu4_1)
        B_feature4_1 = self.layer4_1(B_relu4_1)
        A_feature5_1 = self.layer5_1(A_relu5_1)
        B_feature5_1 = self.layer5_1(B_relu5_1)

        # concatenate features
        if A_feature5_1.shape[2] != A_feature2_1.shape[2] or A_feature5_1.shape[3] != A_feature2_1.shape[3]:
            A_feature5_1 = F.pad(A_feature5_1, (0, 0, 1, 1), "replicate")
            B_feature5_1 = F.pad(B_feature5_1, (0, 0, 1, 1), "replicate")
        A_features = self.layer(torch.cat((A_feature2_1, A_feature3_1, A_feature4_1, A_feature5_1), 1))
        B_features = self.layer(torch.cat((B_feature2_1, B_feature3_1, B_feature4_1, B_feature5_1), 1))

        # pairwise cosine similarity
        theta = self.theta(A_features).view(batch_size, self.inter_channels, -1)  # 2*256*(feature_height*feature_width)
        theta = theta - theta.mean(dim=-1, keepdim=True)  # center the feature
        theta_norm = torch.norm(theta, 2, 1, keepdim=True) + sys.float_info.epsilon
        theta = torch.div(theta, theta_norm)
        theta_permute = theta.permute(0, 2, 1)  # 2*(feature_height*feature_width)*256
        phi = self.phi(B_features).view(batch_size, self.inter_channels, -1)  # 2*256*(feature_height*feature_width)
        phi = phi - phi.mean(dim=-1, keepdim=True)  # center the feature
        phi_norm = torch.norm(phi, 2, 1, keepdim=True) + sys.float_info.epsilon
        phi = torch.div(phi, phi_norm)
        f = torch.matmul(theta_permute, phi)  # 2*(feature_height*feature_width)*(feature_height*feature_width)
        if detach_flag:
            f = f.detach()

        f_similarity = f.unsqueeze_(dim=1)
        similarity_map = torch.max(f_similarity, -1, keepdim=True)[0]
        similarity_map = similarity_map.view(batch_size, 1, feature_height, feature_width)

        # f can be negative
        f_WTA = f if WTA_scale_weight == 1 else WTA_scale.apply(f, WTA_scale_weight)
        f_WTA = f_WTA / temperature
        f_div_C = F.softmax(f_WTA.squeeze_(), dim=-1)  # 2*1936*1936;

        # downsample the reference color
        B_lab = F.avg_pool2d(B_lab_map, 4)
        B_lab = B_lab.view(batch_size, channel, -1)
        B_lab = B_lab.permute(0, 2, 1)  # 2*1936*channel

        # multiply the corr map with color
        y = torch.matmul(f_div_C, B_lab)  # 2*1936*channel
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, channel, feature_height, feature_width)  # 2*3*44*44
        y = self.upsampling(y)
        similarity_map = self.upsampling(similarity_map)

        return y, similarity_map
