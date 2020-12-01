import torch
from utils.util import *


def warp_color(IA_l, IB_lab, features_B, vggnet, nonlocal_net, colornet, feature_noise=0, temperature=0.01):
    IA_rgb_from_gray = gray2rgb_batch(IA_l)
    with torch.no_grad():
        A_relu1_1, A_relu2_1, A_relu3_1, A_relu4_1, A_relu5_1 = vggnet(
            IA_rgb_from_gray, ["r12", "r22", "r32", "r42", "r52"], preprocess=True
        )
        B_relu1_1, B_relu2_1, B_relu3_1, B_relu4_1, B_relu5_1 = features_B

    # NOTE: output the feature before normalization
    features_A = [A_relu1_1, A_relu2_1, A_relu3_1, A_relu4_1, A_relu5_1]

    A_relu2_1 = feature_normalize(A_relu2_1)
    A_relu3_1 = feature_normalize(A_relu3_1)
    A_relu4_1 = feature_normalize(A_relu4_1)
    A_relu5_1 = feature_normalize(A_relu5_1)
    B_relu2_1 = feature_normalize(B_relu2_1)
    B_relu3_1 = feature_normalize(B_relu3_1)
    B_relu4_1 = feature_normalize(B_relu4_1)
    B_relu5_1 = feature_normalize(B_relu5_1)

    nonlocal_BA_lab, similarity_map = nonlocal_net(
        IB_lab,
        A_relu2_1,
        A_relu3_1,
        A_relu4_1,
        A_relu5_1,
        B_relu2_1,
        B_relu3_1,
        B_relu4_1,
        B_relu5_1,
        temperature=temperature,
    )

    return nonlocal_BA_lab, similarity_map, features_A


def frame_colorization(
    IA_lab,
    IB_lab,
    IA_last_lab,
    features_B,
    vggnet,
    nonlocal_net,
    colornet,
    joint_training=True,
    feature_noise=0,
    luminance_noise=0,
    temperature=0.01,
):

    IA_l = IA_lab[:, 0:1, :, :]
    if luminance_noise:
        IA_l = IA_l + torch.randn_like(IA_l, requires_grad=False) * luminance_noise

    with torch.autograd.set_grad_enabled(joint_training):
        nonlocal_BA_lab, similarity_map, features_A_gray = warp_color(
            IA_l, IB_lab, features_B, vggnet, nonlocal_net, colornet, feature_noise, temperature=temperature
        )
        nonlocal_BA_ab = nonlocal_BA_lab[:, 1:3, :, :]
        color_input = torch.cat((IA_l, nonlocal_BA_ab, similarity_map, IA_last_lab), dim=1)
        IA_ab_predict = colornet(color_input)

    return IA_ab_predict, nonlocal_BA_lab, features_A_gray
