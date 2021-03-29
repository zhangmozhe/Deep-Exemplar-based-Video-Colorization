from __future__ import print_function

import argparse
import glob
import os
import time

import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torchvision.transforms as transform_lib
from PIL import Image
from tqdm import tqdm

import lib.TestTransforms as transforms
from models.ColorVidNet import ColorVidNet
from models.FrameColor import frame_colorization
from models.NonlocalNet import VGG19_pytorch, WarpNet
from utils.util import (batch_lab2rgb_transpose_mc, folder2vid, mkdir_if_not,
                        save_frames, tensor_lab2rgb, uncenter_l)
from utils.util_distortion import CenterPad, Normalize, RGB2Lab, ToTensor

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
torch.cuda.set_device(0)


def colorize_video(opt, input_path, reference_file, output_path, nonlocal_net, colornet, vggnet):
    # parameters for wls filter
    wls_filter_on = True
    lambda_value = 500
    sigma_color = 4

    # processing folders
    mkdir_if_not(output_path)
    files = glob.glob(output_path + "*")
    print("processing the folder:", input_path)
    path, dirs, filenames = os.walk(input_path).__next__()
    file_count = len(filenames)
    filenames.sort(key=lambda f: int("".join(filter(str.isdigit, f) or -1)))

    # NOTE: resize frames to 216*384
    transform = transforms.Compose(
        [CenterPad(opt.image_size), transform_lib.CenterCrop(opt.image_size), RGB2Lab(), ToTensor(), Normalize()]
    )

    # if frame propagation: use the first frame as reference
    # otherwise, use the specified reference image
    ref_name = input_path + filenames[0] if opt.frame_propagate else reference_file
    print("reference name:", ref_name)
    frame_ref = Image.open(ref_name)

    total_time = 0
    I_last_lab_predict = None
    
    for index, frame_name in enumerate(tqdm(filenames)):
        frame1 = Image.open(os.path.join(input_path, frame_name))
        IA_lab_large = transform(frame1).unsqueeze(0).cuda()
        IB_lab_large = transform(frame_ref).unsqueeze(0).cuda()
        IA_lab = torch.nn.functional.interpolate(IA_lab_large, scale_factor=0.5, mode="bilinear")
        IB_lab = torch.nn.functional.interpolate(IB_lab_large, scale_factor=0.5, mode="bilinear")

        IA_l = IA_lab[:, 0:1, :, :]
        IA_ab = IA_lab[:, 1:3, :, :]
        IB_l = IB_lab[:, 0:1, :, :]
        IB_ab = IB_lab[:, 1:3, :, :]
        if I_last_lab_predict is None:
            if opt.frame_propagate:
                I_last_lab_predict = IB_lab
            else:
                I_last_lab_predict = torch.zeros_like(IA_lab).cuda()

        # start the frame colorization
        with torch.no_grad():
            I_current_lab = IA_lab
            I_reference_lab = IB_lab
            I_reference_l = I_reference_lab[:, 0:1, :, :]
            I_reference_ab = I_reference_lab[:, 1:3, :, :]
            I_reference_rgb = tensor_lab2rgb(torch.cat((uncenter_l(I_reference_l), I_reference_ab), dim=1))

            # t_start = time.clock()
            # torch.cuda.synchronize()

            features_B = vggnet(I_reference_rgb, ["r12", "r22", "r32", "r42", "r52"], preprocess=True)
            I_current_ab_predict, I_current_nonlocal_lab_predict, features_current_gray = frame_colorization(
                I_current_lab,
                I_reference_lab,
                I_last_lab_predict,
                features_B,
                vggnet,
                nonlocal_net,
                colornet,
                feature_noise=0,
                temperature=1e-10,
            )
            I_last_lab_predict = torch.cat((IA_l, I_current_ab_predict), dim=1)

        # update timing
        # torch.cuda.synchronize()
        # delta_t = time.clock() - t_start
        # print("runtime:", delta_t)
        # if iter_num > 0:
        #     total_time += delta_t
        #     print("%.2g second" % (total_time / iter_num))

        # upsampling
        curr_bs_l = IA_lab_large[:, 0:1, :, :]
        curr_predict = (
            torch.nn.functional.interpolate(I_current_ab_predict.data.cpu(), scale_factor=2, mode="bilinear") * 1.25
        )

        # filtering
        if wls_filter_on:
            guide_image = uncenter_l(curr_bs_l) * 255 / 100
            wls_filter = cv2.ximgproc.createFastGlobalSmootherFilter(
                guide_image[0, 0, :, :].cpu().numpy().astype(np.uint8), lambda_value, sigma_color
            )
            curr_predict_a = wls_filter.filter(curr_predict[0, 0, :, :].cpu().numpy())
            curr_predict_b = wls_filter.filter(curr_predict[0, 1, :, :].cpu().numpy())
            curr_predict_a = torch.from_numpy(curr_predict_a).unsqueeze(0).unsqueeze(0)
            curr_predict_b = torch.from_numpy(curr_predict_b).unsqueeze(0).unsqueeze(0)
            curr_predict_filter = torch.cat((curr_predict_a, curr_predict_b), dim=1)
            IA_predict_rgb = batch_lab2rgb_transpose_mc(curr_bs_l[:32], curr_predict_filter[:32, ...])
        else:
            IA_predict_rgb = batch_lab2rgb_transpose_mc(curr_bs_l[:32], curr_predict[:32, ...])

        # save the frames
        save_frames(IA_predict_rgb, output_path, index)

    # output video
    video_name = "video.avi"
    folder2vid(image_folder=output_path, output_dir=output_path, filename=video_name)
    print()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--frame_propagate", default=False, type=bool, help="propagation mode, , please check the paper"
    )
    parser.add_argument("--image_size", type=int, default=[216 * 2, 384 * 2], help="the image size, eg. [216,384]")
    parser.add_argument("--cuda", action="store_false")
    parser.add_argument("--gpu_ids", type=str, default="0", help="separate by comma")
    parser.add_argument("--clip_path", type=str, default="./sample_videos/clips/v32", help="path of input clips")
    parser.add_argument("--ref_path", type=str, default="./sample_videos/ref/v32", help="path of refernce images")
    parser.add_argument("--output_path", type=str, default="./sample_videos/output", help="path of output clips")
    opt = parser.parse_args()
    opt.gpu_ids = [int(x) for x in opt.gpu_ids.split(",")]
    cudnn.benchmark = True
    print("running on GPU", opt.gpu_ids)

    clip_name = opt.clip_path.split("/")[-1]
    refs = os.listdir(opt.ref_path)
    refs.sort()

    nonlocal_net = WarpNet(1)
    colornet = ColorVidNet(7)
    vggnet = VGG19_pytorch()
    vggnet.load_state_dict(torch.load("data/vgg19_conv.pth"))
    for param in vggnet.parameters():
        param.requires_grad = False

    nonlocal_test_path = os.path.join("checkpoints/", "video_moredata_l1/nonlocal_net_iter_76000.pth")
    color_test_path = os.path.join("checkpoints/", "video_moredata_l1/colornet_iter_76000.pth")
    print("succesfully load nonlocal model: ", nonlocal_test_path)
    print("succesfully load color model: ", color_test_path)
    nonlocal_net.load_state_dict(torch.load(nonlocal_test_path))
    colornet.load_state_dict(torch.load(color_test_path))

    nonlocal_net.eval()
    colornet.eval()
    vggnet.eval()
    nonlocal_net.cuda()
    colornet.cuda()
    vggnet.cuda()

    for ref_name in refs:
        try:
            colorize_video(
                opt,
                opt.clip_path,
                os.path.join(opt.ref_path, ref_name),
                os.path.join(opt.output_path, clip_name + "_" + ref_name.split(".")[0]),
                nonlocal_net,
                colornet,
                vggnet,
            )
        except Exception as error:
            print("error when colorizing the video " + ref_name)
            print(error)

    video_name = "video.avi"
    clip_output_path = os.path.join(opt.output_path, clip_name)
    mkdir_if_not(clip_output_path)
    folder2vid(image_folder=opt.clip_path, output_dir=clip_output_path, filename=video_name)
