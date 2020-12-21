from __future__ import print_function

import argparse
import math
import os
import queue
import time

import cv2
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transform_lib
from torch.utils.data import ConcatDataset, DataLoader, WeightedRandomSampler
from torchvision.transforms import CenterCrop

import lib.TrainTransforms as transforms
from lib.videoloader import VideosDataset
from lib.videoloader_imagenet import VideosDataset_ImageNet
from models.ColorVidNet import ColorVidNet
from models.ContextualLoss import ContextualLoss, ContextualLoss_forward
from models.FrameColor import frame_colorization
from models.GAN_models import Discriminator_x64
from models.NonlocalNet import (NonlocalWeightedAverage, VGG19_pytorch,
                                WarpNet, WeightedAverage,
                                WeightedAverage_color)
from tensorboardX import SummaryWriter
from utils.util import (batch_lab2rgb_transpose_mc, feature_normalize, l1_loss,
                        mkdir_if_not, mse_loss, parse, tensor_lab2rgb,
                        uncenter_l, weighted_l1_loss, weighted_mse_loss)
from utils.util_distortion import (CenterPad_threshold, Normalize, RGB2Lab,
                                   ToTensor)
from utils.util_tensorboard import TBImageRecorder, value_logger
from utils.warping import WarpingLayer

cv2.setNumThreads(0)

parser = argparse.ArgumentParser()
parser.add_argument("--data_root", default="/home/zhanbo/remote/video/video_pair3/", type=str)
parser.add_argument("--data_root_imagenet", default="/home/zhanbo/remote/imagenet_ref_pair/", type=str)
parser.add_argument("--gpu_ids", type=str, default="0,1,2,3", help="separate by comma")
parser.add_argument("--workers", type=int, default=8)
parser.add_argument("--batch_size", type=int, default=16)
parser.add_argument("--image_size", type=int, default=[216, 384])
parser.add_argument("--ic", type=int, default=7)
parser.add_argument("--epoch", type=int, default=40)

parser.add_argument("--resume_epoch", type=int, default=0)
parser.add_argument("--resume", type=bool, default=False)
parser.add_argument("--load_pretrained_model", type=bool, default=True)

parser.add_argument("--lr", type=float, default=0.00001)
parser.add_argument("--beta1", type=float, default=0.5)
parser.add_argument("--lr_step", type=int, default=100)
parser.add_argument("--lr_gamma", type=float, default=0.1)


parser.add_argument("--checkpoint_dir", type=str, default="checkpoints/default")
parser.add_argument("--tb_log_step", type=int, default=50)
parser.add_argument("--print_step", type=int, default=2)

parser.add_argument("--real_reference_probability", type=float, default=0.7)
parser.add_argument("--nonzero_placeholder_probability", type=float, default=0.0)
parser.add_argument("--with_bad", type=bool, default=True)
parser.add_argument("--with_mid", type=bool, default=True)

parser.add_argument("--domain_invariant", type=bool, default=False)
parser.add_argument("--weigth_l1", type=float, default=2.0)
parser.add_argument("--weight_contextual", type=float, default="0.2")
parser.add_argument("--weight_perceptual", type=float, default="0.001")
parser.add_argument("--weight_smoothness", type=float, default="5.0")
parser.add_argument("--weight_gan", type=float, default="0.2")
parser.add_argument("--weight_nonlocal_smoothness", type=float, default="0.0")
parser.add_argument("--weight_nonlocal_consistent", type=float, default="0.0")
parser.add_argument("--weight_consistent", type=float, default="0.02")
parser.add_argument("--luminance_noise", type=float, default="2.0")
parser.add_argument("--permute_data", type=bool, default=True)
parser.add_argument("--contextual_loss_direction", type=str, default="forward", help="forward or backward matching")


def image_logger_fn(
    I_last_lab,
    I_current_lab,
    I_reference_lab,
    I_last_lab_predict,
    I_current_lab_predict,
    I_last_nonlocal_lab,
    I_current_nonlocal_lab,
):
    I_last_image = batch_lab2rgb_transpose_mc(I_last_lab[0:32, 0:1, :, :], I_last_lab[0:32, 1:3, :, :])
    I_current_image = batch_lab2rgb_transpose_mc(I_current_lab[0:32, 0:1, :, :], I_current_lab[0:32, 1:3, :, :])
    I_reference_image = batch_lab2rgb_transpose_mc(I_reference_lab[0:32, 0:1, :, :], I_reference_lab[0:32, 1:3, :, :])
    I_last_image_predict = batch_lab2rgb_transpose_mc(
        I_last_lab_predict[0:32, 0:1, :, :], I_last_lab_predict[0:32, 1:3, :, :]
    )
    I_current_image_predict = batch_lab2rgb_transpose_mc(
        I_current_lab_predict[0:32, 0:1, :, :], I_current_lab_predict[0:32, 1:3, :, :]
    )
    I_last_nonlocal_image = batch_lab2rgb_transpose_mc(
        I_last_nonlocal_lab[0:32, 0:1, :, :], I_last_nonlocal_lab[0:32, 1:3, :, :]
    )
    I_current_nonlocal_image = batch_lab2rgb_transpose_mc(
        I_current_nonlocal_lab[0:32, 0:1, :, :], I_current_nonlocal_lab[0:32, 1:3, :, :]
    )

    img_info = {}
    img_info["1_I_last"] = I_last_image
    img_info["2_I_current"] = I_current_image
    img_info["3_I_reference"] = I_reference_image
    img_info["4_I_last_predict"] = I_last_image_predict
    img_info["5_I_curren_predict"] = I_current_image_predict
    img_info["6_I_last_nonlocal"] = I_last_nonlocal_image
    img_info["7_I_current_nonlocal"] = I_current_nonlocal_image

    return img_info


def training_logger():
    try:
        if total_iter % opt.print_step == 0:
            print("processing time:", elapsed)
            print(
                "Epoch %d, Step[%d/%d], lr: %f, total_loss: %.2f"
                % (
                    epoch,
                    ((iter + 1) % iter_num_per_epoch),
                    iter_num_per_epoch,
                    step_optim_scheduler_g.get_last_lr()[0],
                    total_loss.item(),
                )
            )

            value_logger(
                tb_writer,
                total_iter,
                loss_info={
                    "l1_loss": l1_loss.item(),
                    "feat_loss": feat_loss.item(),
                    "contextual_loss_total": contextual_loss_total.item(),
                    "smoothness_loss": smoothness_loss.item(),
                    "nonlocal_smoothness_loss": nonlocal_smoothness_loss.item(),
                    "nonlocal_consistent_loss": nonlocal_consistent_loss.item(),
                    "consistent_loss": consistent_loss.item(),
                    "generator_loss": generator_loss.item(),
                    "discriminator_loss": discriminator_loss.item(),
                    "total_loss": total_loss.item(),
                },
            )

        if total_iter % opt.tb_log_step == 0:
            I_last_nonlocal_lab = torch.cat(
                (I_last_lab[:, 0:1, :, :], I_last_nonlocal_lab_predict[:, 1:3, :, :]), dim=1
            )
            I_current_nonlocal_lab = torch.cat(
                (I_current_lab[:, 0:1, :, :], I_current_nonlocal_lab_predict[:, 1:3, :, :]), dim=1
            )
            I_last_lab_predict = torch.cat((I_last_l, I_last_ab_predict), dim=1)
            data_queue.put(
                (
                    (
                        I_last_lab.cpu(),
                        I_current_lab.cpu(),
                        I_reference_lab.cpu(),
                        I_last_lab_predict.cpu(),
                        I_current_lab_predict.cpu(),
                        I_last_nonlocal_lab.cpu(),
                        I_current_nonlocal_lab,
                    ),
                    total_iter,
                )
            )

        if total_iter % 2000 == 0:
            if len(opt.gpu_ids) > 1:
                torch.save(
                    nonlocal_net.module.state_dict(),
                    os.path.join(opt.checkpoint_dir, "nonlocal_net_iter_%d.pth") % total_iter,
                )
                torch.save(
                    colornet.module.state_dict(),
                    os.path.join(opt.checkpoint_dir, "colornet_iter_%d.pth") % total_iter,
                )
                torch.save(
                    discriminator.module.state_dict(),
                    os.path.join(opt.checkpoint_dir, "discriminator_iter_%d.pth") % total_iter,
                )
            else:
                torch.save(
                    nonlocal_net.state_dict(),
                    os.path.join(opt.checkpoint_dir, "nonlocal_net_iter_%d.pth") % total_iter,
                )
                torch.save(colornet.state_dict(), os.path.join(opt.checkpoint_dir, "colornet_iter_%d.pth") % total_iter)
                torch.save(
                    discriminator.state_dict(),
                    os.path.join(opt.checkpoint_dir, "discriminator_iter_%d.pth") % total_iter,
                )

        # save the state for resume
        if total_iter % 2000 == 0:
            print("saving the checkpoint")
            if len(opt.gpu_ids) > 1:
                state = {
                    "total_iter": total_iter + 1,
                    "epoch": epoch,
                    "colornet_state": colornet.module.state_dict(),
                    "nonlocal_net_state": nonlocal_net.module.state_dict(),
                    "discriminator_state": discriminator.module.state_dict(),
                    "optimizer_g": optimizer_g.state_dict(),
                    "optimizer_d": optimizer_d.state_dict(),
                    "optimizer_schedule_g": step_optim_scheduler_g.state_dict(),
                    "optimizer_schedule_d": step_optim_scheduler_g.state_dict(),
                }
            else:
                state = {
                    "total_iter": total_iter + 1,
                    "epoch": epoch,
                    "colornet_state": colornet.state_dict(),
                    "nonlocal_net_state": nonlocal_net.state_dict(),
                    "discriminator_state": discriminator.state_dict(),
                    "optimizer_g": optimizer_g.state_dict(),
                    "optimizer_d": optimizer_d.state_dict(),
                    "optimizer_schedule_g": step_optim_scheduler_g.state_dict(),
                    "optimizer_schedule_d": step_optim_scheduler_d.state_dict(),
                }
            torch.save(state, os.path.join(opt.checkpoint_dir, "learning_checkpoint.pth"))
    except Exception as e:
        print("Exception during output")
        print(e)


def gpu_setup():
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    cudnn.benchmark = True
    opt.gpu_ids = [int(x) for x in opt.gpu_ids.split(",")]
    torch.cuda.set_device(opt.gpu_ids[0])
    device = torch.device("cuda")
    print("running on GPU", opt.gpu_ids)
    return device


def load_data():
    print("initializing dataloader")
    transforms_video = [
        CenterCrop(opt.image_size),
        RGB2Lab(),
        ToTensor(),
        Normalize(),
    ]
    transforms_imagenet = [CenterPad_threshold(opt.image_size), RGB2Lab(), ToTensor(), Normalize()]
    extra_reference_transform = [
        transform_lib.RandomHorizontalFlip(0.5),
        transform_lib.RandomResizedCrop(480, (0.98, 1.0), ratio=(0.8, 1.2)),
    ]
    train_dataset_video = VideosDataset(
        data_root=opt.data_root,
        epoch=opt.epoch,
        image_size=opt.image_size,
        image_transform=transforms.Compose(transforms_video),
        real_reference_probability=opt.real_reference_probability,
        nonzero_placeholder_probability=opt.nonzero_placeholder_probability,
    )
    train_dataset_imagenet = VideosDataset_ImageNet(
        data_root=opt.data_root_imagenet,
        image_size=opt.image_size,
        epoch=opt.epoch,
        with_bad=opt.with_bad,
        with_mid=opt.with_mid,
        transforms_imagenet=transforms_imagenet,
        distortion_level=4,
        brightnessjitter=5,
        nonzero_placeholder_probability=opt.nonzero_placeholder_probability,
        extra_reference_transform=extra_reference_transform,
        real_reference_probability=opt.real_reference_probability,
    )

    video_training_length = len(train_dataset_video)
    imagenet_training_length = len(train_dataset_imagenet)
    dataset_training_length = train_dataset_video.real_len + train_dataset_imagenet.real_len
    dataset_combined = ConcatDataset([train_dataset_video, train_dataset_imagenet])
    sampler = WeightedRandomSampler(
        [1] * video_training_length + [1] * imagenet_training_length, dataset_training_length * opt.epoch
    )
    data_loader = DataLoader(
        dataset_combined,
        batch_size=opt.batch_size,
        shuffle=False,
        num_workers=opt.workers,
        pin_memory=True,
        drop_last=True,
        sampler=sampler,
    )
    return dataset_training_length, train_dataset_video, train_dataset_imagenet, data_loader


def define_loss():
    print("defining loss")
    ab_criterion = nn.SmoothL1Loss().to(device)
    nonlocal_criterion = nn.SmoothL1Loss().to(device)
    feat_l2_criterion = nn.MSELoss().to(device)
    feat_l1_criterion = nn.SmoothL1Loss().to(device)
    contextual_loss = ContextualLoss().to(device)
    contextual_forward_loss = ContextualLoss_forward().to(device)
    BCE_stable = nn.BCEWithLogitsLoss().to(device)
    return contextual_loss, contextual_forward_loss


def define_optimizer():
    print("defining optimizer")
    optimizer_g = optim.Adam(
        [{"params": nonlocal_net.parameters(), "lr": 1e-5}, {"params": colornet.parameters(), "lr": 2e-4}],
        betas=(0.5, 0.999),
        eps=1e-5,
        amsgrad=True,
    )
    optimizer_d = optim.Adam(
        filter(lambda p: p.requires_grad, discriminator.parameters()), lr=2 * 1e-4, betas=(0.5, 0.999)
    )
    return optimizer_g, optimizer_d


def resume_model():
    print("resuming the learning")
    checkpoint = torch.load(os.path.join(opt.checkpoint_dir, "learning_checkpoint.pth"))
    total_iter = checkpoint["total_iter"]
    epoch = checkpoint["epoch"]
    colornet.load_state_dict(checkpoint["colornet_state"])
    nonlocal_net.load_state_dict(checkpoint["nonlocal_net_state"])
    discriminator.load_state_dict(checkpoint["discriminator_state"])
    optimizer_g.load_state_dict(checkpoint["optimizer_g"])
    optimizer_d.load_state_dict(checkpoint["optimizer_d"])
    step_optim_scheduler_g.load_state_dict(checkpoint["optimizer_schedule_g"])
    step_optim_scheduler_d.load_state_dict(checkpoint["optimizer_schedule_d"])


def to_device(
    colornet,
    nonlocal_net,
    discriminator,
    vggnet,
    contextual_loss,
    contextual_forward_loss,
    weighted_layer_color,
    nonlocal_weighted_layer,
    warping_layer,
    instancenorm,
):
    print("moving models to device")
    colornet = torch.nn.DataParallel(colornet.to(device), device_ids=opt.gpu_ids)
    nonlocal_net = torch.nn.DataParallel(nonlocal_net.to(device), device_ids=opt.gpu_ids)
    discriminator = torch.nn.DataParallel(discriminator.to(device), device_ids=opt.gpu_ids)
    vggnet = torch.nn.DataParallel(vggnet.to(device), device_ids=opt.gpu_ids)
    contextual_loss = torch.nn.DataParallel(contextual_loss.to(device), device_ids=opt.gpu_ids)
    contextual_forward_loss = torch.nn.DataParallel(contextual_forward_loss.to(device), device_ids=opt.gpu_ids)
    weighted_layer_color = torch.nn.DataParallel(weighted_layer_color.to(device), device_ids=opt.gpu_ids)
    nonlocal_weighted_layer = torch.nn.DataParallel(nonlocal_weighted_layer.to(device), device_ids=opt.gpu_ids)
    warping_layer = torch.nn.DataParallel(warping_layer.to(device), device_ids=opt.gpu_ids)
    instancenorm = torch.nn.DataParallel(instancenorm.to(device), device_ids=opt.gpu_ids)

    return (
        vggnet,
        nonlocal_net,
        colornet,
        discriminator,
        instancenorm,
        contextual_loss,
        contextual_forward_loss,
        weighted_layer_color,
        nonlocal_weighted_layer,
        warping_layer,
    )


def loss_init():
    print("initializing losses")
    zero_loss = torch.Tensor([0]).to(device)
    (
        feat_loss,
        contextual_loss_total,
        smoothness_loss,
        nonlocal_smoothness_loss,
        consistent_loss,
        nonlocal_consistent_loss,
        generator_loss,
        discriminator_loss,
    ) = (zero_loss, zero_loss, zero_loss, zero_loss, zero_loss, zero_loss, zero_loss, zero_loss)

    return (
        feat_loss,
        contextual_loss_total,
        smoothness_loss,
        nonlocal_smoothness_loss,
        consistent_loss,
        nonlocal_consistent_loss,
        generator_loss,
        discriminator_loss,
    )


def video_colorization():
    # colorization for the last frame
    I_last_ab_predict, I_last_nonlocal_lab_predict, features_last_gray = frame_colorization(
        I_last_lab,
        I_reference_lab,
        placeholder_lab,
        features_B,
        vggnet,
        nonlocal_net,
        colornet,
        feature_noise=0,
        luminance_noise=opt.luminance_noise,
    )
    I_last_lab_predict = torch.cat((I_last_l, I_last_ab_predict), dim=1)

    # colorization for the current frame
    I_current_ab_predict, I_current_nonlocal_lab_predict, features_current_gray = frame_colorization(
        I_current_lab,
        I_reference_lab,
        I_last_lab_predict,
        features_B,
        vggnet,
        nonlocal_net,
        colornet,
        feature_noise=0,
        luminance_noise=opt.luminance_noise,
    )
    I_current_lab_predict = torch.cat((I_last_l, I_current_ab_predict), dim=1)
    return I_current_ab_predict, I_last_ab_predict, I_current_nonlocal_lab_predict, I_last_nonlocal_lab_predict


if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn", force=True)
    opt = parse(parser)
    opt.data_root = opt.data_root.split(",")[0]
    opt.data_root_imagenet = opt.data_root_imagenet.split(",")[0]
    mkdir_if_not(opt.checkpoint_dir)
    mkdir_if_not("./runs/")

    device = gpu_setup()
    dataset_training_length, train_dataset_video, train_dataset_imagenet, data_loader = load_data()
    tb_writer = SummaryWriter()
    data_queue = queue.Queue()
    tb_image_reorder = TBImageRecorder(tb_writer, image_logger_fn, data_queue)
    tb_image_reorder.start()

    # define network
    nonlocal_net = WarpNet(opt.batch_size)
    colornet = ColorVidNet(opt.ic)
    discriminator = Discriminator_x64(ndf=64)
    colornet.to(device)
    nonlocal_net.to(device)
    discriminator.to(device)

    weighted_layer = WeightedAverage()
    weighted_layer_color = WeightedAverage_color()
    nonlocal_weighted_layer = NonlocalWeightedAverage()
    instancenorm = nn.InstanceNorm2d(512, affine=False)
    warping_layer = WarpingLayer("gpu")

    vggnet = VGG19_pytorch()
    vggnet.load_state_dict(torch.load("data/vgg19_conv.pth"))
    vggnet.eval()
    for param in vggnet.parameters():
        param.requires_grad = False

    # load pre-trained model
    if opt.load_pretrained_model:
        nonlocal_pretain_path = os.path.join("checkpoints/video_moredata_l1/", "nonlocal_net_iter_76000.pth")
        nonlocal_net.load_state_dict(torch.load(nonlocal_pretain_path))
        color_test_path = "checkpoints/video_moredata_l1/" + "colornet_iter_76000.pth"
        colornet.load_state_dict(torch.load(color_test_path))

    # define loss function
    contextual_loss, contextual_forward_loss = define_loss()

    # define optimizer
    optimizer_g, optimizer_d = define_optimizer()
    step_optim_scheduler_g = optim.lr_scheduler.StepLR(optimizer_g, step_size=opt.lr_step, gamma=opt.lr_gamma)
    step_optim_scheduler_d = optim.lr_scheduler.StepLR(optimizer_d, step_size=opt.lr_step, gamma=opt.lr_gamma)

    # define others
    downsampling_by2 = nn.AvgPool2d(kernel_size=2).to(device)
    downsampling_by4 = nn.AvgPool2d(kernel_size=4).to(device)

    # dataset info
    iter_num_per_epoch = dataset_training_length // opt.batch_size
    total_iter = opt.resume_epoch * iter_num_per_epoch
    print(
        "train_dataset info,  real_len: %d, epoch_len: %d, iter_num_per_epoch: %d"
        % (dataset_training_length, len(train_dataset_video) + len(train_dataset_imagenet), iter_num_per_epoch)
    )

    if opt.resume:
        resume_model()

    # move to GPU processing
    device = opt.gpu_ids[0]
    (
        vggnet,
        nonlocal_net,
        colornet,
        discriminator,
        instancenorm,
        contextual_loss,
        contextual_forward_loss,
        weighted_layer_color,
        nonlocal_weighted_layer,
        warping_layer,
    ) = to_device(
        colornet,
        nonlocal_net,
        discriminator,
        vggnet,
        contextual_loss,
        contextual_forward_loss,
        weighted_layer_color,
        nonlocal_weighted_layer,
        warping_layer,
        instancenorm,
    )

    (
        feat_loss,
        contextual_loss_total,
        smoothness_loss,
        nonlocal_smoothness_loss,
        consistent_loss,
        nonlocal_consistent_loss,
        generator_loss,
        discriminator_loss,
    ) = loss_init()

    # %% Training
    print("start training")
    for iter, data in enumerate(data_loader):
        start_time = time.time()
        total_iter += 1
        epoch = math.ceil(total_iter / iter_num_per_epoch)

        ###### LOADING DATA SAMPLE ######
        (
            I_last_lab,
            I_current_lab,
            I_reference_lab,
            flow_forward,
            flow_backward,
            mask,
            placeholder_lab,
            self_ref_flag,
        ) = data
        I_last_lab = I_last_lab.to(device)
        I_current_lab = I_current_lab.to(device)
        I_reference_lab = I_reference_lab.to(device)
        flow_forward = flow_forward.to(device)
        flow_backward = flow_backward.to(device)
        mask = mask.to(device)
        placeholder_lab = placeholder_lab.to(device)
        self_ref_flag = self_ref_flag.to(device)

        I_last_l = I_last_lab[:, 0:1, :, :]
        I_last_ab = I_last_lab[:, 1:3, :, :]
        I_current_l = I_current_lab[:, 0:1, :, :]
        I_current_ab = I_current_lab[:, 1:3, :, :]
        I_reference_l = I_reference_lab[:, 0:1, :, :]
        I_reference_ab = I_reference_lab[:, 1:3, :, :]
        I_reference_rgb = tensor_lab2rgb(torch.cat((uncenter_l(I_reference_l), I_reference_ab), dim=1))
        features_B = vggnet(I_reference_rgb, ["r12", "r22", "r32", "r42", "r52"], preprocess=True)

        ###### COLORIZATION ######
        (
            I_current_ab_predict,
            I_last_ab_predict,
            I_current_nonlocal_lab_predict,
            I_last_nonlocal_lab_predict,
        ) = video_colorization()

        ###### UPDATE DISCRIMINATOR ######
        if opt.weight_gan > 0:
            optimizer_g.zero_grad()
            optimizer_d.zero_grad()

            fake_data_lab = torch.cat(
                (uncenter_l(I_current_l), I_current_ab_predict, uncenter_l(I_last_l), I_last_ab_predict), dim=1
            )
            real_data_lab = torch.cat((uncenter_l(I_current_l), I_current_ab, uncenter_l(I_last_l), I_last_ab), dim=1)

            if opt.permute_data:
                batch_index = torch.arange(-1, opt.batch_size - 1, dtype=torch.long)
                real_data_lab = real_data_lab[batch_index, ...]

            y_pred_fake, feature_pred_fake = discriminator(fake_data_lab.detach())
            y_pred_real, feature_pred_real = discriminator(real_data_lab.detach())

            y = torch.ones_like(y_pred_real)
            y2 = torch.zeros_like(y_pred_real)
            discriminator_loss = (
                torch.mean((y_pred_real - torch.mean(y_pred_fake) - y) ** 2)
                + torch.mean((y_pred_fake - torch.mean(y_pred_real) + y) ** 2)
            ) / 2
            discriminator_loss.backward()
            optimizer_d.step()

        ###### UPDATE GENERATOR ######
        optimizer_g.zero_grad()
        optimizer_d.zero_grad()

        # extract vgg features for both output and original image
        I_predict_rgb = tensor_lab2rgb(torch.cat((uncenter_l(I_current_l), I_current_ab_predict), dim=1))
        predict_relu1_1, predict_relu2_1, predict_relu3_1, predict_relu4_1, predict_relu5_1 = vggnet(
            I_predict_rgb, ["r12", "r22", "r32", "r42", "r52"], preprocess=True
        )

        I_current_rgb = tensor_lab2rgb(torch.cat((uncenter_l(I_current_l), I_current_ab), dim=1))
        A_relu1_1, A_relu2_1, A_relu3_1, A_relu4_1, A_relu5_1 = vggnet(
            I_current_rgb, ["r12", "r22", "r32", "r42", "r52"], preprocess=True
        )
        B_relu1_1, B_relu2_1, B_relu3_1, B_relu4_1, B_relu5_1 = features_B

        ###### LOSS COMPUTE ######
        # l1 loss
        if opt.weigth_l1 > 0:
            sample_weights = (self_ref_flag[:, 1:3, :, :]) / (sum(self_ref_flag[:, 0, 0, 0]) + 1e-5)
            l1_loss = weighted_l1_loss(I_current_ab_predict, I_current_ab, sample_weights) * opt.weigth_l1

        # generator loss
        if opt.weight_gan > 0:
            y_pred_fake, feature_pred_fake = discriminator(fake_data_lab)
            y_pred_real, feature_pred_real = discriminator(real_data_lab)
            generator_loss = (
                (
                    torch.mean((y_pred_real - torch.mean(y_pred_fake) + y) ** 2)
                    + torch.mean((y_pred_fake - torch.mean(y_pred_real) - y) ** 2)
                )
                / 2
                * opt.weight_gan
            )

        # feature loss
        if opt.domain_invariant:
            feat_loss = (
                mse_loss(instancenorm(predict_relu5_1), instancenorm(A_relu5_1.detach()))
                * opt.weight_perceptual
                * 1e5
                * 0.2
            )
        else:
            feat_loss = mse_loss(predict_relu5_1, A_relu5_1.detach()) * opt.weight_perceptual

        # contextual loss
        if opt.contextual_loss_direction == "backward":
            contextual_style5_1 = torch.mean(contextual_loss(predict_relu5_1, B_relu5_1.detach())) * 8
            contextual_style4_1 = torch.mean(contextual_loss(predict_relu4_1, B_relu4_1.detach())) * 4
            contextual_style3_1 = (
                torch.mean(contextual_loss(downsampling_by2(predict_relu3_1), downsampling_by2(B_relu3_1.detach()))) * 2
            )
        else:
            contextual_style5_1 = torch.mean(contextual_forward_loss(predict_relu5_1, B_relu5_1.detach())) * 8
            contextual_style4_1 = torch.mean(contextual_forward_loss(predict_relu4_1, B_relu4_1.detach())) * 4
            contextual_style3_1 = (
                torch.mean(
                    contextual_forward_loss(downsampling_by2(predict_relu3_1), downsampling_by2(B_relu3_1.detach()))
                )
                * 2
            )
        if opt.weight_contextual > 0:
            contextual_loss_total = (
                contextual_style5_1 + contextual_style4_1 + contextual_style3_1
            ) * opt.weight_contextual

        # smoothness loss
        if opt.weight_smoothness > 0:
            scale_factor = 1
            I_current_lab_predict = torch.cat((I_current_l, I_current_ab_predict), dim=1)
            IA_ab_weighed = weighted_layer_color(
                I_current_lab, I_current_lab_predict, patch_size=3, alpha=10, scale_factor=scale_factor
            )
            smoothness_loss = (
                mse_loss(nn.functional.interpolate(I_current_ab_predict, scale_factor=scale_factor), IA_ab_weighed)
                * opt.weight_smoothness
            )

        if opt.weight_nonlocal_smoothness > 0:
            scale_factor = 0.25
            alpha_nonlocal_smoothness = 0.5
            nonlocal_smooth_feature = feature_normalize(A_relu2_1)
            I_current_lab_predict = torch.cat((I_current_l, I_current_ab_predict), dim=1)
            I_current_ab_weighted_nonlocal = nonlocal_weighted_layer(
                I_current_lab_predict,
                nonlocal_smooth_feature.detach(),
                patch_size=3,
                alpha=alpha_nonlocal_smoothness,
                scale_factor=scale_factor,
            )
            nonlocal_smoothness_loss = (
                mse_loss(
                    nn.functional.interpolate(I_current_ab_predict, scale_factor=scale_factor),
                    I_current_ab_weighted_nonlocal,
                )
                * opt.weight_nonlocal_smoothness
            )

        if opt.weight_consistent:
            I_current_lab_predict_warp = warping_layer(I_current_lab_predict, flow_forward)
            I_current_ab_predict_warp = I_current_lab_predict_warp[:, 1:3, :, :]
            consistent_loss = (
                weighted_mse_loss(I_current_ab_predict_warp, I_last_ab_predict, mask) * opt.weight_consistent
            )

        if opt.weight_nonlocal_consistent:
            I_current_nonlocal_lab_predict_warp = warping_layer(I_current_nonlocal_lab_predict, flow_forward)
            nonlocal_consistent_loss = (
                weighted_mse_loss(
                    I_current_nonlocal_lab_predict_warp[:, 1:3, :, :], I_last_nonlocal_lab_predict[:, 1:3, :, :], mask
                )
                * opt.weight_nonlocal_consistent
            )

        # total loss
        total_loss = (
            l1_loss
            + feat_loss
            + contextual_loss_total
            + smoothness_loss
            + nonlocal_smoothness_loss
            + consistent_loss
            + nonlocal_consistent_loss
            + generator_loss
        )
        total_loss.backward()
        optimizer_g.step()
        end_time = time.time()
        elapsed = end_time - start_time
        training_logger()

        step_optim_scheduler_g.step()
        step_optim_scheduler_d.step()

    data_queue.put((None, None))
