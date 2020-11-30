import numbers
import os
import shutil
import sys
import time

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.utils as vutils
from skimage import color, io
from torch.autograd import Variable

cv2.setNumThreads(0)

# l: [-50,50]
# ab: [-128, 128]
l_norm, ab_norm = 1.0, 1.0
l_mean, ab_mean = 50.0, 0


###### utility ######
def to_np(x):
    return x.data.cpu().numpy()


def utf8_str(in_str):
    try:
        in_str = in_str.decode("UTF-8")
    except Exception:
        in_str = in_str.encode("UTF-8").decode("UTF-8")
    return in_str


class MovingAvg(object):
    def __init__(self, pool_size=100):
        from queue import Queue

        self.pool = Queue(maxsize=pool_size)
        self.sum = 0
        self.curr_pool_size = 0

    def set_curr_val(self, val):
        if not self.pool.full():
            self.curr_pool_size += 1
            self.pool.put_nowait(val)
        else:
            last_first_val = self.pool.get_nowait()
            self.pool.put_nowait(val)
            self.sum -= last_first_val

        self.sum += val
        return self.sum / self.curr_pool_size


###### image normalization ######
# normalization for l
def center_l(l):
    l_mc = (l - l_mean) / l_norm
    return l_mc


# denormalization for l
def uncenter_l(l):
    return l * l_norm + l_mean


# normalization for ab
def center_ab(ab):
    ab_mc = (ab - ab_mean) / ab_norm
    return ab_mc


# normalization for lab image
def center_lab_img(img_lab):
    img_lab_mc = (
        img_lab / np.array((l_norm, ab_norm, ab_norm))[:, np.newaxis, np.newaxis]
        - np.array((l_mean / l_norm, ab_mean / ab_norm, ab_mean / ab_norm))[:, np.newaxis, np.newaxis]
    )
    return img_lab_mc


###### color space transformation ######
def rgb2lab_transpose(img_rgb):
    img_lab = color.rgb2lab(img_rgb).transpose((2, 0, 1))
    return img_lab


def lab2rgb(img_l, img_ab):
    """ INPUTS
            img_l      XxXx1     [0,100]
            img_ab     XxXx2     [-100,100]
        OUTPUTS
            returned value is XxXx3 """
    pred_lab = np.concatenate((img_l, img_ab), axis=2).astype("float64")
    pred_rgb = color.lab2rgb(pred_lab)
    pred_rgb = (np.clip(pred_rgb, 0, 1) * 255).astype("uint8")
    return pred_rgb


def gray2rgb_batch(l):
    # gray image tensor to rgb image tensor
    l_uncenter = uncenter_l(l)
    l_uncenter = l_uncenter / (2 * l_mean)
    return torch.cat((l_uncenter, l_uncenter, l_uncenter), dim=1)


def lab2rgb_transpose(img_l, img_ab):
    """ INPUTS
            img_l      1xXxX     [0,100]
            img_ab     2xXxX     [-100,100]
        OUTPUTS
            returned value is XxXx3 """
    pred_lab = np.concatenate((img_l, img_ab), axis=0).transpose((1, 2, 0))
    pred_rgb = (np.clip(color.lab2rgb(pred_lab), 0, 1) * 255).astype("uint8")
    return pred_rgb


def lab2rgb_transpose_mc(img_l_mc, img_ab_mc):
    if isinstance(img_l_mc, Variable):
        img_l_mc = img_l_mc.data.cpu()
    if isinstance(img_ab_mc, Variable):
        img_ab_mc = img_ab_mc.data.cpu()

    if img_l_mc.is_cuda:
        img_l_mc = img_l_mc.cpu()
    if img_ab_mc.is_cuda:
        img_ab_mc = img_ab_mc.cpu()

    assert img_l_mc.dim() == 3 and img_ab_mc.dim() == 3, "only for batch input"

    img_l = img_l_mc * l_norm + l_mean
    img_ab = img_ab_mc * ab_norm + ab_mean
    pred_lab = torch.cat((img_l, img_ab), dim=0)
    grid_lab = pred_lab.numpy().astype("float64")
    grid_rgb = (np.clip(color.lab2rgb(grid_lab.transpose((1, 2, 0))), 0, 1) * 255).astype("uint8")
    return grid_rgb


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

    img_l = img_l_mc * l_norm + l_mean
    img_ab = img_ab_mc * ab_norm + ab_mean
    pred_lab = torch.cat((img_l, img_ab), dim=1)
    grid_lab = vutils.make_grid(pred_lab, nrow=nrow).numpy().astype("float64")
    grid_rgb = (np.clip(color.lab2rgb(grid_lab.transpose((1, 2, 0))), 0, 1) * 255).astype("uint8")
    return grid_rgb


###### loss functions ######
def feature_normalize(feature_in):
    feature_in_norm = torch.norm(feature_in, 2, 1, keepdim=True) + sys.float_info.epsilon
    feature_in_norm = torch.div(feature_in, feature_in_norm)
    return feature_in_norm


def statistics_matching(feature1, feature2):
    N, C, H, W = feature1.shape
    feature1 = feature1.view(N, C, -1)
    feature2 = feature2.view(N, C, -1)

    mean1 = feature1.mean(dim=-1)
    mean2 = feature2.mean(dim=-1)
    std1 = feature1.var(dim=-1).sqrt()
    std2 = feature2.var(dim=-1).sqrt()

    return mse_loss(mean1, mean2) + mse_loss(std1, std2)


def cosine_similarity(input, target):
    input_norm = torch.norm(input, 2, 1, keepdim=True) + sys.float_info.epsilon
    target_norm = torch.norm(target, 2, 1, keepdim=True) + sys.float_info.epsilon
    normalized_input = torch.div(input, input_norm)
    normalized_target = torch.div(target, target_norm)
    cos_similarity = torch.mul(normalized_input, normalized_target)
    return torch.sum(cos_similarity, dim=1, keepdim=True)


def mse_loss(input, target=0):
    return torch.mean((input - target) ** 2)


def l1_loss(input, target=0):
    return torch.mean(torch.abs(input - target))


def calc_ab_gradient(input_ab):
    x_grad = input_ab[:, :, :, 1:] - input_ab[:, :, :, :-1]
    y_grad = input_ab[:, :, 1:, :] - input_ab[:, :, :-1, :]
    return x_grad, y_grad


def calc_tv_loss(input):
    x_grad = input[:, :, :, 1:] - input[:, :, :, :-1]
    y_grad = input[:, :, 1:, :] - input[:, :, :-1, :]
    loss = torch.sum(x_grad ** 2) / x_grad.nelement() + torch.sum(y_grad ** 2) / y_grad.nelement()
    return loss


def calc_cosine_dist_loss(input, target):
    input_norm = torch.norm(input, 2, 1, keepdim=True) + sys.float_info.epsilon
    target_norm = torch.norm(target, 2, 1, keepdim=True) + sys.float_info.epsilon
    normalized_input = torch.div(input, input_norm)
    normalized_target = torch.div(target, target_norm)
    cos_dist = torch.mul(normalized_input, normalized_target)
    return torch.mean(1 - torch.sum(cos_dist, dim=1))


def weighted_mse_loss(input, target, weights):
    out = (input - target) ** 2
    out = out * weights.expand_as(out)
    loss = out.mean()
    return loss


def weighted_l1_loss(input, target, weights):
    out = torch.abs(input - target)
    out = out * weights.expand_as(out)
    loss = out.mean()
    return loss


def colorfulness(input_ab):
    """
    according to the paper: Measuring colourfulness in natural images
    input is batches of ab tensors in lab space
    """
    N, C, H, W = input_ab.shape
    a = input_ab[:, 0:1, :, :]
    b = input_ab[:, 1:2, :, :]

    a = a.view(N, -1)
    b = b.view(N, -1)

    sigma_a = torch.std(a, dim=-1)
    sigma_b = torch.std(b, dim=-1)

    mean_a = torch.mean(a, dim=-1)
    mean_b = torch.mean(b, dim=-1)

    colorfulness = torch.sqrt(sigma_a ** 2 + sigma_b ** 2) + 0.37 * torch.sqrt(mean_a ** 2 + mean_b ** 2)
    return colorfulness


###### video related #######
def save_frames(image, image_folder, index=None, image_name=None):
    if image is not None:
        image = np.clip(image, 0, 255).astype(np.uint8)
        if image_name:
            io.imsave(os.path.join(image_folder, image_name), image)
        else:
            io.imsave(os.path.join(image_folder, str(index).zfill(5) + ".jpg"), image)


def folder2vid(image_folder, output_dir, filename):
    images = [img for img in os.listdir(image_folder) if img.endswith(".jpg")]
    images.sort()
    print(images)
    frame = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, layers = frame.shape
    # video = cv2.VideoWriter(os.path.join(output_dir, filename), 0x00000021, 24, (width, height))
    print("writing to video " + os.path.join(output_dir, filename))
    video = cv2.VideoWriter(
        os.path.join(output_dir, filename), cv2.VideoWriter_fourcc("D", "I", "V", "X"), 24, (width, height)
    )
    # video = cv2.VideoWriter(os.path.join(output_dir, filename), cv2.VideoWriter_fourcc('H', '2', '6', '4'), 24, (width, height))
    for image in images:
        video.write(cv2.imread(os.path.join(image_folder, image)))

    # cv2.destroyAllWindows()
    video.release()

    # import imageio
    # frames = []
    # for image in images:
    #     frames.append(imageio.imread(os.path.join(image_folder, image)))
    # imageio.mimsave('movie.gif', frames)


###### file system ######
def get_size(start_path="."):
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(start_path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            total_size += os.path.getsize(fp)
    return total_size


def mkdir_if_not(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


def parse(parser, save=True):
    opt = parser.parse_args(args=[])
    args = vars(opt)

    from time import gmtime, strftime

    print("------------ Options -------------")
    for k, v in sorted(args.items()):
        print("%s: %s" % (str(k), str(v)))
    print("-------------- End ----------------")

    # save to the disk
    if save:
        file_name = os.path.join("opt.txt")
        with open(file_name, "wt") as opt_file:
            opt_file.write(os.path.basename(sys.argv[0]) + " " + strftime("%Y-%m-%d %H:%M:%S", gmtime()) + "\n")
            opt_file.write("------------ Options -------------\n")
            for k, v in sorted(args.items()):
                opt_file.write("%s: %s\n" % (str(k), str(v)))
            opt_file.write("-------------- End ----------------\n")
    return opt


###### interactive ######
def clean_tensorboard(directory):
    folder_list = os.walk(directory).__next__()[1]
    for folder in folder_list:
        folder = directory + folder
        if get_size(folder) < 10000000:
            print("delete the folder of " + folder)
            shutil.rmtree(folder)


def imshow(input_image, title=None, type_conversion=False):
    inp = input_image
    if type_conversion or type(input_image) is torch.Tensor:
        inp = input_image.numpy()
    else:
        inp = input_image
    fig = plt.figure()
    if inp.ndim == 2:
        fig = plt.imshow(inp, cmap="gray", clim=[0, 255])
    else:
        fig = plt.imshow(np.transpose(inp, [1, 2, 0]).astype(np.uint8))
    plt.axis("off")
    fig.axes.get_xaxis().set_visible(False)
    fig.axes.get_yaxis().set_visible(False)
    plt.title(title)


def imshow_lab(input_lab):
    plt.imshow((batch_lab2rgb_transpose_mc(input_lab[:32, 0:1, :, :], input_lab[:32, 1:3, :, :])).astype(np.uint8))


###### vgg preprocessing ######
def vgg_preprocess(tensor):
    # input is RGB tensor which ranges in [0,1]
    # output is BGR tensor which ranges in [0,255]
    tensor_bgr = torch.cat((tensor[:, 2:3, :, :], tensor[:, 1:2, :, :], tensor[:, 0:1, :, :]), dim=1)
    # tensor_bgr = tensor[:, [2, 1, 0], ...]
    tensor_bgr_ml = tensor_bgr - torch.Tensor([0.40760392, 0.45795686, 0.48501961]).type_as(tensor_bgr).view(1, 3, 1, 1)
    tensor_rst = tensor_bgr_ml * 255
    return tensor_rst


def torch_vgg_preprocess(tensor):
    # pytorch version normalization
    # note that both input and output are RGB tensors;
    # input and output ranges in [0,1]
    # normalize the tensor with mean and variance
    tensor_mc = tensor - torch.Tensor([0.485, 0.456, 0.406]).type_as(tensor).view(1, 3, 1, 1)
    tensor_mc_norm = tensor_mc / torch.Tensor([0.229, 0.224, 0.225]).type_as(tensor_mc).view(1, 3, 1, 1)
    return tensor_mc_norm


def network_gradient(net, gradient_on=True):
    if gradient_on:
        for param in net.parameters():
            param.requires_grad = True
    else:
        for param in net.parameters():
            param.requires_grad = False
    return net


# def load_gray_image(img_path):
#     img_path = utf8_str(img_path)
#     img_gray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
#     return img_gray

# def load_rgb_image(img_path):
#     img_path = utf8_str(img_path)
#     im = cv2.imread(img_path, cv2.IMREAD_COLOR)
#     if im.shape[2] == 1:
#         print('find gray images: ', img_path)
#         im = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
#     img_rgb = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
#     return img_rgb

# def resize_img_longside(img_rgb, long_size, interpolation):
#     h, w = img_rgb.shape[:2]
#     scalar = long_size / max(h, w)
#     resized_img_rgb = cv2.resize(img_rgb, (int(w * scalar), int(h * scalar)), interpolation=interpolation)
#     return resized_img_rgb

# def resize_img_shortside(img_rgb, short_size, interpolation):
#     h, w = img_rgb.shape[:2]
#     scalar = short_size / min(h, w)
#     resized_img_rgb = cv2.resize(img_rgb, (int(w * scalar), int(h * scalar)), interpolation=interpolation)
#     return resized_img_rgb

# def resize_img(img_rgb, dsize, interpolation):
#     if isinstance(dsize, numbers.Number):
#         dsize = (int(dsize), int(dsize))
#     resized_img_rgb = cv2.resize(img_rgb, dsize, interpolation=interpolation)
#     return resized_img_rgb
