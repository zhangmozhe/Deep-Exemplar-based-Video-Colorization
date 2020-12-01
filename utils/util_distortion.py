import math
import random

import cv2
import lib.functional as F
import numpy as np
from PIL import Image
from scipy.ndimage.filters import gaussian_filter
from skimage import color
from skimage.draw import random_shapes
from skimage.filters import gaussian
from skimage.transform import resize

cv2.setNumThreads(0)
from numba import jit, u1, u2


class RGB2Lab(object):
    def __init__(self):
        pass

    def __call__(self, inputs):
        return color.rgb2lab(inputs)


class Guassian_noise(object):
    """Elastic distortion"""

    def __init__(self, noise_sigma=0.1):
        self.noise_sigma = noise_sigma

    def __call__(self, inputs):
        h = inputs.shape[0]
        w = inputs.shape[1]
        noisy_image = inputs
        noise = np.random.randn(h, w) * self.noise_sigma
        noisy_image[:, :, 0] = inputs[:, :, 0] + noise

        return noisy_image


class Distortion(object):
    """Elastic distortion"""

    def __init__(self, distortion_level=3, flip_probability=0):
        self.alpha_max = distortion_level
        self.flip_probability = flip_probability

    def __call__(self, inputs):
        if np.random.rand() < self.flip_probability:
            inputs = inputs.transpose(Image.FLIP_LEFT_RIGHT)

        inputs = np.array(inputs)
        alpha = np.random.rand() * self.alpha_max
        sigma = 50
        random_state = np.random.RandomState(None)
        shape = inputs.shape[0], inputs.shape[1]

        dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha * 1000
        dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha * 1000

        x, y = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing="ij")
        remap_image = cv2.remap(
            inputs, (dy + y).astype(np.float32), (dx + x).astype(np.float32), interpolation=cv2.INTER_LINEAR
        )

        return Image.fromarray(remap_image)


class Distortion_with_flow(object):
    """Elastic distortion"""

    def __init__(self):
        return

    def __call__(self, inputs, dx, dy):
        inputs = np.array(inputs)
        shape = inputs.shape[0], inputs.shape[1]
        inputs = np.array(inputs)
        remap_image = forward_mapping(inputs, dy, dx, maxIter=3, precision=1e-3)

        return Image.fromarray(remap_image)


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
        return F.to_mytensor(inputs)


class RandomErasing(object):
    """
    Class that performs Random Erasing in Random Erasing Data Augmentation by Zhong et al.
    -------------------------------------------------------------------------------------
    probability: The probability that the operation will be performed.
    sl: min erasing area
    sh: max erasing area
    r1: min aspect ratio
    mean: erasing value
    -------------------------------------------------------------------------------------
    """

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


class CenteredPad(object):
    """
    pad the frame with black border,
    make square image for processing
    """

    def __init__(self, image_size):
        self.image_size = image_size

    def __call__(self, img):
        img = np.array(img)
        width = np.size(img, 1)
        height = np.size(img, 0)
        old_size = [height, width]

        ratio = float(self.image_size) / max(height, width)
        new_size = [int(x * ratio) for x in old_size]
        I_resize = resize(img, new_size, mode="reflect", preserve_range=True, clip=False, anti_aliasing=True)
        width_new = np.size(I_resize, 1)
        height_new = np.size(I_resize, 0)

        I_pad = np.zeros((self.image_size, self.image_size, 3))
        start_height = (self.image_size - new_size[0]) // 2
        start_width = (self.image_size - new_size[1]) // 2
        I_pad[start_height : (start_height + height_new), start_width : (start_width + width_new), :] = I_resize

        return Image.fromarray(I_pad.astype(np.uint8))


class centeredPad_with_height(object):
    """
    pad the image according to the height
    """

    def __init__(self, image_size):
        self.height = image_size[0]
        self.width = image_size[1]

    def __call__(self, image):
        # pad the image to 16:9
        # pad height
        I = np.array(image)

        # for padded input
        if I.shape[0] == I.shape[1] and I.shape[1] == self.width:
            return Image.fromarray(I.astype(np.uint8))

        width0 = np.size(I, 1)
        height0 = np.size(I, 0)
        old_size = [height0, width0]
        height = self.height
        width = self.width

        ratio = height / height0
        new_size = [int(x * ratio) for x in old_size]
        I_resize = resize(I, new_size, mode="reflect", preserve_range=True, clip=False, anti_aliasing=True)
        width_new = np.size(I_resize, 1)
        height_new = np.size(I_resize, 0)

        # if exceed the expected width
        if width_new > width:
            I_resize = I_resize[:, math.floor(width_new - width) // 2 : (math.floor(width_new - width) // 2 + width), :]
            width_new = np.size(I_resize, 1)
            height_new = np.size(I_resize, 0)

        # lines: 56~200
        I_pad = np.zeros((width, width, 3))
        start_height = (width - height_new) // 2
        start_width = (width - width_new) // 2
        I_pad[start_height : (start_height + height_new), start_width : (start_width + width_new), :] = I_resize

        return Image.fromarray(I_pad.astype(np.uint8))


class CenterPad(object):
    def __init__(self, image_size):
        self.height = image_size[0]
        self.width = image_size[1]

    def __call__(self, image):
        # pad the image to 16:9
        # pad height
        I = np.array(image)

        # for padded input
        height_old = np.size(I, 0)
        width_old = np.size(I, 1)
        old_size = [height_old, width_old]
        height = self.height
        width = self.width
        I_pad = np.zeros((height, width, np.size(I, 2)))

        ratio = height / width
        if height_old / width_old == ratio:
            if height_old == height:
                return Image.fromarray(I.astype(np.uint8))
            new_size = [int(x * height / height_old) for x in old_size]
            I_resize = resize(I, new_size, mode="reflect", preserve_range=True, clip=False, anti_aliasing=True)
            return Image.fromarray(I_resize.astype(np.uint8))

        if height_old / width_old > ratio:  # pad the width and crop
            new_size = [int(x * width / width_old) for x in old_size]
            I_resize = resize(I, new_size, mode="reflect", preserve_range=True, clip=False, anti_aliasing=True)
            width_resize = np.size(I_resize, 1)
            height_resize = np.size(I_resize, 0)
            start_height = (height_resize - height) // 2
            I_pad[:, :, :] = I_resize[start_height : (start_height + height), :, :]
        else:  # pad the height and crop
            new_size = [int(x * height / height_old) for x in old_size]
            I_resize = resize(I, new_size, mode="reflect", preserve_range=True, clip=False, anti_aliasing=True)
            width_resize = np.size(I_resize, 1)
            height_resize = np.size(I_resize, 0)
            start_width = (width_resize - width) // 2
            I_pad[:, :, :] = I_resize[:, start_width : (start_width + width), :]

        return Image.fromarray(I_pad.astype(np.uint8))


class CenterPad_threshold(object):
    def __init__(self, image_size, threshold=3 / 4):
        self.height = image_size[0]
        self.width = image_size[1]
        self.threshold = threshold

    def __call__(self, image):
        # pad the image to 16:9
        # pad height
        I = np.array(image)

        # for padded input
        height_old = np.size(I, 0)
        width_old = np.size(I, 1)
        old_size = [height_old, width_old]
        height = self.height
        width = self.width
        I_pad = np.zeros((height, width, np.size(I, 2)))

        ratio = height / width

        if height_old / width_old == ratio:
            if height_old == height:
                return Image.fromarray(I.astype(np.uint8))
            new_size = [int(x * height / height_old) for x in old_size]
            I_resize = resize(I, new_size, mode="reflect", preserve_range=True, clip=False, anti_aliasing=True)
            return Image.fromarray(I_resize.astype(np.uint8))

        if height_old / width_old > self.threshold:
            width_new, height_new = width_old, int(width_old * self.threshold)
            height_margin = height_old - height_new
            height_crop_start = height_margin // 2
            I_crop = I[height_crop_start : (height_crop_start + height_new), :, :]
            I_resize = resize(
                I_crop, [height, width], mode="reflect", preserve_range=True, clip=False, anti_aliasing=True
            )

            return Image.fromarray(I_resize.astype(np.uint8))

        if height_old / width_old > ratio:  # pad the width and crop
            new_size = [int(x * width / width_old) for x in old_size]
            I_resize = resize(I, new_size, mode="reflect", preserve_range=True, clip=False, anti_aliasing=True)
            width_resize = np.size(I_resize, 1)
            height_resize = np.size(I_resize, 0)
            start_height = (height_resize - height) // 2
            I_pad[:, :, :] = I_resize[start_height : (start_height + height), :, :]
        else:  # pad the height and crop
            new_size = [int(x * height / height_old) for x in old_size]
            I_resize = resize(I, new_size, mode="reflect", preserve_range=True, clip=False, anti_aliasing=True)
            width_resize = np.size(I_resize, 1)
            height_resize = np.size(I_resize, 0)
            start_width = (width_resize - width) // 2
            I_pad[:, :, :] = I_resize[:, start_width : (start_width + width), :]

        return Image.fromarray(I_pad.astype(np.uint8))


class CenterPadCrop_numpy(object):
    """
    pad the image according to the height
    """

    def __init__(self, image_size):
        self.height = image_size[0]
        self.width = image_size[1]

    def __call__(self, image, threshold=3 / 4):
        # pad the image to 16:9
        # pad height
        I = np.array(image)
        # for padded input
        height_old = np.size(I, 0)
        width_old = np.size(I, 1)
        old_size = [height_old, width_old]
        height = self.height
        width = self.width
        padding_size = width
        if image.ndim == 2:
            I_pad = np.zeros((width, width))
        else:
            I_pad = np.zeros((width, width, I.shape[2]))

        ratio = height / width
        if height_old / width_old == ratio:
            return I

        if height_old / width_old > threshold:
            width_new, height_new = width_old, int(width_old * threshold)
            height_margin = height_old - height_new
            height_crop_start = height_margin // 2
            I_crop = I[height_start : (height_start + height_new), :]
            I_resize = resize(
                I_crop, [height, width], mode="reflect", preserve_range=True, clip=False, anti_aliasing=True
            )
            return I_resize

        if height_old / width_old > ratio:  # pad the width and crop
            new_size = [int(x * width / width_old) for x in old_size]
            I_resize = resize(I, new_size, mode="reflect", preserve_range=True, clip=False, anti_aliasing=True)
            width_resize = np.size(I_resize, 1)
            height_resize = np.size(I_resize, 0)
            start_height = (height_resize - height) // 2
            start_height_block = (padding_size - height) // 2
            if image.ndim == 2:
                I_pad[start_height_block : (start_height_block + height), :] = I_resize[
                    start_height : (start_height + height), :
                ]
            else:
                I_pad[start_height_block : (start_height_block + height), :, :] = I_resize[
                    start_height : (start_height + height), :, :
                ]
        else:  # pad the height and crop
            new_size = [int(x * height / height_old) for x in old_size]
            I_resize = resize(I, new_size, mode="reflect", preserve_range=True, clip=False, anti_aliasing=True)
            width_resize = np.size(I_resize, 1)
            height_resize = np.size(I_resize, 0)
            start_width = (width_resize - width) // 2
            start_width_block = (padding_size - width) // 2
            if image.ndim == 2:
                I_pad[:, start_width_block : (start_width_block + width)] = I_resize[
                    :, start_width : (start_width + width)
                ]

            else:
                I_pad[:, start_width_block : (start_width_block + width), :] = I_resize[
                    :, start_width : (start_width + width), :
                ]

        crop_start_height = (I_pad.shape[0] - height) // 2
        crop_start_width = (I_pad.shape[1] - width) // 2

        if image.ndim == 2:
            return I_pad[
                crop_start_height : (crop_start_height + height), crop_start_width : (crop_start_width + width)
            ]
        else:
            return I_pad[
                crop_start_height : (crop_start_height + height), crop_start_width : (crop_start_width + width), :
            ]


@jit(nopython=True)
def iterSearchShader(padu, padv, xr, yr, W, H, maxIter, precision):
    # print('processing location', (xr, yr))
    #
    if abs(padu[yr, xr]) < precision and abs(padv[yr, xr]) < precision:
        return xr, yr

        # Our initialize method in this paper, can see the overleaf for detail
    if (xr + 1) <= (W - 1):
        dif = padu[yr, xr + 1] - padu[yr, xr]
    else:
        dif = padu[yr, xr] - padu[yr, xr - 1]
    u_next = padu[yr, xr] / (1 + dif)
    if (yr + 1) <= (H - 1):
        dif = padv[yr + 1, xr] - padv[yr, xr]
    else:
        dif = padv[yr, xr] - padv[yr - 1, xr]
    v_next = padv[yr, xr] / (1 + dif)
    i = xr - u_next
    j = yr - v_next
    i_int = int(i)
    j_int = int(j)

    # The same as traditional iterative search method
    for _ in range(maxIter):
        if not 0 <= i <= (W - 1) or not 0 <= j <= (H - 1):
            return i, j

        u11 = padu[j_int, i_int]
        v11 = padv[j_int, i_int]

        u12 = padu[j_int, i_int + 1]
        v12 = padv[j_int, i_int + 1]

        int1 = padu[j_int + 1, i_int]
        v21 = padv[j_int + 1, i_int]

        int2 = padu[j_int + 1, i_int + 1]
        v22 = padv[j_int + 1, i_int + 1]

        u = (
            u11 * (i_int + 1 - i) * (j_int + 1 - j)
            + u12 * (i - i_int) * (j_int + 1 - j)
            + int1 * (i_int + 1 - i) * (j - j_int)
            + int2 * (i - i_int) * (j - j_int)
        )

        v = (
            v11 * (i_int + 1 - i) * (j_int + 1 - j)
            + v12 * (i - i_int) * (j_int + 1 - j)
            + v21 * (i_int + 1 - i) * (j - j_int)
            + v22 * (i - i_int) * (j - j_int)
        )

        i_next = xr - u
        j_next = yr - v

        if abs(i - i_next) < precision and abs(j - j_next) < precision:
            return i, j

        i = i_next
        j = j_next

    # if the search doesn't converge within max iter, it will return the last iter result
    return i_next, j_next


# Bilinear interpolation
@jit(nopython=True)
def biInterpolation(distorted, i, j):
    i = u2(i)
    j = u2(j)
    Q11 = distorted[j, i]
    Q12 = distorted[j, i + 1]
    Q21 = distorted[j + 1, i]
    Q22 = distorted[j + 1, i + 1]

    return u1(
        Q11 * (i + 1 - i) * (j + 1 - j)
        + Q12 * (i - i) * (j + 1 - j)
        + Q21 * (i + 1 - i) * (j - j)
        + Q22 * (i - i) * (j - j)
    )


@jit(nopython=True)
def iterSearch(distortImg, resultImg, padu, padv, W, H, maxIter=5, precision=1e-2):
    for xr in range(W):
        for yr in range(H):
            # (xr, yr) is the point in result image, (i, j) is the search result in distorted image
            i, j = iterSearchShader(padu, padv, xr, yr, W, H, maxIter, precision)

            # reflect the pixels outside the border
            if i > W - 1:
                i = 2 * W - 1 - i
            if i < 0:
                i = -i
            if j > H - 1:
                j = 2 * H - 1 - j
            if j < 0:
                j = -j

            # Bilinear interpolation to get the pixel at (i, j) in distorted image
            resultImg[yr, xr, 0] = biInterpolation(
                distortImg[:, :, 0],
                i,
                j,
            )
            resultImg[yr, xr, 1] = biInterpolation(
                distortImg[:, :, 1],
                i,
                j,
            )
            resultImg[yr, xr, 2] = biInterpolation(
                distortImg[:, :, 2],
                i,
                j,
            )
    return None


def forward_mapping(source_image, u, v, maxIter=5, precision=1e-2):
    """
    warp the image according to the forward flow
    u: horizontal
    v: vertical
    """
    H = source_image.shape[0]
    W = source_image.shape[1]

    distortImg = np.array(np.zeros((H + 1, W + 1, 3)), dtype=np.uint8)
    distortImg[0:H, 0:W] = source_image[0:H, 0:W]
    distortImg[H, 0:W] = source_image[H - 1, 0:W]
    distortImg[0:H, W] = source_image[0:H, W - 1]
    distortImg[H, W] = source_image[H - 1, W - 1]

    padu = np.array(np.zeros((H + 1, W + 1)), dtype=np.float32)
    padu[0:H, 0:W] = u[0:H, 0:W]
    padu[H, 0:W] = u[H - 1, 0:W]
    padu[0:H, W] = u[0:H, W - 1]
    padu[H, W] = u[H - 1, W - 1]

    padv = np.array(np.zeros((H + 1, W + 1)), dtype=np.float32)
    padv[0:H, 0:W] = v[0:H, 0:W]
    padv[H, 0:W] = v[H - 1, 0:W]
    padv[0:H, W] = v[0:H, W - 1]
    padv[H, W] = v[H - 1, W - 1]

    resultImg = np.array(np.zeros((H, W, 3)), dtype=np.uint8)
    iterSearch(distortImg, resultImg, padu, padv, W, H, maxIter, precision)
    return resultImg


def random_mask(H, W, mask_size=200):
    """
    mask: ranges in [0,1]
    """
    masked_image = np.zeros([H, W, 3])
    mask = random_shapes(
        [H, W],
        max_shapes=1,
        min_shapes=1,
        max_size=mask_size,
        min_size=mask_size / 2,
        multichannel=False,
        intensity_range=[0, 0],
    )[0]
    mask = np.stack((mask, mask, mask), axis=-1)
    random_state = np.random.RandomState(None)
    distortion_range = 50
    alpha = np.random.rand() * 6
    forward_dx = (
        gaussian_filter((random_state.rand(H, W) * 2 - 1), distortion_range, mode="constant", cval=0) * alpha * 1000
    )
    forward_dy = (
        gaussian_filter((random_state.rand(H, W) * 2 - 1), distortion_range, mode="constant", cval=0) * alpha * 1000
    )
    mask = forward_mapping(mask, forward_dy, forward_dx, maxIter=3, precision=1e-3) / 255
    mask = 1 - gaussian(mask, sigma=0, preserve_range=True, multichannel=False, anti_aliasing=True)
    mask = mask[:, :, 0]

    return mask
