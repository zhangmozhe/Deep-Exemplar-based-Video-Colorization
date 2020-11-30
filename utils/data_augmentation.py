import random

import cv2
import lib.functional as F
import numpy as np
from PIL import Image
from scipy.misc import imread, imresize, imsave
from scipy.ndimage.filters import gaussian_filter
from skimage import color

cv2.setNumThreads(0)


# %% Dataloader
class RGB2Lab(object):
    def __init__(self):
        pass

    def __call__(self, inputs):
        image_lab = color.rgb2lab(inputs)
        return image_lab


class Guassian_noise(object):
    """Elastic distortion
    """

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
    """Elastic distortion
    """

    def __init__(self, distortion_level=3, flip_probability=0.1):
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
        outputs = F.to_mytensor(inputs)
        return outputs


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
        I_resize = imresize(img, new_size)
        width_new = np.size(I_resize, 1)
        height_new = np.size(I_resize, 0)

        I_pad = np.zeros((self.image_size, self.image_size, 3))
        start_height = (self.image_size - new_size[0]) // 2
        start_width = (self.image_size - new_size[1]) // 2
        I_pad[start_height : (start_height + height_new), start_width : (start_width + width_new), :] = I_resize

        return Image.fromarray(I_pad.astype(np.uint8))
