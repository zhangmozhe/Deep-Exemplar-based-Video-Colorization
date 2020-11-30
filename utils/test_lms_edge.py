import matplotlib.pyplot as plt
import numpy as np
import skimage
from skimage import color, feature, io, morphology
from skimage.filters import prewitt, roberts, scharr, sobel
from skimage.morphology import square


def rgb2lms(rgb_image):
    lms_image = rgb_image
    r = rgb_image[:, :, 0]
    g = rgb_image[:, :, 1]
    b = rgb_image[:, :, 2]

    l = 17.8824 * r + 43.5161 * g + 4.1194 * b
    m = 3.4557 * r + 27.1554 * g + 3.8671 * b
    s = 0.0300 * r + 0.1843 * g + 1.4671 * b
    lms_image = np.stack((l, m, s), axis=2)
    return l, m, s


def image_normalize(rg_image):
    rg_image = rg_image - rg_image.min()
    rg_image = rg_image / (rg_image.max() - rg_image.min()) * 255
    return rg_image


image_rgb = io.imread("ref5.jpg")
plt.imshow(image_rgb, cmap="gray")
L, M, S = rgb2lms(image_rgb)
image_lab = color.rgb2lab(image_rgb)
a_edge = sobel(image_lab[:, :, 1])
b_edge = sobel(image_lab[:, :, 2])
color_edge = np.sqrt(a_edge ** 2 + b_edge ** 2)
luminance_edge = sobel(image_lab[:, :, 0])
canny_edge_a = feature.canny(image_lab[:, :, 1], sigma=3, low_threshold=3)
canny_edge_b = feature.canny(image_lab[:, :, 2], sigma=3, low_threshold=3)

plt.figure()
# edge = morphology.opening(np.logical_and(color_edge > 8, luminance_edge > 4))
edge = morphology.thin(
    morphology.closing((morphology.opening(np.logical_and(color_edge > 10, luminance_edge > 2))), square(1))
)
plt.imshow(edge, cmap="gray")

plt.figure()
plt.imshow(np.logical_and(canny_edge_a, canny_edge_b), cmap="gray")
plt.show()
