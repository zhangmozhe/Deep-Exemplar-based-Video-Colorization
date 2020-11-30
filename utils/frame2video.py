import argparse
import gc
import glob
import math
import os
import platform
import random
import shutil
import sys
import time
from datetime import datetime
from pdb import set_trace

import cv2
import matplotlib.pyplot as plt
import numpy as np
from skimage import color, io

video_name = "video.mp4"
frame_folder = "../../results/seq1_new/"

images = [
    img for img in os.listdir(frame_folder) if img.endswith(".jpg") or img.endswith(".bmp") or img.endswith(".png")
]
images.sort()
frame = cv2.imread(os.path.join(frame_folder, images[0]))
height, width, layers = frame.shape
video = cv2.VideoWriter("video.avi", cv2.VideoWriter_fourcc("X", "V", "I", "D"), 15, (width, height))
# video = cv2.VideoWriter('video.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 10, (width, height))
for image in images:
    video.write(cv2.imread(os.path.join(frame_folder, image)))

cv2.destroyAllWindows()
video.release()

import imageio

frames = []
for image in images:
    frames.append(imageio.imread(os.path.join(frame_folder, image)))
imageio.mimsave("movie.gif", frames)

