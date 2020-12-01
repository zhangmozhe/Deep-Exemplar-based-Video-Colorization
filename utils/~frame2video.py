import os

import cv2

video_name = "video.mp4"
frame_folder = "../../results/seq1_new/"

images = [
    img for img in os.listdir(frame_folder) if img.endswith(".jpg") or img.endswith(".bmp") or img.endswith(".png")
]
images.sort()
frame = cv2.imread(os.path.join(frame_folder, images[0]))
height, width, layers = frame.shape
video = cv2.VideoWriter("video.avi", cv2.VideoWriter_fourcc("X", "V", "I", "D"), 15, (width, height))

for image in images:
    video.write(cv2.imread(os.path.join(frame_folder, image)))

cv2.destroyAllWindows()
video.release()

import imageio

frames = [imageio.imread(os.path.join(frame_folder, image)) for image in images]

imageio.mimsave("movie.gif", frames)
