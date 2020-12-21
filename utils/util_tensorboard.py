import threading

import numpy as np

from utils.util import to_np


def histogram_logger(tb_writer, iter_index, net=None):
    if net is not None:
        for tag, value in net.named_parameters():
            tag = tag.replace('.', '/')
            tb_writer.add_histogram(tag, to_np(value), iter_index)
            tb_writer.add_histogram(tag + '/grad', to_np(value.grad), iter_index)


def value_logger(tb_writer, iter_index, loss_info):
    for tag, value in loss_info.items():
        tb_writer.add_scalar(tag, value, iter_index)


class TBImageRecorder(threading.Thread):
    """
    TBImageRecorder
    """

    def __init__(self, tb_writer, func, queue):
        super(TBImageRecorder, self).__init__()
        self._tb_writer = tb_writer
        self._func = func
        self._queue = queue

    def run(self):
        while True:
            msgs, iter_index = self._queue.get()
            if msgs:
                img_info = self._func(*msgs)

                print("logging the images")
                for tag, images in img_info.items():
                    if images is not None:
                        self._tb_writer.add_image(tag, np.clip(images, 0, 255).astype(np.uint8), iter_index)
            else:
                break
