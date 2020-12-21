import threading

import numpy as np


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
