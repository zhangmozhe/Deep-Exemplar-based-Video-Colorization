from utils.util import to_np


def histogram_logger(tb_writer, iter_index, net=None):
    if net is not None:
        for tag, value in net.named_parameters():
            tag = tag.replace(".", "/")
            tb_writer.add_histogram(tag, to_np(value), iter_index)
            tb_writer.add_histogram(tag + "/grad", to_np(value.grad), iter_index)


def value_logger(tb_writer, iter_index, loss_info):
    for tag, value in loss_info.items():
        tb_writer.add_scalar(tag, value, iter_index)
