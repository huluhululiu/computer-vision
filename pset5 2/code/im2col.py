import numpy as np


def get_im2col_indices(x_shape, field_height, field_width, stride=1):
    # First figure out what the size of the output should be
    N, H, W, C = x_shape
    # assert (H - field_height) % stride == 0
    # assert (W - field_width) % stride == 0
    out_height = int((H - field_height)/stride + 1)
    out_width = int((W - field_width)/stride + 1)

    i0 = np.repeat(np.arange(field_height), field_width)
    i0 = np.tile(i0, C)
    i1 = stride*np.repeat(np.arange(out_height), out_width)
    j0 = np.tile(np.arange(field_width), field_height * C)
    j1 = stride*np.tile(np.arange(out_width), out_height)
    i = i0.reshape(-1, 1) + i1.reshape(1, -1)
    j = j0.reshape(-1, 1) + j1.reshape(1, -1)

    k = np.repeat(np.arange(C), field_height * field_width).reshape(-1, 1)

    return (i.astype(int), j.astype(int), k.astype(int))


def im2col_indices(x, field_height, field_width, stride=1):
    """ An implementation of im2col based on some fancy indexing """

    i, j, k = get_im2col_indices(x.shape, field_height, field_width, stride)

    cols = x[:, i, j, k]
    C = x.shape[-1]
    return cols.transpose(1, 2, 0).reshape(field_height * field_width * C, -1)


def col2im_indices(cols, x_shape, field_height, field_width, stride=1):
    """ An implementation of col2im based on fancy indexing and np.add.at """
    N, H, W, C  = x_shape
    x = np.zeros(x_shape, dtype=cols.dtype)
    i, j, k = get_im2col_indices(x_shape, field_height, field_width, stride)
    cols_reshaped = cols.reshape(C * field_height * field_width, -1, N)
    cols_reshaped = cols_reshaped.transpose(2, 0, 1)
    np.add.at(x, (slice(None), i, j, k), cols_reshaped)
    return x
