__author__ = 'Guanhua, Joms'

from struct import unpack
from numpy import reshape, ndindex, empty
from scipy.ndimage import imread


def label(filepath, height, width):
    """
    :param filepath: path of label.dat
    :return data_label: ndarray that contain label of each pixel, dim = (height, width)
    :return label_dict: dict with pixel position as keys and superpixel label as values
    """
    label_file = open(filepath, "rb")
    data_label = []
    while True:
        d = label_file.read(4)
        if len(d) != 4:
            break
        else:
            data_label.append(unpack("i", d))
    label_file.close()
    data_label = reshape(data_label, (height, width))
    # create a dict for a quick check of labels
    label_dict = {}
    for x in ndindex((height, width)):
        label_dict[x] = data_label[x[0]][x[1]]

    return data_label, label_dict


def gt(filepath):
    """
    :param filepath: ground_true file
    :return: array that label each pixel as 1 if true, 0 if false
    """
    gt = imread(filepath)
    target = empty((gt.shape[0], gt.shape[1], 1))
    for x in ndindex(gt.shape[0], gt.shape[1]):
        if gt[x][2] != 255:
            target[x] = 0
        else:
            target[x] = 1
    return target