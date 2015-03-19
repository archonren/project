__author__ = 'Guanhua, Joms'
from skimage.color import rgb2lab, rgb2hsv
from numpy import empty, ndindex


def to_lab(ndimage):
    return rgb2lab(ndimage)


def to_hsv(ndimage):
    return rgb2hsv(ndimage)


def normalized_RG(ndimage):
    """
    :param ndimage: a image file
    :return: normalized RG for every pixel
    """
    imagedim = (ndimage.shape[0], ndimage.shape[1])
    normalized_RG = empty((imagedim[0], imagedim[1], 2))
    for x in ndindex(imagedim):
        R = ndimage[x[0]][x[1]][0]
        G = ndimage[x[0]][x[1]][1]
        B = ndimage[x[0]][x[1]][2]
        three_I = int(R)+int(G)+int(B)
        RG = [R/three_I, G/three_I]
        normalized_RG[x] = RG
    return normalized_RG


def OCS(ndimage):
    """
    :param ndimage: a image file
    :return: opponent color space for every pixel
    """
    imagedim = (ndimage.shape[0], ndimage.shape[1])
    feature_ocs = empty((imagedim[0], imagedim[1], 2))
    for x in ndindex(imagedim):
        R = int(ndimage[x[0]][x[1]][0])
        G = int(ndimage[x[0]][x[1]][1])
        B = int(ndimage[x[0]][x[1]][2])
        ocs = [(R-G)/2.0, B/2.0-R/4.0-G/4.0]
        feature_ocs[x] = ocs
    return feature_ocs