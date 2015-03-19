__author__ = 'Guanhua, Joms'
from numpy import zeros, ndindex


def avg(ndimage, label_dict, group_number):
    imagedim = (ndimage.shape[0], ndimage.shape[1])
    feature_avg = zeros((group_number, ndimage.shape[2]))
    total = zeros((group_number, 1))
    for x in ndindex(imagedim):
        label = label_dict[x]
        total[label] += 1
        feature_avg[label] += ndimage[x]
    feature_avg = feature_avg/total
    return feature_avg


def superpixel_target(target, group_number):
    for x in range(group_number):
        if target[x] >= 0.5:
            target[x] = 1
        else:
            target[x] = 0
    return target


def unsample(target, label_dict):
    png = zeros((375, 1242, 3), dtype="uint8")
    for x in ndindex(png.shape[0], png.shape[1]):
        if target[label_dict[x]] != 1:
            png[x] = [0, 0, 130]
            print(type(png[x][0]))
    return png
