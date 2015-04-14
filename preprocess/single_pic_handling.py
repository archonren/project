__author__ = 'Guanhua, Joms'
from data_handling import *
from file_handling import *
from feature_handling import *
from numpy import shape, concatenate
from scipy import ndimage


def single_pic_feature_computation(path_png, path_gt, path_dat):
    # load data
    png = ndimage.imread(path_png)
    data_label, label_dict = label(path_dat, shape(png)[0], shape(png)[1])
    target = gt(path_gt)
    group_number = max(data_label.ravel())+1

    # calculate features
    nrg = normalized_RG(png)
    lab = to_lab(png)
    ocs = OCS(png)
    hsv = to_hsv(png)
    feature = concatenate((png, nrg, ocs, hsv, lab), axis=2)
    feature = avg_feature(feature, label_dict, group_number)


    # calculate superpixel target
    target = avg(target, label_dict, group_number)
    target = superpixel_target(target, group_number)

    return feature, target, shape(png)
