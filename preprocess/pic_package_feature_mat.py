__author__ = 'Guanhua, Joms'
import os
from single_pic_handling import *
from scipy.io import savemat


def pic_package_to_mat(input_fldr, file_heading, number_of_file):
    feature = []
    target = []
    dim = []
    for n in range(number_of_file):
        png_file_name = "%s_%06d.png" % (file_heading, n)
        path_png = os.path.join(input_fldr, png_file_name)
        gt_file_name = "%s_road_%06d.png" % (file_heading, n)
        path_gt = os.path.join(input_fldr, gt_file_name)
        dat_file_name = "%s_%06d.dat" % (file_heading, n)
        path_dat = os.path.join(input_fldr, dat_file_name)
        pic_feature, pic_target, pic_dim = single_pic_feature_computation(path_png, path_gt, path_dat)
        feature.append(pic_feature)
        target.append(pic_target)
        dim.append(pic_dim)
    feature = array(feature)
    target = array(target)
    dim = array(dim)

    # save to mat
    feature_explanation = ["r", "g", "b", "nr", "ng", "o1", "o2", "h", "s", "v", "l", "a", "b", "x", "y"]
    params_dict = {'feature_vector': feature, 'feature_explanation': feature_explanation, 'target': target,
                   'feature_dim': [15], 'total_pic': [number_of_file], 'dim': dim}
    save_path = os.path.join(input_fldr, "%s_data" % file_heading)
    savemat(save_path, params_dict)

pic_package_to_mat('c:/data_road1/SLIC/uu', 'uu', 98)
pic_package_to_mat('c:/data_road1/SLIC/um', 'um', 95)
pic_package_to_mat('c:/data_road1/SLIC/umm', 'umm', 96)
