__author__ = 'Guanhua, Joms'
import os
from numpy import *
from single_pic_handling import *
from scipy.io import savemat


def pic_package_to_mat(input_fldr, file_heading, number_of_file):
    feature = []
    target = []
    for n in range(number_of_file):
        png_file_name = "%s_%06d.png" % (file_heading, n)
        path_png = os.path.join(input_fldr, png_file_name)
        gt_file_name = "%s_road_%06d.png" % (file_heading, n)
        path_gt = os.path.join(input_fldr, gt_file_name)
        dat_file_name = "%s_%06d.dat" % (file_heading, n)
        path_dat = os.path.join(input_fldr, dat_file_name)
        pic_feature, pic_target = single_pic_feature_computation(path_png, path_gt, path_dat)
        feature.append(pic_feature)
        target.append(pic_target)
    feature = array(feature)
    target = array(target)

    #save to mat
    feature_explanation = ["r", "g", "b", "nr", "ng", "o1", "o2", "h", "s", "v", "l", "a", "b"]
    params_dict = {}
    params_dict['feature_vector'] = feature
    params_dict['feature_explanation'] = feature_explanation
    params_dict['target'] = target
    params_dict['feature_dim'] = [13]
    params_dict['total_pic'] = [number_of_file]
    save_path = os.path.join(input_fldr, "%s_data" % file_heading)
    savemat(save_path, params_dict)


