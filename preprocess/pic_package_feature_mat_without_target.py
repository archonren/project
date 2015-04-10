__author__ = 'Guanhua, Joms'
import os
from real_single_pic import *
from scipy.io import savemat
import pickle
from numpy import array


def pic_package_to_mat(input_fldr, file_heading, number_of_file):
    feature = []
    num_super_pixel = []
    dim = []
    for n in range(number_of_file):
        png_file_name = "%s_%06d.png" % (file_heading, n)
        path_png = os.path.join(input_fldr, png_file_name)
        dat_file_name = "%s_%06d.dat" % (file_heading, n)
        pkl_file_name = "%s_%06d.pkl" % (file_heading, n)
        path_dat = os.path.join(input_fldr, dat_file_name)
        path_pkl = os.path.join(input_fldr, pkl_file_name)
        pic_feature, label_dict, pic_dim = single_pic_feature_computation_without_target(path_png, path_dat)
        num_super_pixel.append(shape(pic_feature)[0])
        feature.append(pic_feature)
        dim.append(pic_dim)
        with open(path_pkl, 'wb') as output:
            pickle.dump(label_dict, output, pickle.HIGHEST_PROTOCOL)
    feature = array(feature)
    dim = array(dim)

    # save to mat
    feature_explanation = ["r", "g", "b", "nr", "ng", "o1", "o2", "h", "s", "v", "l", "a", "b"]
    params_dict = {'feature_vector': feature, 'feature_explanation': feature_explanation, 'feature_dim': [13],
                   'total_pic': [number_of_file], 'num_of_super_pixel': num_super_pixel, 'dim': dim}
    save_path = os.path.join(input_fldr, "%s_data" % file_heading)
    savemat(save_path, params_dict)

pic_package_to_mat('c:/data_road/testing/uu/', 'uu', 100)  # about 25 min to process
pic_package_to_mat('c:/data_road/testing/um/', 'um', 96)
pic_package_to_mat('c:/data_road/testing/umm/', 'umm', 94)
