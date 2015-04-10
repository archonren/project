__author__ = 'Guanhua, Joms'
from scipy.io import loadmat, savemat
from numpy import vstack, array
import os


def output_data(key, heading, num_imgs):

    arr = []
    data = vstack(((heading[key][:, i])[0] for i in range(num_imgs)))
    data_mean = data.mean(axis=0).reshape(-1, 1)
    data_std = data.std(axis=0).reshape(-1, 1)
    data_mean = data_mean.T
    data_std = data_std.T
    for i in range(num_imgs):
        arr.append(((heading[key][:, i][0] - data_mean)/data_std))
    arr = array(arr)
    return arr


def mat_output_main(input_fldr, file_heading):
    key = 'feature_vector'
    path = os.path.join(input_fldr, '%s' % file_heading, "%s_data.mat" % file_heading)
    data_mat = loadmat(path)
    num_imgs = data_mat[key].shape[1]
    output = output_data(key, data_mat, num_imgs)
    params_dict = {'feature_vector': output, 'feature_dim': [13], 'total_pic': data_mat['total_pic'],
                   'num_of_super_pixel': data_mat['num_of_super_pixel'], 'dim': data_mat['dim']}
    save_path = os.path.join(input_fldr, "%s_final_data.mat" % file_heading)
    savemat(save_path, params_dict)


mat_output_main('c:/data_road/testing/', 'um')
mat_output_main('c:/data_road/testing/', 'umm')
mat_output_main('c:/data_road/testing/', 'uu')
