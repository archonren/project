__author__ = 'joh'
import argparse,os
import split as s
from numpy import *
from scipy.io import savemat


parser = argparse.ArgumentParser()
parser.add_argument('output_fldr',type=str, help='output folder')
arguments = parser.parse_args()
path = os.path.join(arguments.output_fldr, "data_normalized.mat")


def normalize_data(data):
    data_mean = data.mean(axis=1).reshape(-1,1)
    data_std = data.std(axis=1).reshape(-1,1)
    data_out = (data - data_mean)/data_std
    return data_out


def normalize():
    ''' A simple Naive Gaussian Bayes Classifier.
    s.split_main()[0] - Training Data
    s.split_main()[1] - Training Targets
    s.split_main()[4] - Test Data
    s.split_main()[5] - Test Targets
    '''
    training_data = normalize_data(s.split_main()[0].T).T
    dev_data = normalize_data(s.split_main()[2].T).T
    test_data = normalize_data(s.split_main()[4].T).T
    training_target = s.split_main()[1]
    dev_target = s.split_main()[3]
    test_target = s.split_main()[5]
    # save to mat
    params_dict = {}
    params_dict['training_data'] = training_data
    params_dict['dev_data'] = dev_data
    params_dict['test_data'] = test_data
    params_dict['training_target'] = training_target
    params_dict['dev_target'] = dev_target
    params_dict['test_target'] = test_target
    params_dict['feature_dim'] = [13]
    params_dict['training_dim'] = [shape(training_data)[0]]
    params_dict['dev_dim'] = [shape(dev_data)[0]]
    params_dict['test_dim'] = [shape(test_data)[0]]
    savemat(path, params_dict)


normalize()