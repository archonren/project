__author__ = 'Guanhua, Joms'
from scipy.io import loadmat
from numpy import vstack


def merge_um_umm_uu(key, UM, num_imgs_um, UMM, num_imgs_umm, UU, num_imgs_uu):
    ''' Merges the data set in the 3 .mat files specified by the key - 
        feature_vector or target.
        (Merges the data or the targets in the 3 .mat files and returns it.)
    '''
    
    um = vstack(((UM[key][:,i])[0] for i in range(num_imgs_um)))
    umm = vstack(((UMM[key][:,i])[0] for i in range(num_imgs_umm)))
    uu = vstack(((UU[key][:,i])[0] for i in range(num_imgs_uu)))
    
    d = vstack((um, umm, uu))
    
    return d
    

def split_d(d):
    ''' Splits the merged data or targets as specified by d into training, 
        validation, and test data sets.
        Returns the tuple consisting of training, validation and testing set.
    '''

    train_size = int (0.6 * len(d))
    validation_size = int (0.1 * len(d))

    train_d = d[0:train_size]
    validation_d = d[train_size:(train_size+validation_size)]
    test_d = d[(train_size+validation_size):]
    
    return (train_d, validation_d, test_d)
    

def split_main():
    ''' Returns a tuple of (training data, training targets, validation data, validation targets, 
        test data, test targets)
    '''
    
    UM = loadmat('data/um_data.mat')
    UMM = loadmat('data/umm_data.mat')
    UU = loadmat('data/uu_data.mat')
    num_imgs_um = UM['feature_vector'].shape[1]
    num_imgs_umm = UMM['feature_vector'].shape[1]
    num_imgs_uu = UU['feature_vector'].shape[1]
    
    data = merge_um_umm_uu('feature_vector', UM, num_imgs_um, UMM, num_imgs_umm, UU, num_imgs_uu)
    train_data, validation_data, test_data =  split_d(data)
    
    targets = merge_um_umm_uu('target', UM, num_imgs_um, UMM, num_imgs_umm, UU, num_imgs_uu)
    train_tgts, validation_tgts, test_tgts =  split_d(targets)   
    
    return (train_data, train_tgts, validation_data, validation_tgts, test_data, test_tgts)




    
