import os
from scipy.io import loadmat
from skimage.color import rgb2gray
from numpy import array, vstack, reshape, delete
from skimage.feature import local_binary_pattern
from skimage.feature import hog
from skimage.filters import sobel_h,sobel_v
from scipy.io import savemat    

from sklearn.metrics import precision_recall_fscore_support
from sklearn.ensemble import AdaBoostClassifier

from sklearn import preprocessing
from sklearn.feature_selection import VarianceThreshold

from sklearn.svm import SVC
 
 
def normalize_data(x):
    scaler = preprocessing.StandardScaler().fit(x)
    return scaler 
    
       

def make_feature_vector(D, num_of_files, input_fldr, file_heading):             

    feature = D['feature_vector'] #first 13 features
    target = D['target']
    all_super_features = [] #to have all 18 features
        
    for n in range(num_of_files):
        #get r, g, b values for superpixels of the image being processed now                
        super_image = vstack((feature[0])[n][:,k] for k in range(3))
        super_image = super_image.T
    
        #this if does the following:
        #1. limits number of superpixels to 385 in the hope that all images have at least 385 images
        #Note: Case where images have less than 385 superpixels - throws error
        #2. The superpixels on the extreme right edge of the image is removed. This is based on an approximation that there are approximately 11 superpixels along the y-axis of the image.
        #This assumptions seems to be consistent throughout the training data - um, umm and uu.
        #3. When these superpixels are removed, their corresponding features in the feature vector and the targets in the target vector too are removed.
 
        if(super_image.shape[0] > 385):
            diff = super_image.shape[0] - 385
            for i in range(diff):
                #remove from r, g, b of superpixels
                super_image = delete(super_image, (i+1)*11, 0) #super_image.shape[0]-1
                #remove from targets
                ((target[:,n])[0]) = delete(((target[:,n])[0]), (i+1)*11, 0) #delete extreme rightmost column of superpixels(hopefully)
            
                #remove from feature vector
                (feature[:,n])[0] = delete((feature[:,n])[0], (i+1)*11, 0)

    
        #reshape the superpixel to an approximated dimension of 11*35 (This can be later automated and read from the feature vector for better performance and accuracy. Leaving this for now.)
        super_image = reshape(super_image,(11, 35, 3))
    
        #convert to grayscale
        gray = rgb2gray(super_image)
    
        #these features are dependent on the shape of the image, i.e. image as a whole. image is reshaped for this.
        #60, 10 are values selected by cross-validation
        l = local_binary_pattern(gray, 60, 10)
        h_gradient = sobel_h(gray)
        v_gradient = sobel_v(gray)
    
        #combine all 17 features together into 1 feature_vector
        #The 9th(0 index) feature - "v" - does not have 385 entries (only about 100 or so; don't know why). So I am not including that.
        #, reshape((feature[0])[n][:9], (1, (feature[0])[n][:9].size))
        all_features = vstack((reshape((feature[0])[n][:,0], (1, 385)), reshape((feature[0])[n][:,1], (1, 385)), reshape((feature[0])[n][:,2], (1, 385)), reshape((feature[0])[n][:,3], (1, 385)), reshape((feature[0])[n][:,4], (1, 385)), reshape((feature[0])[n][:,5], (1, 385)), reshape((feature[0])[n][:,6], (1, 385)), reshape((feature[0])[n][:,7], (1, 385)), reshape((feature[0])[n][:,8], (1, 385)), reshape((feature[0])[n][:,10], (1, 385)), reshape((feature[0])[n][:,11], (1, 385)), reshape((feature[0])[n][:,12], (1, 385)), reshape((feature[0])[n][:,13], (1, 385)), reshape((feature[0])[n][:,14], (1, 385)), reshape(l,(1, 385)), reshape(h_gradient, (1, 385)) , reshape(v_gradient, (1, 385))  ))
        all_features = all_features.T
    
        if n!=0:
            all_super_features = vstack((all_super_features, all_features))
        else:
            all_super_features = all_features
        
        
        #save the new feature vector with 17 features. "v" is not included
        feature_explanation = ["r", "g", "b", "nr", "ng", "o1", "o2", "h", "s", "l", "a", "b", "x", "y", "texture_lbp", "h_gradient", "v_gradient"]
        params_dict = {}
        params_dict['feature_vector'] = all_super_features
        params_dict['feature_explanation'] = feature_explanation
        params_dict['target'] = target
        params_dict['feature_dim'] = [17]
        params_dict['total_pic'] = [num_of_files]
        save_path = os.path.join(input_fldr, "%s_data" % file_heading)
        savemat(save_path, params_dict)


if __name__ == '__main__': 
    
    #change the path; i know this is sloppy
    
    #the um_data and the other files are the files with the 15 features: (including x, y)
    UM = loadmat("C:\Users\Joms\Desktop\um\um_data.mat")
    make_feature_vector(UM, 95, "C:\Users\Joms\Desktop\um", "um_all")
    
    UMM = loadmat("C:\Users\Joms\Desktop\umm\umm_data.mat")
    make_feature_vector(UMM, 96, "C:\Users\Joms\Desktop\umm", "umm_all")

    UU = loadmat("C:\Users\Joms\Desktop\um\uu_data.mat")
    make_feature_vector(UU, 98, "C:\Users\Joms\Desktop\uu", "uu_all")                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      
