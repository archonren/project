import os
from scipy.io import loadmat
from numpy import array, vstack, reshape, delete
 

from sklearn.metrics import precision_recall_fscore_support
from sklearn.ensemble import AdaBoostClassifier

from sklearn import preprocessing
from sklearn.feature_selection import VarianceThreshold

from sklearn.svm import SVC



def normalize_data(x):
    scaler = preprocessing.StandardScaler().fit(x)
    return scaler 


def adaboost(D):
      
             
    feature = D['feature_vector']
    target = D['target']




    print feature.shape
    #print feature[50]


    um_targets = vstack((((target[:,i])[0]) for i in range(95))) #[0:385]

    print um_targets.shape




    #print "####################################################"
    #print ((target[:,0])[0])[0:385].shape      
                

    train_size = int (0.6 * len(feature))
    validation_size = int (0.1 * len(feature))

    train_d = feature[0:train_size]
    validation_d = feature[train_size:(train_size+validation_size)]
    test_d = feature[(train_size+validation_size):]

    train_t = um_targets[0:train_size]
    validation_t = um_targets[train_size:(train_size+validation_size)]
    test_t = um_targets[(train_size+validation_size):]


    #print train_t.shape
    #print train_d.shape
    #print train_d[50]

    #print validation_t[:,0].shape

    scaler = normalize_data(train_d)
    #scale the training data so that each feature has zero mean and unit variance
    scaled_tr_data = scaler.transform(train_d)
    
    #print (scaled_tr_data)
    
    scaled_validn_data = scaler.transform(validation_d)
    
    
    #scale the test data in a similar fashion
    scaled_test_data = scaler.transform(test_d)




    clf = AdaBoostClassifier(n_estimators=1000)
    clf.fit(scaled_tr_data, train_t[:,0])
 
    pred = clf.predict(scaled_test_data)  

    print("Number of mislabeled points out of a total %d  points : %d" % (test_d.shape[0], (test_t[:,0] != pred).sum()))
    print precision_recall_fscore_support(test_t[:,0], pred)
    
    
    
UM = loadmat("C:\Users\Joms\Desktop\um\um_all_data.mat") 
UMM = loadmat("C:\Users\Joms\Desktop\umm\umm_all_data.mat") 
UU = loadmat("C:\Users\Joms\Desktop\uu\uu_all_data.mat") 

print "UM"
adaboost(UM)
print "UMM"
adaboost(UMM)
print "UU"
adaboost(UU)
