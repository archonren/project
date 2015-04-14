from sklearn.cross_validation import cross_val_score
from sklearn.ensemble import AdaBoostClassifier

import preprocess.split as s
from sklearn import preprocessing
from sklearn.metrics import precision_recall_fscore_support
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn import cross_validation


from sklearn.cross_validation import train_test_split
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.svm import SVC


def adaboost_test():


    tr = feature_selection(s.split_main()[0])
    test = feature_selection(s.split_main()[4])
    validn = feature_selection(s.split_main()[2])
    
    #get scaler that fits the training data
    scaler = normalize_data(tr)
    
   
    
    #scale the training data so that each feature has zero mean and unit variance
    scaled_tr_data = scaler.transform(tr)
    
    #print (scaled_tr_data.shape)
    
    scaled_validn_data = scaler.transform(validn)
    
    
    #scale the test data in a similar fashion
    scaled_test_data = scaler.transform(test)


    clf = AdaBoostClassifier(n_estimators=100)
    clf.fit(scaled_tr_data, s.split_main()[1][:,0])
    #scores = cross_val_score(clf, scaled_tr_data, scaled_test_data)
    #scores.mean()  
    pred = clf.predict(scaled_test_data)  
    print("Number of mislabeled points out of a total %d points : %d" % (test.shape[0],(s.split_main()[5][:,0] != pred).sum()))
    print precision_recall_fscore_support(s.split_main()[5], pred) 



def feature_selection(x):
    #print s.split_main()[0][0]
    sel = VarianceThreshold(threshold=(.8 * (1 - .8)))
    X = sel.fit_transform(x)
    #print X.shape
    #print X[0]
    return X
    

def normalize_data(x):
    scaler = preprocessing.StandardScaler().fit(x)
    return scaler
