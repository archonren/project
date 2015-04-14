from __future__ import print_function

#from sklearn import svm
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



def svm_test():
    ''' A simple SVM Classifier.
        s.split_main()[0] - Training Data
        s.split_main()[1] - Training Targets
        s.split_main()[4] - Test Data
        s.split_main()[5] - Test Targets
    '''
    
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
    
    
    
    tuned_parameters = [{'C': [1, 10, 100, 1000, 2000], 'gamma': [0.01, 0.001, 0.0001, .00001], 'kernel': ['rbf']},]
    scores = ['precision', 'recall']

    #for score in scores:
    #    print("# Tuning hyper-parameters for %s" % score)
    #    print()

    #    clf = GridSearchCV(SVC(cache_size=3000, C=1), tuned_parameters, cv=5, scoring=score)
    #    print (s.split_main()[1][:,0].shape)
    #    clf.fit(scaled_tr_data, s.split_main()[1][:,0])

    #    print("Best parameters set found on development set:")
    #    print()
    #    print(clf.best_params_)
    #    print()
    #    print("Grid scores on development set:")
    #    print()
    #    for params, mean_score, scores in clf.grid_scores_:
    #        print("%0.3f (+/-%0.03f) for %r"
    #          % (mean_score, scores.std() * 2, params))
    #    print()

    #    print("Detailed classification report:")
    #    print()
    #    print("The model is trained on the full development set.")
    #    print("The scores are computed on the full evaluation set.")
    #    print()
    #    y_true, y_pred = s.split_main()[3][:,0], clf.predict(scaled_validn_data)
    #    print(classification_report(y_true, y_pred))
    #    print()
    
    #s_classifier = svm.SVC(C=1000, class_weight='auto', cache_size=1000, gamma=0.01)
    s_classifier = SVC(cache_size=5000, C=1000, gamma=0.0001)
    #print s_classifier.kernel
    #print s_classifier.gamma
    s_classifier.fit(scaled_tr_data, s.split_main()[1][:,0])
    pred = s_classifier.predict(scaled_test_data)
    print("Number of mislabeled points out of a total %d points : %d" % (test.shape[0],(s.split_main()[5][:,0] != pred).sum()))
    print (precision_recall_fscore_support(s.split_main()[5], pred))
    
def normalize_data(x):
    scaler = preprocessing.StandardScaler().fit(x)
    return scaler
    
def feature_selection(x):
    #print s.split_main()[0][0]
    sel = VarianceThreshold(threshold=(.8 * (1 - .8)))
    X = sel.fit_transform(x)
    #print X.shape
    #print X[0]
    return X
    
    #X_new = SelectKBest(chi2, k=2).fit_transform(s.split_main()[0], s.split_main()[1])
    #print X_new.shape
    #print X_new
    
