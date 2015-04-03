__author__ = 'Guanhua, Joms'
from sklearn import svm
import preprocess.split as s

def svm_test():
    ''' A simple SVM Classifier.
        s.split_main()[0] - Training Data
        s.split_main()[1] - Training Targets
        s.split_main()[4] - Test Data
        s.split_main()[5] - Test Targets
    '''
    s_classifier = svm.SVC()
    s_classifier.fit(s.split_main()[0], s.split_main()[1][:,0])
    pred = s_classifier.predict(s.split_main()[4])
    print("Number of mislabeled points out of a total %d points : %d" % (s.split_main()[4].shape[0],(s.split_main()[5][:,0] != pred).sum()))
