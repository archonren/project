from sklearn.naive_bayes import GaussianNB
import preprocess.split as s

def gnb_test():
    ''' A simple Naive Gaussian Bayes Classifier.
        s.split_main()[0] - Training Data
        s.split_main()[1] - Training Targets
        s.split_main()[4] - Test Data
        s.split_main()[5] - Test Targets
    '''
    gnb = GaussianNB()
    y_pred = gnb.fit(s.split_main()[0], s.split_main()[1][:,0]).predict(s.split_main()[4])
    print("Number of mislabeled points out of a total %d points : %d" % (s.split_main()[4].shape[0],(s.split_main()[5][:,0] != y_pred).sum()))
