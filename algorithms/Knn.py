__author__ = 'joh'
from numpy import *
from collections import Counter
from scipy.io import loadmat
import split as s

def classify(train_data, train_targets, test_data, k):
    resultlist = []
    c = 0
    for data in test_data.T:
        c +=1
        data = data.reshape((shape(data)[0]), 1)
        diff = abs(train_data - data)
        distance = sum(diff, axis=0)
        sortedindices = argsort(distance)
        first_occur_dict = {}
        votelist = []
        for i in range(k):
            votelist.append(int(train_targets[sortedindices[i]]))
            if (first_occur_dict.get(int(train_targets[sortedindices[i]]), None)) == None:
                first_occur_dict[int(train_targets[sortedindices[i]])] = i
        count = Counter(votelist)
        highestcount = count.most_common(1)[0][1]
        first_occur = k+1
        for key in count.keys():
            if count[key]== highestcount and first_occur_dict[key]<first_occur:
                first_occur = first_occur_dict[key]
                result = key
        resultlist.append(result)
        if int(c/100)*100 ==c or c ==1:
            print (c)
    return resultlist


def knn(train_data, train_targets, test_data, test_targets,k):
    road_num = 0.0
    road_acc = 0.0
    non_road_num = 0.0
    non_road_acc = 0.0
    resultlist = classify(train_data, train_targets, test_data, k)
    for result, targets in zip(resultlist, test_targets):
        targets = int(targets[0])
        if targets == 0:
            non_road_num +=1
            if result == targets:
                non_road_acc += 1
        if targets == 1:
            road_num +=1
            if result == targets:
                road_acc += 1
    print("the accuracy  for k = %i: for road is %.3f, for non road is %.3f\n" % (k, road_acc/road_num*100, non_road_acc/non_road_num*100))

def normalize_data(data):
    data_mean = data.mean(axis=1).reshape(-1,1)
    data_std = data.std(axis=1).reshape(-1,1)
    data_out = (data - data_mean)/data_std
    return data_out


training_data = normalize_data(s.split_main()[0].T).T
dev_data = normalize_data(s.split_main()[2].T).T
test_data = normalize_data(s.split_main()[4].T).T
training_target = s.split_main()[1]
dev_target = s.split_main()[3]
test_target = s.split_main()[5]
print ('start')
knn(training_data.T, training_target, test_data.T, test_target,11) #the accuracy  for k = 5: for road is 57.968, for non road is 94.524 the accuracy  for k = 11: for road is 60.908, for non road is 94.484

