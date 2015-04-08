from numpy import *
from scipy.io import loadmat
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('train_path', help='Path to training database')
arguments = parser.parse_args()
roaddata = loadmat(arguments.train_path)


class Prob_Neural_Network(object):
    def __init__(self, roaddata):
        self._training_data = roaddata['training_data']
        self._dev_data = roaddata['dev_data']
        self._test_data = roaddata['test_data']
        self._training_target = roaddata['training_target']
        self._dev_target = roaddata['dev_target']
        self._test_target = roaddata['test_target']
        self._feature_dim = roaddata['feature_dim']
        self._training_dim = roaddata['training_dim']
        self._dev_dim = roaddata['dev_dim']
        self._test_dim = roaddata['test_dim']
        road_data = []
        non_road_data = []
        for i in range(self._training_dim):
            if self._training_target[i] == 1:
                road_data.append(self._training_data[i].tolist())
            else:
                non_road_data.append(self._training_data[i].tolist())
        road_data = array(road_data)
        non_road_data = array(non_road_data)
        self._road_data = road_data.T
        self._non_road_data = non_road_data.T
        self._road_data_dim = shape(road_data)[0]
        self._non_road_data_dim = shape(non_road_data)[0]

    def predict(self, new_point, smoothing):
        Class_1 = self.activation(new_point, self._road_data, smoothing)/self._road_data_dim
        Class_0 = self.activation(new_point, self._non_road_data, smoothing)/self._non_road_data_dim
        if Class_1 >= Class_0:
            Class = 1
        else:
            Class = 0
        return Class

    def activation(self, new_point, data, smoothing):
        return sum(exp((dot(new_point, data)-1)/(smoothing**2))/self._training_dim)

prob_nn = Prob_Neural_Network(roaddata)
acc = 0.0
road_acc = 0.0
nonroad_acc = 0.0
road_num = 0
nonroad_num = 0
list_non = []
list_road = []
res = 0.0
# smoothing value is critical
smoothing = 2.9
for i in range(roaddata['test_dim']):
    result = prob_nn.predict(roaddata['test_data'][i], smoothing)
    if roaddata['test_target'][i] == 1:
        road_num += 1
    else:
        nonroad_num += 1
    if result == roaddata['test_target'][i]:
        acc += 1
        if roaddata['test_target'][i] == 1:
            road_acc += 1
        else:
            nonroad_acc += 1


acc = acc/roaddata['test_dim']*100
acc = acc[0][0].item()
road_acc = road_acc/road_num*100
nonroad_acc = nonroad_acc/nonroad_num*100
avg_acc = (nonroad_acc+road_acc)/2
print('smoothing value is', smoothing)
print('overall acc is %.2f' % acc)
print('acc for road is %.2f' % road_acc)
print('acc for non road is %.2f' % nonroad_acc)
print('avg acc is %.2f ' % avg_acc)
