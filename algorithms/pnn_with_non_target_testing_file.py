__author__ = 'Guanhua, Joms'
from numpy import array, shape, sum, exp, dot, int8, zeros
from scipy.io import loadmat
import argparse, pickle, os
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('train_path', help='Path to training database')
arguments = parser.parse_args()
roaddata = loadmat(arguments.train_path)


class ProbNeuralNetwork(object):
    def __init__(self, roaddata):
        self._training_data = roaddata['training_data']
        self._training_target = roaddata['training_target']
        self._training_dim = roaddata['training_dim']
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


def computation(input_fldr, file_heading, smoothing):
    path = os.path.join(input_fldr, "%s_final_data.mat" % file_heading)
    data_mat = loadmat(path)
    for i in range(data_mat['total_pic']):
        data = data_mat['feature_vector'][0][i]
        pkl_path = os.path.join(input_fldr, '%s/' % file_heading, "%s_%06d.pkl" % (file_heading, i))
        with open(pkl_path, 'rb') as input:
            label_join = pickle.load(input)
        target = zeros(data_mat['dim'][i])
        result = []
        for j in range(shape(data)[0]):
            result.append(prob_nn.predict(data[j], smoothing))
        for key, val in label_join.items():
            if result[val] == 1:
                target[key[0]][key[1]] = (130, 0, 0)
            else:
                target[key[0]][key[1]] = (0, 0, 130)
        target = target.astype(int8)
        img_path = os.path.join(input_fldr, '%s/' % file_heading, "%s_%06d_predict.png" % (file_heading, i))
        plt.imshow(target)
        plt.savefig(img_path)


prob_nn = ProbNeuralNetwork(roaddata)
computation('c:/data_road/testing/', 'um', 2.9)  # smoothing value is critical
computation('c:/data_road/testing/', 'umm', 2.9)  # takes about 7 min to process
computation('c:/data_road/testing/', 'uu', 2.9)