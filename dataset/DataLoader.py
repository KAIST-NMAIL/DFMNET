from copy import deepcopy
import os

import numpy as np
import quaternion

from scipy.io import loadmat
from torch.utils.data import Dataset as torchDataset
from sklearn.preprocessing import StandardScaler

class DataLoader():
    def __init__(self, window_size=120):
        self.window_size = window_size
        self.dataset_tags = ['BR', 'SQ', 'WM']
        self.dataset = dict()

        self.scaler = None

        self.load_dataset()

    def load_dataset(self):
        for idx, tag in enumerate(self.dataset_tags):
            path = os.path.join(
                os.getcwd(),
                'dataset',
                tag + '.pick'
            )
            with open(path, 'rb') as f:
                self.dataset[tag] = pickle.load(f)

    def generateWindow(self, x_raw,  window_size):
        x_data = []
        n_data = x_raw.shape[0]

        for idx in range(n_data-window_size):
            x_data.append(
                x_raw[idx:idx+self.window_size].reshape(1, window_size, -1)
            )
        
        x_data = np.concatenate(x_data)
        return x_data

    def getTrainDataSet(self):
        x_data = []
        y_data = []
        for tag, data in self.dataset.items():
            x = self.generateWindow(data['train_x'], self.window_size)
            x_data.append(x)
            y_data.append(data['train_y'])
        return np.concatenate(x_data, 0), np.concatenate(y_data, 0)

    def getStandardTrainDataSet(self):
        self.scaler = StandardScaler()

        x_data = []
        y_data = []

        for tag, data in self.dataset.items():
            x_data.append(data['train_x'])
            y_data.append(data['train_y'])

        self.scaler.fit(np.concatenate(x_data, 0))

        x_data = []

        for tag, data in self.dataset.items():
            x = self.scaler.transform(data['train_x'])
            x = self.generateWindow(x, self.window_size)

            x_data.append(x)

        return np.concatenate(x_data, 0), np.concatenate(y_data, 0)

    def getStandardTestDataSet(self, tag):
        if self.scaler is None:
            print("!-- ERROR --!")
            print("Standard scaler was not initialized")
            print("!-- ERROR --!")
            exit(-1)

        x_data = self.dataset[tag]['test_x']
        y_data = self.dataset[tag]['test_y']

        x_data = self.scaler.transform(x_data)

        return self.generateWindow(x_data, self.window_size), y_data

    def getTestDataSet(self, tag):
        x_data = self.dataset[tag]['test_x']
        y_data = self.dataset[tag]['test_y']
        return self.generateWindow(x_data, self.window_size), y_data

    def forwardKinematics(self, y_data, tag):
        hip_data = self.dataset[tag]['test_hip']
        hip_quart = quaternion.as_quat_array(hip_data[:, :4].copy())
        hip_position = hip_data[:, 4:].copy()

        y_data = deepcopy(y_data)
        outs = np.zeros_like(y_data)

        marker_idxs = np.array(range(y_data.shape[1])).reshape(-1, 3)

        for idx in range(y_data.shape[0]):
            rtMat = quaternion.as_rotation_matrix(hip_quart[idx])
            pos = hip_position[idx]
            for marker_idx in marker_idxs:
                outs[idx, marker_idx] = np.matmul(rtMat.T, (y_data[idx, marker_idx])) + pos

        return outs

    def getHipPosition(self, tag):
        hip_data = self.dataset[tag]['test_hip']
        return hip_data[:, 4:].copy()


class DiabetesDataset(torchDataset):
    def __init__(self, x_data, y_data):
        self.len = x_data.size(0)
        self.x_data = x_data
        self.y_data = y_data

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len
