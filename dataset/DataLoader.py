import numpy as np
import quaternion
import sys, os
from copy import deepcopy

from scipy.io import loadmat

import torch
from torch.autograd import Variable
from torch.utils.data import Dataset as torchDataset


class DataLoader():
    def __init__(self, window_size=120):
        self.window_size = window_size
        self.dataset_tags = ['BR', 'SQ', 'WM']
        self.file_names = ['BnR.mat', 'squat.mat', 'windmill.mat']
        self.dataset = dict()

        self.load_dataset()

    def load_dataset(self):
        for idx, tag in enumerate(self.dataset_tags):
            file_name = os.getcwd() + '/dataset/' + self.file_names[idx]
            self.dataset[tag] = loadmat(file_name)

    def generateWindow(self, x_raw, y_raw, window_size):
        x_data = []
        n_data = x_raw.shape[0]

        for idx in range(n_data-window_size):
            x_data.append(
                x_raw[idx:idx+self.window_size].reshape(1, window_size, -1)
            )
        
        x_data = np.concatenate(x_data)
        y_data = y_raw
        return x_data, y_data


    def getTrainDataSet(self):
        x_data = []
        y_data = []
        for tag, data in self.dataset.items():
            x, y = self.generateWindow(data['train_x'], data['train_y'], self.window_size)
            x_data.append(x)
            y_data.append(y)
        return np.concatenate(x_data, 0), np.concatenate(y_data, 0)

    def getTestDataSet(self, tag):
        x_data = self.dataset[tag]['test_x']
        y_data = self.dataset[tag]['test_y']
        return self.generateWindow(x_data, y_data, self.window_size)

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
        if torch.cuda.is_available():
            dtype = torch.cuda.FloatTensor
        else:
            dtype = torch.FloatTensor

        self.len = x_data.shape[0]
        self.x_data = torch.from_numpy(x_data).type(dtype)
        self.y_data = torch.from_numpy(y_data).type(dtype)

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len
