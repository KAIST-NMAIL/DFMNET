import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import Dataset
from torch.utils.data import DataLoader as torchDataLoader
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error as SkMSE
from scipy.io import savemat
from copy import deepcopy

import os, errno
from time import localtime, strftime


from dataset.DataLoader import DataLoader
torch.cuda.set_device(0)

class DiabetesDataset(Dataset):
    def __init__(self, x_data, y_data):
        self.len = x_data.shape[0]
        self.x_data = x_data
        self.y_data = y_data

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len

class Saver():
    def __init__(self, postfix):
        self.path = os.getcwd()
        self.rootpath = os.path.join(self.path, 'temp_result')
        times = strftime("%y%m%d_%H%M%S", localtime())
        postfix = postfix + "_" + times
        self.save_path = os.path.join(self.rootpath, postfix)
        self.makeSaveFolder(self.save_path)

    def makeSaveFolder(self, save_location):
        if not os.path.exists(save_location):
            try:
                os.makedirs(save_location)
            except OSError as e:
                if e.errno != errno.EEXIST:
                    raise

    def testPostProcess(self, data_loader, y_data, y_pred, tag):
        y_data = data_loader.forwardKinematics(y_data, tag)
        y_pred = data_loader.forwardKinematics(y_pred, tag)

        self.saveResult(y_data,
                        y_pred,
                        data_loader.getHipPosition(tag),
                        tag)

    def saveResult(self, y_data, y_pred, hip_data, tag):
        file_name = tag + "_result.mat"
        save_location = os.path.join(self.save_path, file_name)
        dic_dataset = {
            'gt_raw': y_data,
            'pred_raw': y_pred,
            'hip_quaternion': hip_data[:, :4],
            'hip_position': hip_data[:, 4:]
        }
        savemat(save_location, dic_dataset)


class LSTMR(nn.Module):
    def __init__(self, n_input, n_output):
        super(LSTMR, self).__init__()

        self.n_input = n_input
        self.n_output = n_output

        self.n_lstm_layer = 2
        self.n_lstm_hidden = 128

        self.n_hidden_1 = 128
        self.n_hidden_2 = 128
        self.n_hidden_3 = 128
        self.n_hidden_4 = 128
        self.n_hidden_5 = 128

        self.lstm = nn.LSTM(input_size=self.n_input,
                            hidden_size=self.n_lstm_hidden,
                            num_layers=self.n_lstm_layer,
                            dropout=0.5)

        self.bodyNet1 = nn.Sequential(
            nn.Linear(self.n_lstm_hidden + self.n_input, self.n_hidden_1),
            nn.ReLU(),
            nn.Linear(self.n_hidden_1, self.n_hidden_2),
            nn.ReLU(),
            nn.Linear(self.n_hidden_2, self.n_hidden_3),
            nn.ReLU())

        self.decoder = nn.Sequential(
            nn.Linear(self.n_hidden_3, self.n_hidden_4),
            nn.ReLU(),
            nn.Linear(self.n_hidden_4, self.n_hidden_5),
            nn.ReLU(),
            nn.Linear(self.n_hidden_5, self.n_output)
        )

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal(m.weight.data)

    def forward(self, x):
        xlstm, _ = self.lstm(x.transpose(1, 0))
        xlstm = xlstm[-1, :, :]
        xbody = self.bodyNet1(torch.cat([xlstm, x[:, -1, :]], 1))
        xbody = self.decoder(xbody)
        return xbody


if __name__ =='__main__':
    batch_size = 10
    n_epoch = 30
    lr = 0.001
    window_size = 120
    data_loader = DataLoader(window_size)

    dataset_range = np.array(range(63)) + 1


    train_dataset = data_loader.getTrainDataSet()

    x_data = train_dataset[0]
    y_data = train_dataset[1]

    x_data = torch.from_numpy(x_data)
    y_data = torch.from_numpy(y_data)


    n_input = 20
    n_output = 39

    model = LSTMR(n_input, n_output).cuda()

    train_history = []

    train_dataset_loader = torchDataLoader(dataset=DiabetesDataset(x_data, y_data),
                                           batch_size=batch_size,
                                           shuffle=True,
                                           drop_last=False
                                           )
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # training
    for epoch in range(n_epoch):
        for i, data in enumerate(train_dataset_loader, 0):
            model.train()

            x_data = Variable(data[0], requires_grad=False).type(torch.cuda.FloatTensor)
            y_data = Variable(data[1], requires_grad=False).type(torch.cuda.FloatTensor)

            optimizer.zero_grad()
            y_pred = model(x_data)
            loss = criterion(y_pred, y_data)
            loss.backward()
            optimizer.step()

        model.eval()
        train_error = loss.cpu().data.numpy()[0]
        train_history.append(np.sqrt(train_error))
        if epoch % 10 == 0:
            print("Epoch: %04d" % epoch,
                  " train_cost: ", "{:.9f}".format(train_error))


    # test
    model.eval()
    saver = Saver("")
    testResultCase1RMSE = dict()
    testResultCase1STD = dict()

    gts = dict()
    preds = dict()

    for tag in data_loader.dataset_tags:
        dataset = data_loader.getTestDataSet(tag)
        x_data = dataset[0]
        y_data = dataset[1]

        x_data = Variable(torch.from_numpy(x_data), requires_grad=False).type(torch.cuda.FloatTensor)
        y_pred = model(x_data).cpu().data.numpy()

        gts[tag] = y_data
        preds[tag] = y_pred

        saver.testPostProcess(data_loader, y_data, y_pred, tag)
