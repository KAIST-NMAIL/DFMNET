from copy import deepcopy
import os

from time import localtime
from time import strftime
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader as torchDataLoader
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error as SkMSE
from sklearn.base import BaseEstimator, RegressorMixin
from tensorboardX import SummaryWriter

from Utils.trainUtils import *
from dataset.DataLoader import DataLoader, DiabetesDataset

INPUT_DIM = 20
OUTPUT_DIM = 39
CUDA_ID = 0

# times = strftime("%y%m%d_%H%M%S", localtime())
# SAVE_PATH = os.path.join(os.getcwd(), 'logdir')
# makeFolder(SAVE_PATH)

if torch.cuda.is_available():
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(CUDA_ID)
    DEVICE = 'cuda'
else:
    DEVICE = 'cpu'


class DFMNET(nn.Module, BaseEstimator, RegressorMixin):
    def __init__(self, n_input: int, n_output: int, n_lstm_layer=2, n_lstm_hidden=128, n_KDN_hidden=128, lr=0.001,
                 n_epochs=100, batch_size=16, writer=None):
        super().__init__()
        self.n_input = n_input
        self.n_output = n_output
        self.n_lstm_layer = n_lstm_layer
        self.n_lstm_hidden = n_lstm_hidden
        self.n_KDN_hidden = n_KDN_hidden
        self.lr = lr
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.writer = writer

        self.n_hidden_1 = n_KDN_hidden
        self.n_hidden_2 = n_KDN_hidden
        self.n_hidden_3 = n_KDN_hidden
        self.n_hidden_4 = n_KDN_hidden
        self.n_hidden_5 = n_KDN_hidden

        self.train_x = None
        self.train_y = None

        self.SEN = None
        self.KDN = None
        self.criterion = None
        self.optimizer = None
        self.valid_data = False

        self._build_model()
        self.to(DEVICE)

        self._set_optimizer()

    def _build_model(self):
        self.SEN = nn.LSTM(
            input_size=self.n_input,
            hidden_size=self.n_lstm_hidden,
            num_layers=self.n_lstm_layer,
            dropout=0.5
        )

        self.KDN = nn.Sequential(
            nn.Linear(self.n_lstm_hidden + self.n_input, self.n_hidden_1),
            nn.ReLU(),

            nn.Linear(self.n_hidden_1, self.n_hidden_2),
            nn.ReLU(),

            nn.Linear(self.n_hidden_2, self.n_hidden_3),
            nn.ReLU(),

            nn.Linear(self.n_hidden_3, self.n_hidden_4),
            nn.ReLU(),

            nn.Linear(self.n_hidden_4, self.n_hidden_5),
            nn.ReLU(),

            nn.Linear(self.n_hidden_5, self.n_output)
        )

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)

    def _set_optimizer(self):
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.parameters(), lr = self.lr)

    def forward(self, x):
        r, _ = self.SEN(x.transpose(1, 0))

        r = r[-1, :, :]
        r = torch.cat([r, x[:, -1, :]], 1)

        y = self.KDN(r)

        return y


    def fit(self, X: np.ndarray, y: np.ndarray, X_valid=None, y_valid=None):

        # preprocessing
        self.train_x = torch.from_numpy(train_x).type(torch.float).to(DEVICE)
        self.train_y = torch.from_numpy(train_y).type(torch.float).to(DEVICE)

        train_dataset_loader = torchDataLoader(dataset=DiabetesDataset(self.train_x, self.train_y),
                                               batch_size=self.batch_size,
                                               shuffle=True,
                                               drop_last=False)

        if (X_valid is not None) and (y_valid is not None):
            self.valid_data = True
        else:
            pass

        for epoch in tqdm(range(self.n_epochs)):
            for i, (x, y) in enumerate(train_dataset_loader, 0):
                self.train()
                self.optimizer.zero_grad()
                y_pred = self(x)
                loss = self.criterion(y_pred, y)
                loss.backward()

                # optimizer mode
                if self.valid_data:
                    pass
                else:
                    self.optimizer.step()

            self.writer.add_scalar('train/train_loss', loss.item(), epoch )

        return self


    def fit_transform(self, X: np.ndarray, y: np.ndarray, X_valid=None, y_valid=None):
        self.fit(X, y, X_valid, y_valid)
        return self.predict(X)

    def predict_proba(self, X: np.ndarray):
        return self

    def predict(self, X):
        self.eval()
        with torch.no_grad():
            y = self(X)
        return y


if __name__ == '__main__':
    writer = SummaryWriter()
    loader = DataLoader()

    dfmnet = DFMNET(INPUT_DIM, OUTPUT_DIM, writer=writer)

    train_x, train_y = loader.getStandardTrainDataSet()
    dfmnet.fit(train_x, train_y)

    writer.close()