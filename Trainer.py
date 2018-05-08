import os, errno
from time import localtime, strftime

from scipy.io import savemat
import numpy as np

import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader as torchDataLoader
from DATASET.DataLoader import DataLoader, DiabetesDataset



class Trainer():
    def __init__(self, args):
        self.args = args
        if torch.cuda.is_available():
            self.dtype = torch.cuda.FloatTensor
            torch.cuda.set_device(self.args.cuda)
        else:
            self.dtype = torch.FloatTensor

        self.save_path = self.getSavePath()
        self.makeSaveFolder(self.save_path)

        self.dataloader = None

        self.pre_train_loss_history = []
        self.pre_validation_loss_history = []
        self.re_train_loss_history = []
        self.re_validation_loss_history = []

    def getSavePath(self):
        pwd = os.getcwd()
        rootpath = os.path.join(pwd, 'Result')
        time = strftime("%y%m%d_%H%M%S", localtime())
        save_path = self.args.model + "_"
        save_path += str(self.args.n_epoch_pre_train) + "_"
        save_path += str(self.args.n_epoch_re_train) + "_"
        save_path += str(self.args.batch_size) + "_"
        save_path += str(self.args.window_size) + "_"
        save_path += str(self.args.lr) + "_"
        save_path += self.args.feature_mode + "_"
        save_path += time
        return os.path.join(rootpath, save_path)

    def makeSaveFolder(self, save_location):
        if not os.path.exists(save_location):
            try:
                os.makedirs(save_location)
            except OSError as e:
                if e.errno != errno.EEXIST:
                    raise

    def loadDataSet(self):
        self.dataloader = DataLoader(self.args.window_size, feature_mode=self.args.feature_mode)

    def loadModel(self):
        if self.args.model == 'LSTMF':
            from MODEL.LSTM.LSTMF import MODEL

        else:
            print("model import error")
            exit(-1)

        self.model = MODEL(self.args, self.dataloader.getDataSetShape())

        if torch.cuda.is_available():
            dtype = torch.cuda.FloatTensor
            model = self.model.cuda()

        else:
            dtype = torch.FloatTensor

        self.model.setOptimizer()



    def train(self, n_epoch, train_dataset_loader, validation_set):
        train_loss = []
        validation_loss = []
        x_dataV, y_dataV = self.convertEvalVariabel(validation_set)

        for epoch in range(n_epoch):
            for i, data in enumerate(train_dataset_loader, 0):
                self.model.train()
                x_data, y_data = self.convertTrainVariable(data)

                self.model.optimizer.zero_grad()

                y_pred = self.model(x_data)
                loss_t = self.model.criterion(y_pred, y_data)

                loss_t.backward()
                self.model.optimizer.step()

            # Validation
            self.model.eval()
            y_pred = self.model(x_dataV)
            loss_e = self.model.criterion(y_pred, y_dataV)

            # save log

            if torch.cuda.is_available():
                loss_t = loss_t.cpu()
                loss_e = loss_e.cpu()

            train_loss.append(loss_t.data.numpy()[0])
            validation_loss.append(loss_e.data.numpy()[0])

            if epoch % 20 == 0:
                print("Epoch: %04d" % epoch,
                      " train_cost: ", "{:.9f}".format(loss_t.data.numpy()[0]),
                      " validation_cost: ", "{:.9f}".format(loss_e.data.numpy()[0]))

        return train_loss, validation_loss


    def preTraining(self):
        train_dataset_loader = torchDataLoader(dataset=DiabetesDataset(*self.dataloader.getPreTrainDataSet()),
                                               batch_size=self.args.batch_size,
                                               shuffle=True,
                                               drop_last=True
                                               )

        self.pre_train_loss_history, self.pre_validation_loss_history = \
            self.train(self.args.n_epoch_pre_train, train_dataset_loader, self.dataloader.getPreValidationDataSet())


    def reTraining(self):
        train_dataset_loader = torchDataLoader(dataset=DiabetesDataset(*self.dataloader.getReTrainDataSet()),
                                               batch_size=self.args.batch_size,
                                               shuffle=True,
                                               drop_last=True
                                               )

        self.re_train_loss_history, self.re_validation_loss_history = \
            self.train(self.args.n_epoch_re_train, train_dataset_loader, self.dataloader.getReValidationDataSet())

    def testVE(self):
        mode = "VE"
        self.model.eval()
        tags = self.dataloader.pre_train_dataset_tags

        for tag in tags:
            dataset = self.dataloader.getTrainEvalDataSet(tag)
            x_data = Variable(torch.from_numpy(dataset[0]), requires_grad=False).type(self.dtype)
            y_pred = self.model(x_data)

            if torch.cuda.is_available():
                y_pred = y_pred.cpu()
            y_pred = y_pred.data.numpy()
            self.testEVPostProcess(dataset[1], y_pred, tag, mode)

    def test(self, mode):
        self.model.eval()
        tags = self.dataloader.test_dataset_tags

        for tag in tags :
            datasets = self.dataloader.getTestDataSet(tag)
            x_data = Variable(torch.from_numpy(datasets[0]), requires_grad = False).type(self.dtype)
            y_pred = self.model(x_data)

            if torch.cuda.is_available():
                y_pred = y_pred.cpu()

            y_pred = y_pred.data.numpy()
            self.testPostProcess(datasets[1], y_pred, tag, mode)


    def convertTrainVariable(self, data):
        return Variable(data[0], requires_grad =False), Variable(data[1], requires_grad=False)

    def convertEvalVariabel(self, data):

        x_data = Variable(torch.from_numpy(data[0]), requires_grad = False).type(self.dtype)
        y_data = Variable(torch.from_numpy(data[1]), requires_grad = False).type(self.dtype)
        return x_data, y_data

    def appendPretrainingHistory(self, train_loss, validation_loss):
        if torch.cuda.is_available():
            train_loss = train_loss.cpu()
            validation_loss = validation_loss.cpu()

        self.pre_train_loss_history.append(train_loss.data.numpy()[0])
        self.pre_validation_loss_history.append(validation_loss.data.numpy()[0])


    def testEVPostProcess(self, y_data, y_pred, tag, mode):
        y_data = self.dataloader.inverseRotationEV(y_data, tag)
        y_pred = self.dataloader.inverseRotationEV(y_pred, tag)

        self.saveResult(y_data,
                        y_pred,
                        self.dataloader.getHipDataSetEV(tag),
                        tag,
                        mode)


    def testPostProcess(self, y_data, y_pred, tag, mode):
        y_data = self.dataloader.inverseRotation(y_data, tag)
        y_pred = self.dataloader.inverseRotation(y_pred, tag)

        self.saveResult(y_data,
                        y_pred,
                        self.dataloader.getHipDataSet(tag),
                        tag,
                        mode)

    def saveParams(self):
        location = os.path.join(self.save_path, 'model_param.txt')
        f = open(location, 'w')

        print(self.model, file=f)
        print("learning rate: " + str(self.args.lr), file=f)
        print("window_size: " + str(self.args.window_size), file=f)
        print("epoch_pre_train: " + str(self.args.n_epoch_pre_train), file=f)
        print("epoch_re_train: " + str(self.args.n_epoch_re_train), file=f)
        print("batch_size: " + str(self.args.batch_size), file=f)
        print("feature_mode: " + self.args.feature_mode, file=f)
        f.close()

    def saveModel(self, mode):
        name = mode + "_model.torch"
        torch_model_location = os.path.join(self.save_path, name)
        torch.save(self.model, torch_model_location)

    def saveLogs(self):
        save_location = os.path.join( self.save_path , 'history_logs.mat')
        log_dict = {
            'pre_train_loss': self.pre_train_loss_history,
            'pre_validation_loss': self.pre_validation_loss_history,
        }

        savemat(save_location, log_dict)


    def saveResult(self, y_data, y_pred, hip_data, tag, mode):
        file_name = tag + "_" + mode + "_result.mat"
        save_location = os.path.join(self.save_path, file_name)
        dic_dataset = {
            'gt_raw': y_data,
            'pred_raw' : y_pred,
            'hip_quaternion' : hip_data[:, :4],
            'hip_position': hip_data[:,4:]
        }
        savemat(save_location, dic_dataset)
