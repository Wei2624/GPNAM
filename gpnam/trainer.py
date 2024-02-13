import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from os.path import join as pjoin
import time
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score
from sklearn.metrics import mean_squared_error
import math
from scipy.sparse.linalg import cg

class Trainer(nn.Module):
    def __init__(self,
                 model,
                 data,
                 experiment_name=None,
                 optimizer="CG",
                 optimizer_params={},
                 n_epochs=300,
                 lr=0.01,
                 batch_size=256,
                 problem='regression',
                 verbose=False,
                 n_last_checkpoints=5,
                 display_freq=1):
        super().__init__()
        self.model = model
        self.verbose = verbose
        self.lr = lr
        self.data = DataLoader(data, batch_size=batch_size, shuffle=True)
        # self.test_data = DataLoader(test, batch_size=batch_size, shuffle=False)
        self.n_epochs = n_epochs
        self.optimizer_name = optimizer
        self.display_freq = display_freq

        params = [p for p in self.model.parameters() if p.requires_grad]
        self.opt = self.construct_opt(optimizer, lr, params, optimizer_params)
        if self.optimizer_name != 'CG':
            self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.opt, lr_lambda=lambda epoch: 1/math.sqrt(epoch+1))

        self.step = 0
        self.n_last_checkpoints = n_last_checkpoints
        self.problem = problem

        if problem == 'classification':
            self.loss_function = (
                lambda x, y: F.binary_cross_entropy_with_logits(x, y.float()) if x.ndim == 1
                else F.cross_entropy(x, y)
            )
        elif problem == 'regression':
            self.loss_function = (lambda y1, y2: F.mse_loss(y1.float(), y2.float()))
        else:
            raise NotImplementedError()

        if experiment_name is None:
            experiment_name = 'untitled_{}.{:0>2d}.{:0>2d}_{:0>2d}:{:0>2d}'.format(*time.gmtime()[:5])
            if self.verbose:
                print('using automatic experiment name: ' + experiment_name)

        self.experiment_path = pjoin('logs/', experiment_name)

    def train(self, device = None):
        if not device:
            device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        self.model.to(device)

        if self.optimizer_name != 'CG':
            for epoch in range(self.n_epochs):
                running_loss = 0
                performane = 0
                for i_batch, sample_batched in enumerate(self.data):
                    x, y = sample_batched['x'].to(device), sample_batched['y'].to(device)

                    self.opt.zero_grad()

                    if self.problem == 'classification':
                        y_pred = self.model(x)
                    else:
                        y_pred, _ = self.model(x)

                    loss = self.loss_function(y_pred, y)
                    loss.backward()

                    self.opt.step()

                    running_loss += loss.item()
                    if self.problem == 'classification':
                        performane += (torch.round(torch.sigmoid(y_pred)) == y).float().sum() # calculate accuracy
                    else:
                        performane += torch.sqrt(((y_pred - y)**2).mean()).item()


                self.scheduler.step()

                if (epoch + 1) % self.display_freq == 0:
                    to_display = {
                        'epoch': str(epoch) + '/' + str(self.n_epochs),
                        'loss': round(running_loss, 4),
                        'learning rate': round(self.scheduler.get_lr()[0],5)
                    }
                    if self.problem == 'classification':
                        to_display['accuracy'] = performane.cpu().numpy() / len(self.data.dataset)
                    else:
                        to_display['RMSE'] = performane
                    self.display(to_display)



                # print('loss at ', epoch, ' : ', running_loss)
                # print('accu at ', epoch, ' : ', correct / len(self.data.dataset))
        else:
            A = []
            b = []
            for i_batch, sample_batched in enumerate(self.data):
                x, y = sample_batched['x'].to(device), sample_batched['y'].to(device)

                _, rff_mapping = self.model(x)
                A.append(torch.transpose(rff_mapping, 0, 1) @ rff_mapping)
                b.append(torch.transpose(rff_mapping, 0, 1) @ y.float())

            A = sum(A).cpu().numpy()
            temp = A[-1,-1]
            A = A + np.identity(A.shape[0])/20
            A[-1,-1] = temp

            b = sum(b).cpu().numpy()

            w = self.opt(A, b)[0]

            self.model.w = nn.Parameter(torch.from_numpy(w).float())

            print("Conjugate gradient optimization is done!")



    def evaluate(self, test_data, device=None):
        if not device:
            device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        self.model.to(device)
        self.model.eval()
        test_data = DataLoader(test_data, batch_size=self.batch_size, shuffle=False)

        y_pred = []
        y_truth = []
        with torch.no_grad():
            for i_batch, sample_batched in enumerate(test_data):
                x, y = sample_batched['x'].to(device), sample_batched['y'].to(device)

                if self.problem == 'classification':
                    y_pred.append(torch.sigmoid(self.model(x)).cpu().numpy())
                else:
                    y_pred.append(self.model(x)[0].cpu().numpy())
                y_truth.append(y.cpu().numpy())

            y_pred = np.hstack(y_pred)
            y_truth = np.hstack(y_truth)

            if self.problem == 'classification':
                auc = roc_auc_score(y_truth, y_pred)
                print('AUC: ', auc)
            else:
                mse = mean_squared_error(y_truth, y_pred)
                print('RMSE:', math.sqrt(mse))

    def display(self, info_dict):
        info = ''
        for key, val in info_dict.items():
            info += key + ' : ' + str(val) + ' '
        print(info)


    def construct_opt(self, optimizer, lr, params, optimizer_params):
        if not optimizer:
            return torch.optim.Adam(params, lr=lr, **optimizer_params)
        if optimizer == "Adam":
            return torch.optim.Adam(params, lr=lr, **optimizer_params)
        if optimizer == "SGD":
            return torch.optim.SGD(params, lr=lr, momentum=0.9, dampening=0.9)
            # return torch.optim.SGD(params, lr=lr)
        if optimizer == "CG":
            return cg


