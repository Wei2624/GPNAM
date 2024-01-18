import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from os.path import join as pjoin, exists as pexists
import time

class Trainer(nn.Module):
    def __init__(self, model, experiment_name=None, optimizer=None, optimizer_params={}, lr=0.01, problem='classification', verbose=False,n_last_checkpoints=5):
        super().__init__()
        self.model = model
        self.verbose = verbose
        self.lr = lr

        params = [p for p in self.model.parameters() if p.requires_grad]
        self.opt = self.construct_opt(params, lr, params, optimizer_params)

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

    def train(self, device=None):
        if not device:
            device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")



    def construct_opt(self, optimizer, lr, params, optimizer_params):
        if not optimizer:
            return torch.optim.Adam(params, lr=lr, **optimizer_params)
        if optimizer == "Adam":
            return torch.optim.Adam(params, lr=lr, **optimizer_params)

