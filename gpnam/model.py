import numpy as np
import torch
import torch.nn as nn
import math
from scipy.stats import norm

class GPNAMClass(nn.Module):
    def __init__(self, input_dim, kernel_width=0.2, rff_num_feat=100):
        """
        Build GPNAM classification model.
        :param kernel_width: kernel width of RFF approximation
        :param rff_num_feat: RFF dimension
        :param input_dim: the dimensions of input data
        """
        super(GPNAMClass, self).__init__()
        self.kernel_width = kernel_width
        self.rff_num_feat = rff_num_feat
        self.input_dim = input_dim

        self.c = 2*math.pi*torch.rand(rff_num_feat,input_dim).sort(dim=0)[0]/rff_num_feat
        self.Z = torch.from_numpy(norm.ppf([each/(rff_num_feat+1) for each in range(1, rff_num_feat+1)])).float()
        # self.c = torch.ones(rff_num_feat, input_dim) # for debug only

        self.c.requires_grad = False
        self.Z.requires_grad = False

        self.w = nn.Parameter(torch.zeros(input_dim*rff_num_feat + 1, 1), requires_grad=True)

    def forward(self, x):
        rff_mapping = math.sqrt(2/self.rff_num_feat)*torch.cos(torch.einsum('i,pq -> piq', self.Z, x)/self.kernel_width + self.c)
        rff_mapping = torch.transpose(rff_mapping,1,2).reshape(x.shape[0],-1)
        rff_mapping = torch.column_stack((rff_mapping, torch.ones(rff_mapping.shape[0]))).float()


        pred = rff_mapping @ self.w

        if pred.dim() == 2 and pred.shape[1] == 1: pred = torch.squeeze(pred, 1)

        return pred

class GPNAMReg(nn.Module):
    def __init__(self, input_dim, kernel_width=0.2, rff_num_feat=50):
        """
        Build GPNAM regression model.
        :param kernel_width: kernel width of RFF approximation
        :param rff_num_feat: RFF dimension
        :param input_dim: the dimensions of input data
        """
        super(GPNAMReg, self).__init__()
        self.kernel_width = kernel_width
        self.rff_num_feat = rff_num_feat
        self.input_dim = input_dim

        self.c = 2*math.pi*torch.rand(rff_num_feat,input_dim).sort(dim=0)[0]/rff_num_feat
        self.Z = torch.from_numpy(norm.ppf([each/(rff_num_feat+1) for each in range(1, rff_num_feat+1)])).float()
        # self.c = torch.ones(rff_num_feat,input_dim) # for debug only

        self.c.requires_grad = False
        self.Z.requires_grad = False

        self.w = nn.Parameter(torch.zeros(input_dim*rff_num_feat + 1, 1), requires_grad=True)
        # self.b = nn.Parameter(torch.zeros(1), requires_grad=True)

    def forward(self, x):
        rff_mapping = math.sqrt(2/self.rff_num_feat)*torch.cos(torch.einsum('i,pq -> piq', self.Z, x)/self.kernel_width + self.c)
        rff_mapping = torch.transpose(rff_mapping, 1, 2).reshape(x.shape[0],-1)
        rff_mapping = torch.column_stack((rff_mapping,torch.ones(rff_mapping.shape[0]))).float()

        pred = rff_mapping @ self.w

        if pred.dim() == 2 and pred.shape[1] == 1: pred = torch.squeeze(pred, 1)

        return pred, rff_mapping




