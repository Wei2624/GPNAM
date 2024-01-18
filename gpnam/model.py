import numpy as np
import torch
import torch.nn as nn
import math
from scipy.stats import norm

class GPNAM(nn.module):
    def __init__(self, input_dim, kernel_width=0.2, rff_num_feat=100):
        """
        Build GPNAM model.
        :param kernel_width: kernel width of RFF approximation
        :param rff_num_feat: RFF dimension
        :param input_dim: the dimensions of input data
        """
        super(GPNAM, self).__init__()
        self.kernel_width = kernel_width
        self.rff_num_feat = rff_num_feat
        self.input_dim = input_dim

        self.c = 2*math.pi*torch.rand(rff_num_feat,input_dim)/rff_num_feat
        self.Z = torch.from_numpy(norm.cdf([each/(rff_num_feat+1) for each in range(1, rff_num_feat+1)]))

        self.c.requires_grad = False
        self.Z.requires_grad = False

        self.w = nn.linear(input_dim*rff_num_feat+1,1)

    def forward(self, x):
        rff_mapping = torch.sqrt(2/self.rff_num_feat)*torch.cos(torch.einsum('i,pq -> piq', self.Z, x)/self.kernel_width + self.c)
        rff_mapping = rff_mapping.view(x.shape[0],-1)
        pred = self.w(rff_mapping)

        return pred


