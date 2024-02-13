"""This file implements a higher-level GPNAM model that can just call fit(X, y).

The goal is to provide a simple interface for users who just want to use it like::

    >>> model = GPGAM()
    >>> model.fit(X, y)
"""

import os
import time
from torch.utils.data import DataLoader

from .model import GPNAMClass
from .model import GPNAMReg
from .trainer import Trainer
from .data import CustomDataset_sklearn



class GPGAMBase(object):
    """Base class for GPGAM."""
    def __init__(
    	self,
    	input_dim,
    	name=None,
    	preprocessed=True,
    	kernel_width=0.2,
    	rff_num_feat=100,
    	optimizer="CG",
	    optimizer_params={},
	    n_epochs=300,
	    lr=0.01,
	    batch_size=256,
	    problem='regression',
	    objective='rmse',
	    verbose=False,
	    n_last_checkpoints=5,
	    display_freq=1,
	    device=None):

	    assert objective in ['ce_loss', 'rmse'], \
	            'Invalid objective: ' + str(objective)
	    if name is None:
	            name = 'tmp_{}.{:0>2d}.{:0>2d}_{:0>2d}:{:0>2d}'.format(*time.gmtime()[:5])

	    self.name = name
	    self.input_dim = input_dim
	    self.preprocessed = preprocessed
	    self.kernel_width = kernel_width
	    self.rff_num_feat = rff_num_feat
	    self.optimizer = optimizer
	    self.optimizer_params = optimizer_params
	    self.n_epochs = n_epochs
	    self.lr = lr
	    self.batch_size = batch_size
	    self.problem = problem
	    self.objective = objective
	    self.verbose = verbose
	    self.n_last_checkpoints = n_last_checkpoints
	    self.display_freq = display_freq
	    self.device = device

	def fit(X, y):
		"""Train the model.

        Args:
            X (numy array): inputs.
            y (numpy array): targets.
            X_val (pandas dataframe): if set, instead of splitting validation set from the X, it
                uses this X as validation set.
            y_val (numpy array): if set, uses this as validation y.
        """
        train_data = CustomDataset_sklearn(X, y, self.problem, ToTensor())
        train_data = DataLoader(train_data, batch_size=self.batch_size, shuffle=True)

        if self.problem == 'classification':
	        model = GPNAMClass(self.input_dim)
	    elif self.problem == 'regression':
	        model = GPNAMReg(self.input_dim, self.kernel_width)
	    else:
	        raise NotImplementedError()


	    trainer = Trainer(
	    			model,
	    			train_dataset, 
	    			batch_size=self.batch_size, 
	    			problem=self.problem,
	    			optimizer=self.optimizer, 
	    			n_epochs=self.n_epochs)

	   	trainer.train(self.device)
	    


