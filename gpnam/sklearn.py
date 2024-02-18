"""This file implements a higher-level GPNAM model that can just call fit(X, y).

The goal is to provide a simple interface for users who just want to use it like::

	>>> model = GPNAM()
	>>> model.fit(X, y)
"""

import os
import time
import pandas as pd
import torch
from torch.utils.data import DataLoader

from .model import GPNAMClass
from .model import GPNAMReg
from .trainer import Trainer
from .data import CustomDataset_sklearn
from .utils import process_in_chunks_regressor, process_in_chunks_classifier, check_numpy, sigmoid, ToTensor		 



class GPNAM(object):
	"""Base class for GPNAM."""
	def __init__(
		self,
		input_dim,
		problem,
		name=None,
		preprocessed=True,
		kernel_width=0.2,
		rff_num_feat=100,
		optimizer="CG",
		optimizer_params={},
		n_epochs=300,
		lr=0.01,
		batch_size=256,
		objective='rmse',
		verbose=False,
		n_last_checkpoints=5,
		display_freq=1,
		device=None):

		assert objective in ['ce_loss', 'rmse'], \
				'Invalid objective: ' + str(objective)
		assert optimizer in ['SGD', 'CG', 'Adam'], \
				'Invalid optimizer: ' + str(optimizer)
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
		if not device:
			self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
		else:
			self.device = device


		if self.problem == 'classification':
			self.model_wrapper = GPNAMClassifier(self.input_dim, device=self.device)
		elif self.problem == 'regression':
			self.model_wrapper = GPNAMRegressor(self.input_dim, kernel_width=self.kernel_width, device=self.device)
		else:
			raise NotImplementedError()

	def fit(self, X, y):
		"""Train the model.

		Args:
			X (numy array): inputs.
			y (numpy array): targets.
			X_val (pandas dataframe): if set, instead of splitting validation set from the X, it
				uses this X as validation set.
			y_val (numpy array): if set, uses this as validation y.
		"""
		train_data = CustomDataset_sklearn(X, y, self.problem, ToTensor())
		# train_data = DataLoader(train_data, batch_size=self.batch_size, shuffle=True)

		trainer = Trainer(
					self.model_wrapper.model,
					train_data, 
					batch_size=self.batch_size, 
					problem=self.problem,
					optimizer=self.optimizer, 
					n_epochs=self.n_epochs)

		trainer.train(self.device)

	def predict(self, X):
		return self.model_wrapper.predict(X)

	def predict_prob(self, X):
		assert self.problem == 'classification',\
			'Predict_prob is only valid for classification.'

		return self.model_wrapper.predict_prob(X)


class GPNAMRegressor(object):
	"""Regression class for GPNAM."""
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

		assert objective in ['rmse'], \
				'Invalid objective: ' + str(objective)
		if name is None:
				name = 'tmp_{}.{:0>2d}.{:0>2d}_{:0>2d}:{:0>2d}'.format(*time.gmtime()[:5])

		self.model = GPNAMReg(input_dim, kernel_width)
		self.device = device
		self.batch_size = batch_size

	def predict(self, X):
		"""Predict the data.

		Args:
			X (numy array): inputs.
		"""
		self.model.to(self.device)
		self.model.eval()

		if isinstance(X,(pd.core.frame.DataFrame)):
			X = X.values

		X = torch.from_numpy(X).to(self.device)

		with torch.no_grad():
			prediction = process_in_chunks_regressor(self.model, X, batch_size=self.batch_size)
			prediction = check_numpy(prediction)

		return prediction



class GPNAMClassifier(object):
	"""Regression class for GPNAM."""
	def __init__(
		self,
		input_dim,
		name=None,
		preprocessed=True,
		kernel_width=0.2,
		rff_num_feat=100,
		optimizer="Adam",
		optimizer_params={},
		n_epochs=300,
		lr=0.01,
		batch_size=256,
		problem='classification',
		objective='ce_loss',
		verbose=False,
		n_last_checkpoints=5,
		display_freq=1,
		device=None):

		assert objective in ['ce_loss'], \
				'Invalid objective: ' + str(objective)
		if name is None:
				name = 'tmp_{}.{:0>2d}.{:0>2d}_{:0>2d}:{:0>2d}'.format(*time.gmtime()[:5])

		self.model = GPNAMClass(input_dim)
		self.device = device
		self.batch_size = batch_size

	def predict(self, X):
		"""Predict the data.

		Args:
			X (numy array): inputs.
		"""
		self.model.to(self.device)
		self.model.eval()

		if isinstance(X,(pd.core.frame.DataFrame)):
			X = X.values

		X = torch.from_numpy(X).to(self.device)

		with torch.no_grad():
			logits = process_in_chunks_classifier(self.model, X, batch_size=self.batch_size)
			logits = check_numpy(logits)
		return logits

	def predict_prob(self, X):
		logits = self.predict(X)
		prob = sigmoid_np(logits)
		prob = np.vstack([1. - prob, prob]).T

		return prob





