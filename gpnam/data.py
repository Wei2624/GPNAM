import os
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
from .utils import ToTensor
from sklearn.preprocessing import MinMaxScaler
from .download_datasets import DATASETS

class CustomDataset(Dataset):
    """dataset."""

    def __init__(self, x_file, y_file, problem, transform=None):
        self.data = pd.read_csv(x_file, header=None).astype('float32')
        self.target = pd.read_csv(y_file, header=None).values
        if self.target.ndim == 2 and self.target.shape[1] == 1: self.target = self.target.squeeze(axis=1)

        self.problem = problem

        self.transform = transform

    def __len__(self):
        return self.target.shape[0]

    def __getitem__(self, idx):
        sample = {'x': self.data.iloc[idx].values, 'y': np.asarray(self.target[idx])}

        if self.transform:
            sample = self.transform(sample)

        return sample

    def get_input_dim(self):
        return self.data.shape[1]

    def get_kernel_width(self):
        return self.data.std(axis=0).values/24 #+ (self.data.quantile(0.6).values - self.data.quantile(0.4).values)



def get_dataset(dataset_name):
    root = Path(__file__).parent.parent
    dataset_folder = os.path.join(root, "datasets", dataset_name)

    problem = DATASETS[dataset_name]()



    train_x_file = os.path.join(dataset_folder, "train_data.csv")
    train_y_file = os.path.join(dataset_folder, "train_label.csv")
    test_x_file = os.path.join(dataset_folder, "test_data.csv")
    test_y_file = os.path.join(dataset_folder, "test_label.csv")

    train_dataset = CustomDataset(train_x_file, train_y_file, problem, ToTensor())
    test_dataset = CustomDataset(test_x_file, test_y_file, problem, ToTensor())



    return train_dataset, test_dataset, problem