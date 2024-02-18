import gpnam
from gpnam.download_datasets import DATASETS
import sys
import pandas as pd
import math
from sklearn.metrics import roc_auc_score
from sklearn.metrics import mean_squared_error

if __name__ == "__main__":
    if len(sys.argv) <= 1:
        print("You have to specify which dataset to reproduce. Current options are: LCD, CAHousing")
        sys.exit()

    if sys.argv[1] == 'LCD':
        problem = DATASETS[sys.argv[1]]()
        data = pd.read_csv('datasets/LCD/train_data.csv', header=None)
        target = pd.read_csv('datasets/LCD/train_label.csv', header=None).values
        test_data = pd.read_csv('datasets/LCD/test_data.csv', header=None)
        test_target = pd.read_csv('datasets/LCD/test_label.csv', header=None).values

        model = gpnam.sklearn.GPNAM(data.shape[1], problem, optimizer="Adam")
        model.fit(data, target)

        pred = model.predict(test_data)

        print('AUC: ', roc_auc_score(test_target, pred))
    elif sys.argv[1] == 'CAHousing':
        problem = DATASETS[sys.argv[1]]()
        data = pd.read_csv('datasets/CAHousing/train_data.csv', header=None)
        target = pd.read_csv('datasets/CAHousing/train_label.csv', header=None).values
        test_data = pd.read_csv('datasets/CAHousing/test_data.csv', header=None)
        test_target = pd.read_csv('datasets/CAHousing/test_label.csv', header=None).values

        kernel_width = data.std(axis=0).values/24.0

        model = gpnam.sklearn.GPNAM(data.shape[1], problem, kernel_width=kernel_width)
        model.fit(data, target)

        pred = model.predict(test_data)

        print('RMSE: ', math.sqrt(mean_squared_error(test_target, pred)))
    else:
        print('Dataset are not available yet. ')