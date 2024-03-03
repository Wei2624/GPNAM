# GPNAM: Gaussian Process Neural Additive Models
[![arXiv](https://img.shields.io/badge/arXiv-2402.12518-b31b1b.svg?style=plastic)](https://arxiv.org/abs/2402.12518) <img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="image" height="18px">

![The framework of GPNAM](./imgs/framework.jpg)
*The framework of GPNAM. `z_s, c_s` and the sinusoidal function are predefined from the paper and do not require training. The only trainable parameter is `W` that maps to the output of each shape function.*

This repository contains the source code for the paper Gaussian Process Neural Additive Models that appears at AAAI 2024. 

Basically, the GPNAM constructs a Neural Additive Model (NAM) by a GP with Random Fourier Features as the shape function for each input feature, which leads to a convex optimization with a significant reduction in trainable parameters. 


## Sklearn interface

You can install the package by:
```
pip install gpnam
```

Then, you can run the model simply by:
```python
from gpnam.sklearn import GPNAM

"""
input_dim: the dimensions of input data
problem: type of the task, 'classification' or 'regression'
"""
gpnam = GPNAM(input_dim, problem)
gpnam.fit(X, y)

y_pred = gpnam.predict(X_test)
```

### Use Sklearn to reproduce
You can reproduce the results of LCD data set in the paper by:
```commandline
python reproduce_results_sklearn.py LCD
```
or the results of CAHousing data set:
```commandline
python reproduce_results_sklearn.py CAHousing
```

## Build on your own

### Setup
1. Create a new environment with `Python>=3.7` by:
```commandline
conda create -n gpnam python=3.7
conda activate gpnam
```
2. install the required packages by:
```commandline
pip install -r requirements.txt
```

### Data sets preparation

You can download the data sets locally:
```
python ./gpnam/download_datasets.py LCD GMSC CAHousing
```

### Experiments

Then, you can run the experiment:
```
python main.py --dataset LCD --optimizer Adam --n_epochs 200
```

### Matlab
**Note**: Statistics and Machine Learning Toolbox is required.


## Citation
If you find this repo useful, please consider citing our paper:
```
@article{zhang2024gaussian,
  title={Gaussian Process Neural Additive Models},
  author={Zhang, Wei and Barr, Brian and Paisley, John},
  journal={arXiv preprint arXiv:2402.12518},
  year={2024}
}
```