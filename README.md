# GPNAM: Gaussian Process Neural Additive Models

![The framework of GPNAM](./imgs/framework.jpg)
*The framework of GPNAM. `Equation 6,7 and 8` from the paper are predefined and do not require training. The only trainable parameter is `W` that maps to the output of each shape function.*

This repository contains the source code for the paper Gaussian Process Neural Additive Models that appears at AAAI 2024. 

Basically, the GPNAM constructs a Neural Additive Model (NAM) by a GP with Random Fourier Features as the shape function for each input feature, which leads to a convex optimization with a significant reduction in trainable parameters. 

## Data sets preparation

You can download the data sets locally:
```
python download_datasets.py LCD GMSC
```


