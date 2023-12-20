# GPNAM: Gaussian Process Neural Additive Models

![The framework of GPNAM](./imgs/framework.jpg)

This repository contains the source code for the paper Gaussian Process Neural Additive Models that appears at AAAI 2024. 

Basically, the GPNAM constructs a GP as the shape function for each input feature, which leads to a convex optimization with a significant reduction in trainable parameters. 

## Data sets preparation

You can download the data sets locally:
```
python download_datasets.py LCD GMSC
```


