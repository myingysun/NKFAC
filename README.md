# NKFAC: A Fast and Stable KFAC Optimizer for Deep Neural Networks

The hyperparameters adopted for different optimizers in our paper are provided below to reproduce the results reported.

## Learning rate (LR) and weight decay (WD) on CIFAR100/10.
Optimizer |SGDM |AdamW |RAdam |Adabelief | KFAC | SKFAC |NKFAC
---|:--:|:--:|:--:|:--:|:--:|:--:|---:
LR| 0.1 |  0.001 | 0.001 | 0.001 | 0.0005 | 0.0005 | 0.05 
WD| 0.0005 | 0.5 | 0.5 | 0.5 | 0.1 | 0.1 | 0.001 

## Learning rate (LR) and weight decay (WD) on ImageNet.
Optimizer |SGDM |AdamW |RAdam |Adabelief | KFAC | SKFAC |NKFAC
---|:--:|:--:|:--:|:--:|:--:|:--:|---:
LR|  0.1 | 0.001 | 0.001 | 0.001 | 0.0005 | 0.0005 | 0.05
WD |0.0001 | 0.1 |  0.1 | 0.5  | 0.02 | 0.02 | 0.0002 

You are also welcome to try NKFAC and AdaNKFAC in the MMDetection toolbox to reproduce our results on COCO datasets.
