# Dynamic Barycenter Averaging Kernel in RBF Networks for Time Series Classification
The code in this repository for the paper "[Dynamic Barycenter Averaging Kernel in RBF Networks for Time Series Classification](https://ieeexplore.ieee.org/abstract/document/8684820)" accepted by IEEE Access.



## Abstract

Radial basis function (RBF) network has been utilized in many applications due to its simpletopological  structure  and  strong  capacity  on  function  approximation.   The  core  of  RBF  network  is  itsstatic kernel function, which is based on the Euclidean distance and cannot obtain good performance for time series classification (TSC) due to the time-shift invariance, complex dynamics and different length of temporal data. We proposed a new temporal kernel, namely, the Dynamic Barycenter Averaging Kernel (DBAK) and introduced it into RBF network.  DBAK is based on altered Gaussian DTW (AGDTW). First, we combine k-means clustering with a dynamic time warping (DTW) based averaging algorithm called DTW barycenter averaging (DBA) to determine the center of DBAK. Then, in order to facilitate the stable gradient-training process in the whole network, a normalization term is added into the kernel formulation. By integrating the information of the whole time warping path, our DBAK based RBF network (DBAK-RBF) performs efficiently for TSC tasks.



## Usage

- `DBAKRBF/* ` the source code of DBAK-RBF network and components analysis in the paper. It is based on [RBF Network MATLAB Code](http://mccormickml.com/2013/08/16/rbf-network-matlab-code/). the explanation is as follows
  - `DBAKRBF/costFunctionRBFN.m` compute cost and gradients for DBAK-RBF network
  - `DBAKRBF/exp_trainRBFN.m` the training process of DBAK-RBF network
  - `DBAKRBF/RBF_calcDtw.m` calculate DTW distance between two points
  - `DBAKRBF/RBF_calcAGDTWKernel.m` same as `RBF_calcDtw` but use altered Gaussian DTW (AGDTW) instead of DTW
   - `DBAKRBF/testRBFN.m` Computes output of DBAK-RBF Network for the provided input
- `fmin_adam/fmin_adam.m` Function Adam optimiser, with matlab calling format. Adam is an implementation of the Adam optimisation algorithm (gradient descent with Adaptive learning rates individually on each parameter, with momentum) from [Adam: A Method for Stochastic Optimization](https://arxiv.org/abs/1412.6980) and is designed to work on stochastic gradient descent problems. We use it to train our model.
- `kMeans/*` kmeans clustering use to calucate the center of RBF network, also from [RBF Network MATLAB Code](http://mccormickml.com/2013/08/16/rbf-network-matlab-code/)
- `DBA.m` DBA is an averaging method that is consistent with Dynamic Time Warping. In our work, this technique
  will be used for estimating the kernel’s centers. The code of DBA is from [DBA](https://github.com/fpetitjean/DBA)
- `DBAKRBF.m` main function of our DBAK-RBF network
- `readData.m` read data from datasets



## Reference

```
@article{shi2019dynamic,
  title={Dynamic Barycenter Averaging Kernel in RBF Networks for Time Series Classification},
  author={Shi, Kejian and Qin, Hongyang and Sima, Chijun and Li, Sen and Shen, Lifeng and Ma, Qianli},
  journal={IEEE Access},
  volume={7},
  pages={47564--47576},
  year={2019},
  publisher={IEEE}
}
```

