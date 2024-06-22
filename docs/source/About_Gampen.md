# About GaMPEN

## Why was GaMEPN developed?
Although Convolutional Neural Networks (CNNs) have been used for galaxy morphology determination for quite some time now, a few challenges had persisted. 

Most previously developed CNNs provided broad morphological classifications; and there had been very limited work on estimating structural parameters of galaxies or associated uncertainties using CNN. Even popular non-machine learning tools like Galfit severely underestimate uncertainties by values as high as ∼75%. 

The computation of full Bayesian posteriors for these structural parameters is crucial for drawing scientific inferences that account for uncertainty and are indispensable in the derivation of robust scaling relations or tests of theoretical models using morphology.

One other challenge of using CNNs in astronomy, is the necessity to use fixed cutout sizes. Many practitioners choose to use a large cutout size for which "most galaxies" would remain in the frame. However, this means that typical cutouts contain other galaxies in the frame, often leading to less accurate results. Thus, this becomes a bottleneck when applying CNNs to galaxies over a wide magnitudes or redshifts.

In order to address these above challenges, we developed GaMPEN:-

1. GaMPEN estimates posterior distributions for (user-selected) structural parameters of galaxies.

    * GaMPEN's predicted posteriors are **extremely well-calibrated and accurate ($\lesssim 5\%$ derivation)**. They have been shown to be **upto $\sim60\%$ more accurate compared to uncertainties predicted by light-profile fitting algorithms.**

    * GaMPEN takes into account both aleatoric & epistemic uncertainties.

    * GaMPEN incorporates the full covariance matrix in its loss function allowing it to achieve well-calibrated uncertainties for all output parameters.

2. GaMPEN automatically crops input images to an optimal size before determining their morphology.
    *  Due to GaMPEN's design this step requires no additional training step; except the training to predict structural parameters. 


## What Parameters and Surveys can GaMPEN be Used for?
The publicly released GaMPEN models are turned to predict specific parameters for specific surveys. For example, our [Hyper Suprime-Cam (HSC) models](./Public_data.md#hsc-wide-pdr2-galaxies) can be used to estimate the bulge-to-total light ratio, effective radius, and flux of HSC galaxies till $z < 0.75$.

However, GaMPEN models can be trained from scratch to determine **any combination of parametric and non-parametric structural parameters** (e.g., Sersic Index, Concentration, Asymmetry, etc.) for **any space or ground-based imaging survey**. The only catch is if you are trying to do predictions on a survey other than the ones for which we have released models; or you trying to predict other alternative parameters; you will need to train your GaMPEN model from scratch.

 Don't hesitate to contact us if you want our help/advice in training a GaMPEN model for your survey/parameters! 

## More Details About GaMPEN

### GaMPEN's Architecture

![GaMPEN architecture](../assets/GaMPEN_architecture.png "Architecture of GaMPEN")

GaMPEN's architecture consists of two separate entities:-
 * an upstream Spatial Transformer Network (STN) which enables GaMPEN to automatically crop galaxies to an optimal size;
 * a downstream Convolutional Neural Network (CNN) which enables GaMPEN to predict posterior distributions for various morphological parameters.

GaMPEN's design is based on our previously successful classification CNN, [GaMorNet](https://gamornet.readthedocs.io/en/latest/), as well as as different variants of the Oxford Visual Geometry Group networks. We tried a variety of different architectures before finally converging on this design.

### GaMPEN's Posteriors/Uncertainties
To predict posteriors, GaMMPEN takes into account both aleatoric and epistemic uncertainties. It uses the negative log-likelihood of the output parameters as the loss function combined with the Monte Carlo Dropout technique. GaMPEN also incorporates the full covariance matrix in the loss function, using a series of algebraic manipulations.

The uncertainties/posteriors produced by GaMPEN have been shown to be extremely well-calibrated ($\lesssim 5\%$ deviation. As shown in [Ghosh et. al. 2022b](https://arxiv.org/abs/2212.00051) this represents a significant improvement over state-of-the-art light profile fitting tools which underestimate uncertainties by $\sim15\%-60\%$ depending on the brightness of the source. 

### Predictional Stabiltily Against Rotational Transformations
The video below shows the stability of predictions made by trained GaMPEN HSC models when an input galaxy image is rotated through various angles. As can be seen, GaMPEN's predictions of all three output parameters are fairly stable against rotations.

![Rotational Transformation](./../assets/real_data_gampen_video.gif "Rotational Transformation")


## Publications

GaMPEN was initially introduced in 2022 in this [ApJ paper](https://iopscience.iop.org/article/10.3847/1538-4357/ac7f9e). 

Since then, GaMPEN has been used in a number of other publications. We always try to maintain an updated record of GaMPEN's trained models and catalogs produced [on this page](http://gampen.ghosharitra.com/)


## Attribution Info.

Please cite the below mentioned publication if you make use of GaMPEN or some code herein.

```tex
   @article{Ghosh2022,
   author = {Aritra Ghosh and C. Megan Urry and Amrit Rau and Laurence Perreault-Levasseur and Miles Cranmer and Kevin Schawinski and Dominic Stark and Chuan Tian and Ryan Ofman and Tonima Tasnim Ananna and Connor Auge and Nico Cappelluti and David B. Sanders and Ezequiel Treister},
   doi = {10.3847/1538-4357/ac7f9e},
   issn = {0004-637X},
   issue = {2},
   journal = {The Astrophysical Journal},
   month = {8},
   pages = {138},
   title = {GaMPEN: A Machine-learning Framework for Estimating Bayesian Posteriors of Galaxy Morphological Parameters},
   volume = {935},
   year = {2022},
   }
```

Additionally, if you are using publicly released GaMPEN models or catalogs for a specific survey, please cite the relevant publication(s) in which the data was released. For example, if you are using the GaMPEN HSC models, please cite [this article](https://arxiv.org/abs/2212.00051).

## License

Copyright 2022 Aritra Ghosh, Amrit Rau & contributors

Made available under a [GNU GPL v3.0](https://github.com/aritraghsh09/GaMPEN/blob/master/LICENSE) license. 


## Getting Help/Contributing

We always welcome contributions to GaMPEN! If you have any questions about using GaMPEN, please feel free to send me an e-mail at this ``aritraghsh09@xxxxx.com`` GMail address.

If you have spotted a bug in the code/documentation or you want to propose a new feature, please feel free to open an issue/a pull request on [GitHub](https://github.com/aritraghsh09/GaMPEN).