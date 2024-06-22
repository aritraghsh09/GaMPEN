```{toctree}
:maxdepth: 2

index
Getting_Started
Tutorials
Using_GaMPEN
Public_data
```

```{attention}
Note that although GaMPEN's current documentation is fairly substantive, we are still working on some parts of the documentation and some Tutorials. If you run into issues while trying to use GaMPEN, please contact us! We will be more than happy to help you!
```

# About GaMPEN

![Introductory Image](./../assets/gampen_intro_crop.png)

***

The Galaxy Morphology Posterior Estimation Network (GaMPEN) is a Bayesian machine learning framework that can estimate robust posteriors (i.e., values + uncertainties) for structural parameters of galaxies. As the above image shows, GaMPEN also automatically crops input images to an optimal size before structural parameter estimation.

:::{admonition} Feature Highlight
:class:tip

GaMPEN's predicted posteriors are extremely well-calibrated ($<5\%$ deviation) and have been shown to be up to $\sim 60\%$ more accurate compared to the uncertainties predicted by many
light-profile fitting algorithms. 

Once trained, it takes GaMPEN less than a millisecond to perform a single model evaluation on a CPU. Thus, GaMPEN's posterior prediction capabilities are ready for large galaxy samples expected from upcoming large imaging surveys, such as Rubin-LSST, Euclid, and NGRST.
:::

## First Steps with GaMPEN
0. For a quick blog-esque introduction to the most important features of GaMPEN, please check out [this page](http://www.astro.yale.edu/aghosh/gampen.html). For a deep-dive, please refer to [Ghosh et. al. 2022](https://doi.org/10.3847/1538-4357/ac7f9e) and [Ghosh et. al 2023](https://doi.org/10.3847/1538-4357/acd546).
1. Follow the installation instructions and quick-start guide in [Getting Started](./Getting_Started.md).
2. Go through the [Tutorials](./Tutorials.md) to learn how to use GaMPEN for a variety of different tasks.
3. Review the [Using GaMPEN](./Using_GaMPEN.md) page to dive into the details about the various user-facing functions that GaMPEN provides.

Note that if you want to access the publicly released GaMPEN models or structural parameter catalogs for specific surveys (e.g., Hyper Suprime-Cam), please refer to the [Public Catalogs & Trained Models](./Public_data.md) page.


## What Parameters and Surveys can GaMPEN be Used for?

The publicly released GaMPEN models are turned to predict specific parameters for specific surveys. For example, our Hyper Suprime-Cam (HSC) models can be used to estimate the bulge-to-total light ratio, effective radius, and flux of HSC galaxies till $z < 0.75$.

**However, GaMPEN models can be trained from scratch to determine any combination of morphological parameters** (even different from the ones mentioned above -- e.g. Sersic Index) **for any space or ground-based imaging survey**. Please check out our [FAQs](./FAQs.md) page for our recommendations if you want to train a GaMPEN model tuned to a specific survey. Also, don't hesitate to contact us if you want our help/advice in training a GaMPEN model for your survey/parameters.

## More Details About GaMPEN

### GaMPEN's Architecture
GaMPEN consists of a two sequential neural network modules -- a Spatial Transformer Network (STN) and a Convolutional Neural Network (CNN). The image below shows the detailed architecture of both these networks. Note that both the networks are trained simulataneously using the same loss function and optimizer. For further details about the architecture, please refer to [Ghosh et. al. 2022](https://iopscience.iop.org/article/10.3847/1538-4357/ac7f9e) or the `vgg16_w_stn_oc_drp.py` file in the `GaMPEN/ggt/models/` directory.


![GaMPEN architecture](../assets/GaMPEN_architecture.png "Architecture of GaMPEN")

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


