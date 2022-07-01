# About GaMPEN

The **Ga**laxy **M**orphology **P**osterior **E**stimation **N**etwork (GaMPEN) is a novel machine learning framework for estimating the Bayesian posteriors of morphological parameters for arbitrarily large numbers of galaxies. GaMPEN estimates values and uncertainties for a galaxy’s bulge-to-total light
ratio (*L_B/L_T*), effective radius (*R_e*), and flux (*F*). 

To estimate posteriors, GaMPEN uses the Monte Carlo Dropout technique and incorporates the full covariance matrix in its loss function. GaMPEN also uses a Spatial Transformer Network (STN) to automatically crop input galaxy frames to an optimal size before determining their morphology which allows it to be applied to new data without prior knowledge of galaxy size. For more information, refer to .....

![GaMPEN architecture](../assets/GaMPEN_architecture.png)

## First contact with GaMPEN

GaMPEN's user-faced functions have been written in a way so that it’s easy to start using them even if you have not dealt with STNs or convolutional neural networks before. For eg. to perform predictions on an array of SDSS images using our trained models, the following line of code is all you need.

```bash
python ggt/train/train.py \
  --experiment_name='ggt-quickstart' \
  --data_dir='data/sdss/' \
  --split_slug='balanced-lg' \
  --expand_data=1 \
  --batch_size=64 \
  --epochs=40 \
  --lr=0.005 \
  --normalize \
  --transform
```

In order to start using GaMPEN, please first look at the Getting Started section for instructions on how to install GaMPEN. Thereafter, we recommend trying out the Tutorials in order to get a handle on how to use GaMPEN.

Finally, you should have a look at the Public Data Release Handbook for our recommendations on how to use different elements of GaMPEN’s public data release for your own work and the API Documentation for detailed documentation of the different functions in the module.

## Publication & Other Data

## Attribution Info.

## License

## Getting Help/Contributing

If you have a question, please first have a look at the FAQs section. If your question is not answered there, please send me an e-mail at this ............ Gmail address.

If you have spotted a bug in the code/documentation or you want to propose a new feature, please feel free to open an issue/a pull request on [GitHub](https://github.com/aritraghsh09/GaMReN).