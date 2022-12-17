# Using GaMPEN
On this page, we go over the most important user-facing functions that GaMPEN has and what each of these functions do. Read this page carefully to understand the various arguments/options that can be set while using these functions. 

## Running the trainer

The backbone of GaMPEN's training feature is the `GaMPEN/ggt/train/train.py`. This script is meant to be invoked from the command line while passing in the appropriate arguments. 

```{eval-rst}  

:py:mod:`ggt.train.train`
=========================

Functions
~~~~~~~~~

.. py:function:: train(**kwargs)

   Runs the training procedure using MLFlow.

```
### Parameters:

* **experiment_name** (*str*; default=`"demo"`) - MLFlow variable which controls the name of the experiment in MLFlow.

* **run_id** (*str*; default=`None`) - An MLFlow variable. This only needs to be used if you are resuming a previosuly run experiment, and want the information to be logged under the previous run.

* **run_name** (*str*; default=`None`) - The name assigned to the MLFlow run. A run is supposed to be a sub-class of an experiment in MLFLow. Typically you will have multiple runs (e.g., using multiple hyper-parameters) within an experiment.

* **model_type** (*str*; default=`"vgg16_w_stn_oc_drp"`) - The type of model you want to train. For most purposes, if you are trying to use GaMPEN as it was originally used, you should use `vgg16_w_stn_oc_drp`. We recommend referring to the source [source code](https://github.com/aritraghsh09/GaMReN/blob/trial_network/ggt/models) for information about the other options.
    * **ggt** 
    * **vgg16** 
    * **ggt_no_gconv**
    * **vgg16_w_stn**
    * **vgg16_w_stn_drp**
    * **vgg16_w_stn_drp_2**
    * **vgg16_w_stn_at_drp**
    * **vgg16_w_stn_oc_drp**

* **model_state** (*str*; default=`None`) - The path to a previosuly saved model file. This only needs to be set when you want to start training from a previously saved model.

* **data_dir** (*str*; required variable)- The path to the data directory containing the `info.csv` file and the `cutouts/` folder.

* **split_slug** (*str*; required variable) - This specifies which data-split is used from the `splits` folder in `data_dir`. For split, the options are `balanced`, `unbalanced` and for slug the options are `xs`, `sm`, `md`, `lg`, `xl`, `dev`, and `dev2`. Refer to the [Make Splits](#make-splits) section for more information.

* **target_metrics** (*str*; default=`"bt_g"`) - Enter the column names of `info.csv` that you want the model to learn/predict separated by a comma. For example, if you want the model to predict the `R_e` and `bt` columns, then you should enter `R_e,bt`. 

* **loss** (*str*; default=`"aleatoric_cov"`) - This can be set to the following options:-
    * **mse**
    * **aleatoric**
    * **aleatoric_cov**  

    For most purposes when you want full posterior distributions similar to the original GaMPEN results, you should use `aleatoric_cov`. The aleatoric covariance loss is the full loss function is given by 

    $$ - \log \mathcal{L} \propto  \sum_{n} \frac{1}{2}\left[\boldsymbol{Y}_{n}-\boldsymbol{\hat{\mu}}_{n}\right]^{\top} \boldsymbol{\hat{\Sigma_n}}^{-1}\left[\boldsymbol{Y}_{n}-\boldsymbol{\hat{\mu}}_{n}\right] + \frac{1}{2} \log [\operatorname{det}(\boldsymbol{\hat{\Sigma_n}})] $$

    where $Y_n$ is the target variable (values passed in `info.csv`); $\boldsymbol{\hat{\mu}}_n$ and $\boldsymbol{\hat{\Sigma}}_n$ are the mean and covariance matrix of the multivariate Gaussian distribution predicted by GaMPEN for an image. For an extended derivation of the loss function, please refer to [Ghosh et. al. 2022](https://iopscience.iop.org/article/10.3847/1538-4357/ac7f9e). 

    The `aleatoric_loss` option implements a similar loss function as above, but instead of using the full covariance matrix, it uses only the diagonal elements.

    The `mse` options implements a standard mean squared error loss function.


* **expand_data** (*int*; default=`1`) - This controls the factor by which the training data is augmented. For example, if you set this to 2, and you have 1000 images in the training set, then the training set will be expanded to 2000 images. This is useful when you want to train the model on a larger dataset. 

    If you are using this option, then you should also set the `transform` option to `True`. This will ensure that the images are passed through a series of random transformations during training.

* **cutout_size** (*int*; default=`167`) - This variable is used to set the size of the input layer of the GaMPEN model. This should be set to the size of the cutouts that you are using; otherwise you will get size-mismatch errors.

    If you have cutouts that vary in size, you must ensure that all the cutouts are bigger than this `cutout_size` value, and you could use the `crop` option to crop the cutouts to this `cutot_size` value before being fed into the model.

* **channels** (*int*; default=`3`) - This variable is used to set the number of channels in the input layer of the GaMPEN model. Since GaMPEN's original based CNN is based on a VGG-16 pre-trained model, this variable is set to 3 by default.

* **n_workers** (*int*; default=`4`) - The number of workers to be used during the
data loading process. You should set this to the number of CPU threads you have available. 

* **batch_size** (*int*; default=`16`) - The batch size to be used during training the model. This variable specifies how many images will be processed in a single batch. This is a hyperparameter and must be tuned. The default value is a good starting point. While tuning this value, we recommend changing this by a factor of 2 each time.

* **epochs** (*int*; default=`40`) - The number of epochs you want to train GaMPEN for. Each epoch refers to all the training images being processed through the network once. You will need to optimize this value based on how much the loss function is decreasing with the number of epochs. The default value is a good starting point.

* **lr** (*float*; default=`5e-7`) - This is the learning rate to be used during the training process. This is a hyperparameter that should be tuned during the training process. The default value is a good starting point.  While tuning this value, we recommend changing this by an order of magnitude each time. 

* **momentum** (*float*; default=`0.9`) - The value of the momentum to be used in the gradient descent optimizer that is used to train the model. This must always be $\geq0$. This accelerates the gradient descent process. This is a hyperparameter that should be tuned during the training process. The default value is a good starting point. For tuning, we recommend trying out values between 0.8 and 0.99.

* **weight_decay** (*float*; default=`0`) - This represents the value you want to set for the L2 regularization term. This is a hyperparameter that should be tuned during the training process. The default value is a good starting point. For tuning we recommend starting from `1e-5` and increasing/decreasing by an order of magnitude each time.

    The `weight_decay` value is simply passed to the PyTorch [SGD optimizer](https://pytorch.org/docs/stable/generated/torch.optim.SGD.html) 

* **parallel/ no-parallel** (*bool*; default=`True`) - The parallel argument controls whether or not to use multiple GPUs during training when they are available. 

   ```{note}
   The above notation (which is used for other arugments as well) implies that if you pass is `--parallel` then the `parallel` argument is set to `True`. If you pass `--no-parallel` then the `parallel` argument is set to `False`. 
    ```

* **normalize/no-normalize** (*bool*; default=`True`) - The normalize argument controls whether or not, the loaded images will be normalized using the `arsinh` function. 

* **label_scaling** (*str*; default=`"std"`) - The label scaling option controls whether to standardize the training labels or not. Set this to `std` for [sklearn's `StandardScaling()`](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html) and `minmax` for [sklearn's `MinMaxScaler()`](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html). This should usually always be set to `std` especially when using multiple target variables.

* **transform/no-transform** (*bool*; default=`True`) - If True, the training images are passed through a series of sequential random transformations:-

    * First the images are cropped to the `cutout_size` value.
    * Then the images are flipped horizontally with a $50\%$ probability.
    * The the images are vertically flipped with a $50\%$ probability.
    * Finally a random rotation is applied to the image with any value between 0 and 360 degrees.

    All the above transformations are performed using the [kornia library](https://kornia.readthedocs.io/en/latest/index.html). 

* **crop/no-crop** (*bool*; default=`True`) - If True, all images are passed through a cropping operation before being fed into the network. Images are cropped
to the `cutout_size` parameter.

* **nesterov/no-nesterov** (*bool*; default=`False`) - Whether to use Nesterov momentum or not. This variable is simply passsed to the PyTorch [SGD optimizer](https://pytorch.org/docs/stable/generated/torch.optim.SGD.html). This is a hyperparameter that should be tuned during the training process.

* **repeat_dims/no-repeat_dims** (*bool*; default=`True`) - When you have a multichannel network and you are feeding in images with only one channel, you should set this parameter to `True`. This automatically repeats the image as many times as the number of channels in the network, while data-loading. 

* **dropout_rate** (*float*; default=`None`) - The dropout rate to use for all the layers in the model. If this is set to None, then the default dropout rate in the specific model is used.

    ```{caution}
    The `dropout_rate` is an important hyperparameter that among other things, also controls the predicted epistemic uncertainity when using Monte Carlo Dropout. This hyperparameter should be tuned in order to achieve callibrated coverage probabilities.

    We recommend tuning the `dropout_rate` once you have tuned all other hyperparameters. Refer to [Ghosh et. al. 2022](https://iopscience.iop.org/article/10.3847/1538-4357/ac7f9e) and [Ghosh et. al. 2022b](https://arxiv.org/abs/2212.00051) for more details on how we tuned this hyperparameter. 

    ```


## Make Splits

```{eval-rst}  
:py:mod:`ggt.data.make_splits`
==============================

.. py:module:: ggt.data.make_splits


Functions
~~~~~~~~~
   
.. py:function:: interleave(L)


.. py:function:: make_splits(x, weights, split_col=None)


.. py:function:: main(data_dir, target_metric)

   Generate train/devel/test splits from the dataset provided.

```

 * **split_types** - 
     * xs - train=0.027, devel=0.003, test=0.970
     * sm - train=0.045, devel=0.005, test=0.950
     * md - train=0.090, devel=0.010, test=0.900
     * lg - train=0.200, devel=0.050, test=0.750
     * xl - train=0.450, devel=0.050, test=0.500
     * dev - train=0.70, devel=0.15, test=0.15

### Parameters:
* **data_dir** - Path to the directory where data is stored

* **target_metric** (str) - Generate train/devel/test splits from the dataset provided.

### Returns:
```
csv files that contain the splitted training, devel, test dataset 
at the location `..data/data_dir/splits`. 
```

## Inference

```{eval-rst}
:py:mod:`ggt.modules.inference`
===============================

.. py:module:: ggt.modules.inference

Functions
~~~~~~~~~~

.. py:function:: predict(model_path, dataset, cutout_size, channels, parallel=False, batch_size=256, n_workers=1, model_type='ggt', n_out=1, mc_dropout=False, dropout_rate=None)

   Using the model defined in model path, return the output values for
   the given set of images


.. py:function:: main(model_path, output_path, data_dir, cutout_size, channels, parallel, slug, split, normalize, batch_size, n_workers, label_cols, model_type, repeat_dims, label_scaling, mc_dropout, dropout_rate, transform, errors, cov_errors, n_runs, ini_run_num)

```
### Parameters

* **model_type**  - Same as the model types mentioned in Running the Trainer section previously. If using our pre-trained models, this should be set to `vgg16_w_stn_oc_drp`.

* **model_path** - The full path to the trained `.pt` model file which you want to use for performing prediction.

* **output_path** - The full path to the output directory where the predictions of the model will be stored.

* **data_dir** - The full path to the data directory that should contain a `cutouts` folder with all the images that you want to perform predictions on as well as an `info.csv` file that contains the filenames for all the images. For more information on how to create this directory structure during performing inference, plese refer to the [Predictions Tutorial](https://gampen.readthedocs.io/en/latest/Tutorials.html#making-predictions)

* **cutout_size** (*int*) - Size of the input image that the model takes as input. For our pre-trained models, this should be set to 239, 143, 96 for the low, mid, and high redshift models respectively.

* **channels** (*int*) - Number of channels in the input image. For our pre-trained models, this should be set to 3.

* **slug** (*str*) - This specifies which slug (balanced/unbalanced xs, sm, lg, dev, dev2) is used to perform predictions on. Each slug refers to a different way to split the data into train, devel, and test sets. For more information on the fraction of data assigned to the train/deve/test sets for each slug, please refer to the [`make_splits`](https://gampen.readthedocs.io/en/latest/Using_GaMPEN.html#make-splits) function and the [``make_splits.py`` file](https://github.com/aritraghsh09/GaMPEN/blob/master/ggt/data/make_splits.py).

    If you are performing predictions on a dataset for which you don't have access to the ground truth labels (and thus you haven't run `make_splits`), this should be set to `None` as shown in the [Predictions Tutorial](https://gampen.readthedocs.io/en/latest/Tutorials.html#making-predictions).

* **split** (*str*) - The split of the data that you want to perform predictions on. This should be set to `test` if you are performing predictions on the test set. If you are performing predictions on the train or devel set, this should be set to `train` or `devel` respectively.

    If you are performing predictions on a dataset for which you don't have access to the ground truth labels (and thus you haven't run `make_splits`), this should be set to `None` as shown in the [Predictions Tutorial](https://gampen.readthedocs.io/en/latest/Tutorials.html#making-predictions).

* **normalize/no-normalize** (*bool*) - The normalize argument controls whether or not, the loaded images will be normalized using the `arsinh` function. 

* **label_scaling** (*str*) - The label scaling option controls whether to
standardize the labels or not. Set this to `std` for sklearn's
`StandardScaling()` and `minmax` for sklearn's `MinMaxScaler()`.
This is especially important when predicting multiple
outputs. For more information, visit [here](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html).
    
    Note that you should pass the same argument for
`label_scaling` as was used during the training phase (of the
model being used for inference). For all our pre-trained models, this should be set to `std`."

* **batch_size** (*int*) - The batch size to be used during inference. This specfies how many images will be processed in a single batch. During inference, the only consideration is to keep the batch size small enough so that the batch can be fit within the memory of the GPU.

* **n_workers** (*int*) - The number of workers to be used during the
data loading process. You should set this to the number of threads you have access to. 

* **parallel/ no-parallel** (*bool*) - The parallel argument controls whether or not to use multiple GPUs when they are available. 

    Note that this variable needs to be set to whatever value was used during the training phase (of the model being used for inference). For all our pre-trained models, this should be set to `parallell`

* **label_cols** (*str*) - Enter the label column(s) separated by commas. Note that you should pass the exactly same argument for label_cols as was used during the training phase (of the model being used for inference)

* **repeat_dims/no-repeat_dims** (*bool*) - In case of multi-channel data, whether to repeat a two dimensional image as many times as the number of channels. Note that you should pass the exactly same argument for repeat_dims as was used during the training phase (of the model being used for inference). For all our pre-trained models, this should be set to `repeat_dims`

* **mc_dropout/no-mc_dropout** (*bool*) - Turns on Monte Carlo dropout during inference. For most cases, this should be set to `mc_dropout`.

* **n_runs** (*int*) - The number of different models that will be generated using Monte Carlo dropout and used for infererence. 

* **ini_run_num** (*int*) - Specifies the starting run-number for `n_runs`. For example, if `n_runs` is set to 5 and `ini_run_num` is set to 10, then the output csv files will be named as `inf_10.csv`, `inf_11.csv`, `inf_12.csv`, `inf_13.csv`, `inf_14.csv`.

* **dropout_rate** - This should be set to the dropout rate used during training.

* **transform/no-transform** - If `True`, the images are passed through a cropping transformation to ensure proper cutout size. This should be left on for most cases.

* **errors/no-errors** - If True and if the model allows for it, aleatoric uncertainties are written to the output file. Only set this to True if you trained the model with `aleatoric` loss. 

* **cov_errors/no-cov_errors** - If True and if the model allows for it, aleatoric uncertainties with full covariance conisdered are written to the output file. Only set this to True if you trained the model with `aleatoric_cov` loss. For our pre-trained models, this should be set to `cov_errors`.

* **labels/no-labels** - If True, this means you have labels available for the dataset. If False, this means that you have no labels available and want to perform predictions on a dataset for which you don't know the ground truth labels.

    This primarily used to control which files are used to perform scaling the prediction variables. If `--no-labels`, then you need to specify the data directory and slug that should be used to perform the scaling. If `--labels`, then the  `scaling_data_dir` and `scaling_slug` are automatically set to values for `data_dir` and `slug` provided before. 

* **scaling_data_dir** - The data directory that should be used to perform unscaling of the prediction variables. You should only set this if using `--no-labels`. Refer to the [Predictions Tutorial](https://gampen.readthedocs.io/en/latest/Tutorials.html#making-predictions) for a demonstration. 

* **scaling_slug** - This specifies which slug (balanced/unbalanced, xs, sm, lg, dev) corresponding to the scaling_data_dir is used to perform the data scaling on. You should only set this if using `--no-labels`. Refer to the [Predictions Tutorial](https://gampen.readthedocs.io/en/latest/Tutorials.html#making-predictions) for a demonstration.


### Returns
```
csv files containing predicted output
```

## Result Aggregator

```{eval-rst}
:py:mod:`ggt.modules.result_aggregator`
===============================

.. py:module:: ggt.modules.result_aggregator

Functions
~~~~~~~~~~

.. py:function:: main(data_dir,num,out_summary_df_path,out_pdfs_path, unscale,scaling_df_path, drop_old)

```

```{attention}
The result aggregator module also converts flux
to magnitudes. However, this conversion is only
valid for HSC. If you are using the module for
some other survey, please alter the relevant line in the `unscale_preds` function of `result_aggregator.py` or ignore the mangitudes produced by the `result_aggregator` module.
```

### Parameters

* **data_dir**  - Full path to the directory that has the prediction csv files that need to be aggregated.

* **num** - The number of prediction csv files that need to be aggregated.

* **out_summary_df_path** - Full path to the output csv file that will contain the summary statistics. 

* **out_pdfs_path** - Full path to the output directory that will contain the pdfs of the prediction distributions.

* **unscale/no-unscale** - If True, the predictions are unscaled using the information `scaling_df_path`. This unscaling is for the inverse logit and logarithmic tansformations.
cd
* **scaling_df_path** - Full path to the `info.csv` file that contains the scaling information. This is only used if `unscale` is set to True. Refer to the [Predictions Tutorial](https://gampen.readthedocs.io/en/latest/Tutorials.html#making-predictions) for a demonstration.

* **drop_old/no-drop_old** - If True, the unscaled prediction columns will be dropped.




