# Using GaMPEN
On this page, we go over the most important user-facing functions that GaMPEN has and what each of these functions do. Read this page carefully to understand the various arguments/options that can be set while using these functions. 


## Make Splits

```{eval-rst}  
:py:mod:`ggt.data.make_splits`
==============================

.. py:module:: ggt.data.make_splits


Functions
~~~~~~~~~

.. py:function:: main(data_dir, target_metric)

   Generates train/devel/test splits from the dataset provided.

```

This `GaMPEN/ggt/data/make_splits.py` script slices and dices the data in `info.csv` in a bunch of different ways to create a lot of options for training, testing, and devel (validation) sets. All these splits of the `info.csv` file are stored in a `splits/` folder within the parent `data_dir` directory.

First, this will create two types of splits:- `balanced` and `unbalanced`. In the `unbalanced` splits, objects are picked randomly from the dataset for each split without any constraint. This function also creates `balanced` splits, where it first splits the dataset into 4 partitions based on the `target_metric` variable; and then draws samples such that the samples used for trianing are balanced across these 4 partitions. 

Finally, for both `balanced` and `unbalanced` split, a number of different sub-splits are created with different fractions of data assigned to the training, devel, and test sets. The fractions assigned to each partition for the various types are mentioned below:-

 * **split_types** - 
     * xs - train=0.027, devel=0.003, test=0.970
     * sm - train=0.045, devel=0.005, test=0.950
     * md - train=0.090, devel=0.010, test=0.900
     * lg - train=0.200, devel=0.050, test=0.750
     * xl - train=0.450, devel=0.050, test=0.500
     * dev - train=0.70, devel=0.15, test=0.15
     * dev2 - train=0.700, devel=0.050, test=0.250

You can change these or definte your own splits. Simply alter `split_types` dictionary at the top of the `GaMPEN/ggt/data/make_splits.py` file.

### Parameters:
* **data_dir** (*str*, required variable) - Path to the directory where the `info.csv` file is stored

* **target_metric** (*str*, default=`"bt_g"`) - Used for creating the `balanced` splits. This is the name of the column in the `info.csv` file that you want to use for creating the `balanced` splits.



## Running the trainer

The backbone of GaMPEN's training feature is the `GaMPEN/ggt/train/train.py`. This script is meant to be invoked from the command line while passing in the appropriate arguments. 

```{eval-rst}  

:py:mod:`ggt.train.train`
=========================

.. py:module:: ggt.train.train

Functions
~~~~~~~~~

.. py:function:: train(**kwargs)

   Trains GaMPEN models using passed arguments.

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
   The above notation (which is used for other arugments as well) implies that if you pass `--parallel` then the `parallel` argument is set to `True`. If you pass `--no-parallel` then the `parallel` argument is set to `False`. 
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

    ```{attention}
    The `dropout_rate` is an important hyperparameter that among other things, also controls the predicted epistemic uncertainity when using Monte Carlo Dropout. This hyperparameter should be tuned in order to achieve callibrated coverage probabilities.

    We recommend tuning the `dropout_rate` once you have tuned all other hyperparameters. Refer to [Ghosh et. al. 2022](https://iopscience.iop.org/article/10.3847/1538-4357/ac7f9e) and [Ghosh et. al. 2022b](https://arxiv.org/abs/2212.00051) for more details on how we tuned this hyperparameter. 

    ```



## Inference

```{eval-rst}
:py:mod:`ggt.modules.inference`
===============================

.. py:module:: ggt.modules.inference

Functions
~~~~~~~~~~

.. py:function:: main(model_path, output_path, data_dir, cutout_size, channels, parallel, slug, split, normalize, batch_size, n_workers, label_cols, model_type, repeat_dims, label_scaling, mc_dropout, dropout_rate, transform, errors, cov_errors, n_runs, ini_run_num)

```
The `GaMPEN/ggt/modules/inference.py` script provides users the functionality to perform predictions on images using trained GaMPEN models.

### Parameters

* **model_type** (*str*; default=`"vgg16_w_stn_oc_drp"`) - Same as the model types mentioned in [Running the Trainer](#running-the-trainer) section previously. If using our pre-trained models, this should be set to `vgg16_w_stn_oc_drp`.

* **model_path** (*str*; required variable)- The full path to the trained `.pt` model file which you want to use for performing prediction.

    ```{attention}
    The model path should be enclosed in single quotes `'/path/to/model/xxxxx.pt'` and NOT within double quotes `"/path/to/model/xxxxx.pt"`. If you encluse it within double quotes, then the script will throw up an error.
    ```

* **output_path** (*str*; required variable) - The full path to the output directory where the predictions of the model will be stored.

* **data_dir** (*str*; required variable) - The full path to the data directory that should contain a `cutouts` folder with all the images that you want to perform predictions on as well as an `info.csv` file that contains the filenames for all the images. For more information on how to create this directory structure during performing inference, plese refer to the [Predictions Tutorial](https://gampen.readthedocs.io/en/latest/Tutorials.html#making-predictions)

* **cutout_size** (*int*; default=`167`) - Size of the input image that the model takes as input. For our pre-trained models, this should be set to `239`, `143`, `96` for the low, mid, and high redshift models respectively.

* **channels** (*int*; default=`3`) - Number of channels in the input image. For our pre-trained models, this should be set to `3`.

* **slug** (*str*; required variable) - This specifies which slug (balanced/unbalanced xs, sm, lg, dev, dev2) is used to perform predictions on. Each slug refers to a different way to split the data into train, devel, and test sets. For consistent results, you should set this to the same slug that was used to train the model. For more information on the fraction of data assigned to the train/deve/test sets for each slug, please refer to the [`make_splits`](#make-splits) function.

    If you are performing predictions on a dataset for which you don't have access to the ground truth labels (and thus you haven't run `make_splits`), this should be set to `None` as shown in the [Predictions Tutorial](https://gampen.readthedocs.io/en/latest/Tutorials.html#making-predictions).

* **split** (*str*; default=`"test"`) - The split of the data that you want to perform predictions on. This should be set to `test` if you are performing predictions on the test set. If you are performing predictions on the train or devel set, this should be set to `train` or `devel` respectively.

    If you are performing predictions on a dataset for which you don't have access to the ground truth labels (and thus you haven't run `make_splits`), this should be set to `None` as shown in the [Predictions Tutorial](https://gampen.readthedocs.io/en/latest/Tutorials.html#making-predictions).

* **normalize/no-normalize** (*bool*; default=`True`) - The normalize argument controls whether or not, the loaded images will be normalized using the `arsinh` function. This should be set to the same value as what was used during training the model.

* **label_scaling** (*str*; default=`"std"`) - The label scaling option controls whether to perform an inverse-transformation on the predicted values. Set this to `std` for [sklearn's `StandardScaling()`](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html) and `minmax` for [sklearn's `MinMaxScaler()`](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html). This should usually always be set to `std` especially when using multiple target variables.
    
    Note that you should pass the same argument for `label_scaling` as was used during the training phase (of the model being used for inference). For all our pre-trained models, this should be set to `std`."

* **batch_size** (*int*; default=`256`) - The batch size to be used during inference. This specfies how many images will be processed in a single batch. During inference, the only consideration is to keep the batch size small enough so that the batch can be fit within the memory of the GPU.

* **n_workers** (*int*; default=`4`) - The number of workers to be used during the
data loading process. You should set this to the number of threads you have access to. 

* **parallel/ no-parallel** (*bool*; default=`True`) - The parallel argument controls whether or not to use multiple GPUs when they are available. 

    Note that this variable needs to be set to whatever value was used during the training phase (of the model being used for inference). For all our pre-trained models, this should be set to `parallel`

* **label_cols** (*str*; default=`bt_g`) - Enter the label column(s) separated by commas. Note that you should pass the exactly same argument for label_cols as was used during the training phase (of the model being used for inference)

* **repeat_dims/no-repeat_dims** (*bool*; default=`True`) - In case of multi-channel data, whether to repeat a two dimensional image as many times as the number of channels. Note that you should pass the exactly same argument for `repeat_dims` as was used during the training phase (of the model being used for inference). For all our pre-trained models, this should be set to `repeat_dims`

* **mc_dropout/no-mc_dropout** (*bool*; default=`True`) - Turns on Monte Carlo dropout during inference. For most cases, this should be set to `mc_dropout`.

* **n_runs** (*int*; default=`1`) - The number of different models that will be generated using Monte Carlo dropout and used for infererence. 

* **ini_run_num** (*int*; default=`1`) - Specifies the starting run-number for `n_runs`. For example, if `n_runs` is set to 5 and `ini_run_num` is set to 10, then the output csv files will be named as `inf_10.csv`, `inf_11.csv`, `inf_12.csv`, `inf_13.csv`, `inf_14.csv`.

* **dropout_rate** (*float*; default=`None`) - This should be set to the dropout rate that was used while training the model.

* **transform/no-transform** (*bool*; default=`False`) - If `True`, the images are passed through a cropping transformation to ensure proper cutout size. This should be left on for most cases.

   ```{attention}
   Note that if you set this to True and then use cutouts which have a smaller size than the `cutout_size`, this will lead to unpredictable behaviour.
   ```

* **errors/no-errors** (*bool*, default=`False`) - If True and if the model allows for it, aleatoric uncertainties are written to the output file. Only set this to True if you trained the model with `aleatoric` loss. 

* **cov_errors/no-cov_errors** (*bool*, default=`False`) - If True and if the model allows for it, aleatoric uncertainties with full covariance conisdered are written to the output file. Only set this to True if you trained the model with `aleatoric_cov` loss. For our pre-trained models, this should be set to `cov_errors`.

* **labels/no-labels** (*bool*, default=`True`)- If True, this means you have labels available for the dataset. If False, this means that you have no labels available and want to perform predictions on a dataset for which you don't know the ground truth labels.

    This primarily used to control which files are used to perform scaling the prediction variables. If `--no-labels`, then you need to specify the data directory and slug that should be used to perform the scaling. If `--labels`, then the  `scaling_data_dir` and `scaling_slug` are automatically set to values for `data_dir` and `slug` provided before. 

* **scaling_data_dir** (*str*; default=`None`) - The data directory that should be used to perform unscaling of the prediction variables. You should only set this if using `--no-labels`.

    This scaling refers to the `label_scaling` variable that you passed before. Essentially to inverse transform the predictions, we need access to the original scaling parameters that were used to scale the data during trianing. In case you are using a pre-trained model directly on some data for which you have no labels, you need to point this to the `/splits` folder of the data-directory that was used to train the model. For all our pre-trained models, we make the relevant scaling files available. Refer to the [Predictions Tutorial](https://gampen.readthedocs.io/en/latest/Tutorials.html#making-predictions) for a demonstration. 

* **scaling_slug** (*str*; default=`None`) - This needs to be set only if you are using `--no-labels`. This specifies which slug (`balanced/unbalanced`, `xs`, `sm`, `lg`, `dev`,`dev2`) corresponding to the scaling_data_dir that should be used to perform the data scaling on. 

    For example, if you want a `balanced-dev2-train.csv` file in the `scaling_data_dir` , then you should set this to `balanced-dev2`. Refer to the [Predictions Tutorial](https://gampen.readthedocs.io/en/latest/Tutorials.html#making-predictions) for a demonstration.



## Result Aggregator

The `GaMPEN/ggt/modules/result_aggregator.py` module is used to aggregate the prediction `.csv` files generated by the  [inference module](#inference). 

```{attention}
The unsccaling properties of the `result_aggregator` module is mostly useful when you are predicting variables similar to the ones used in [Ghosh et. al. 2022](https://doi.org/10.3847/1538-4357/ac7f9e).

If you are using your own custom scaling of variables (or predicting other variables), then you will need to run the `result_aggregator` module with `--no-unscale` and perform the unscaling of variables yourself. Alternatively, you can also choose to alter the `unscale_preds` function in the `result_aggregator.py` module to suit your needs.
```


```{attention}
The result aggregator module also converts flux to magnitudes. However, this conversion is only valid for HSC. If you are using the module for some other survey, please alter the magnitude conversion line in the `unscale_preds` function of `result_aggregator.py` or ignore the mangitudes produced by the `result_aggregator` module.
```


```{eval-rst}
:py:mod:`ggt.modules.result_aggregator`
===============================

.. py:module:: ggt.modules.result_aggregator

Functions
~~~~~~~~~~

.. py:function:: main(data_dir,num,out_summary_df_path,out_pdfs_path, unscale,scaling_df_path, drop_old)

```

### Parameters

* **data_dir** (*str*; required variable)  - Full path to the directory that has the prediction csv files that need to be aggregated.

* **num** (*int*; default=`500`) - The number of prediction csv files that need to be aggregated.

* **out_summary_df_path** (*str*; required variable) - Full path to the output csv file that will contain the summary statistics. 

* **out_pdfs_path** (*str*; required variable) - Full path to the output directory that will contain the posterior distribution functions (PDFs) of the predicted output variables for each galaxy.

* **unscale/no-unscale** (*bool*, default=`False`) - If `True`, the predictions are unscaled using the information `scaling_df_path`. This unscaling is for the inverse logit and logarithmic tansformations (e.g., converting $\log R_e$ to $R_e$)

    This is only useful if you are using our pre-trained models/you are predicted the same variables as in [Ghosh et. al. 2022](https://doi.org/10.3847/1538-4357/ac7f9e). For all other cases, if you want to set this to `True`, you will need to modify the `unscale_preds` function in `result_aggregator.py` according to the variables you are predicting and the transformations you made to them during training.  

    ```{attention}
    In order to make sure that you are not making a mistake, the module will throw an error if you are using the `--unscale` option and the inference `.csvs` do not have the column names exactly as is expected for our trained models (i.e., `custom_logit_bt`,`ln_R_e_asec`,`ln_total_flux_adus`).

    When you are using using some different scaling, you need to run this script with `--no-unscale` and transform the predictions yourself. Or you can also alter the `unscale_preds` function in `result_aggregator.py` according to your needs.
    ``` 

* **scaling_df_path** (*str*; default=`None`) - Full path to the `info.csv` file that contains the scaling information. This is only used if `unscale` is set to True. 

    This is needed to perform the inverse logit transforamtion. As the logit transformation goes to infinty at the edges of the variable space and we need to perform an approximation. To perform this approximation, we need access to the `info.csv` file that was used during training. We make the `info.csv` files for all our pre-trained models available. Refer to the [Predictions Tutorial](https://gampen.readthedocs.io/en/latest/Tutorials.html#making-predictions) for a demonstration.

* **drop_old/no-drop_old** (*bool*; default=`True`)- If `True`, the unscaled prediction columns will be dropped.



## AutoCrop

```{eval-rst}
:py:mod:`ggt.modules.autocrop`
===============================

.. py:module:: ggt.modules.autocrop

Functions
~~~~~~~~~~

.. py:function:: main(model_type, model_path, cutout_size, channels, n_pred,image_dir, out_dir, normalize, transform, repeat_dims, parallel, cov_errors,errors,)

```
The `GaMPEN/ggt/modules/autocrop.py` script provides users the functionality to perform cropping using a trained GaMPEN model and then save these cropped images as fits files for further analysis.

### Parameters

* **model_type** (*str*; default=`"vgg16_w_stn_oc_drp"`) - Same as the model types mentioned in [Running the Trainer](#running-the-trainer) section previously. If using our pre-trained models, this should be set to `vgg16_w_stn_oc_drp`.

* **model_path** (*str*; required variable)- The full path to the trained `.pt` model file which you want to use for performing prediction.

    ```{attention}
    The model path should be enclosed in single quotes `'/path/to/model/xxxxx.pt'` and NOT within double quotes `"/path/to/model/xxxxx.pt"`. If you encluse it within double quotes, then the script will throw up an error.
    ```
* **cutout_size** (*int*; default=`167`) - Size of the input image that the model takes as input. For our pre-trained models, this should be set to `239`, `143`, `96` for the low, mid, and high redshift models respectively.

* **channels** (*int*; default=`3`) - Number of channels in the input image. For our pre-trained models, this should be set to `3`.

* **n_pred** (*int*; default=`1`) - Number of output variables that were used while training the model.

* **image_dir** (*str*; required variable) - Full path to the directory that contains the images that need to be cropped.

* **out_dir** (*str*; required variable) - Full path to the directory where the cropped images will be saved.

* **normalize/no-normalize** (*bool*; default=`True`) - The normalize argument controls whether or not, the loaded images will be normalized using the `arsinh` function. This should be set to the same value as what was used during training the model.

* **transform/no-transform** (*bool*; default=`True`) - The transform argument controls whether or not, the loaded images will be cropped to the mentioned `cutout_size` while being loaded. This should be set to `True` for most cases.

* **repeat_dims/no-repeat_dims** (*bool*; default=`True`) - In case of multi-channel data, whether to repeat a two dimensional image as many times as the number of channels. Note that you should pass the exactly same argument for `repeat_dims` as was used during the training phase (of the model being used for inference). For all our pre-trained models, this should be set to `repeat_dims`

* **parallel/ no-parallel** (*bool*; default=`True`) - The parallel argument controls whether or not to use multiple GPUs when they are available. 

    Note that this variable needs to be set to whatever value was used during the training phase (of the model being used for inference). For all our pre-trained models, this should be set to `parallel`

* **errors/no-errors** (*bool*, default=`False`) - If True and if the model allows for it, aleatoric uncertainties are written to the output file. Only set this to True if you trained the model with `aleatoric` loss. 

* **cov_errors/no-cov_errors** (*bool*, default=`False`) - If True and if the model allows for it, aleatoric uncertainties with full covariance conisdered are written to the output file. Only set this to True if you trained the model with `aleatoric_cov` loss. For our pre-trained models, this should be set to `cov_errors`.


