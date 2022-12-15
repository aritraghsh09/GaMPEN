# Using GaMPEN
We describe various user oriented functions and the information about various parameters needed to execute them. 

## Running the trainer

This section contains the information regarding various click options required to run the user oriented function `train.py`. 
```{eval-rst}  
:py:mod:`ggt.train.train`
=========================

.. py:module:: ggt.train.train


Functions
~~~~~~~~~

.. autoapisummary::

   ggt.train.train.train


.. py:function:: train(**kwargs)

   Runs the training procedure using MLFlow.

```
****kwargs** - Parameters passed into the `ggt.models.train.train` base function. 

### Parameters:

* **experiment_name** (*str*) - An experiment name should be assigned which provides significant insight about the experiment. 
* **run_id** (*str*) - This only needs to be used
if you are resuming a previosuly run experiment.

* **run_name** (*str*) - A run is supposed to be a sub-class of an experiment.
So this variable should be specified accordingly. 

* **model_type** - One of the following models has to be chosen as the model type. For further information, please refer to the [source code](https://github.com/aritraghsh09/GaMReN/blob/trial_network/ggt/models).
    * **ggt** - It stands for Galaxy Group-Equivariant Transformer model. 
    * **vgg16** - It is a 16 layer deep convolutional neural network model that is pretrained on the imagenet database. For more information, visit [here](https://pytorch.org/vision/main/models/generated/torchvision.models.vgg16.html). 
    * **ggt_no_gconv**
    * **vgg16_w_stn**
    * **vgg16_w_stn_drp**
    * **vgg16_w_stn_drp_2**
    * **vgg16_w_stn_at_drp**
    * **vgg16_w_stn_oc_drp**

* **model_state**

* **data_dir** - Add path to the data directory. The directory should be made as per the directions given in README.

* **split_slug** - This specifies how the data is split into train/
devel/test sets. *Balanced*/*Unbalanced* refer to whether selecting
equal number of images from each class. *xs*, *sm*, *lg*, *dev* all refer
to what fraction is picked for train/devel/test.

* **target_metrics** - Enter the target metrics separated by commas.

* **loss** - One of the following loss functions has to be chosen. For further information about the loss functions, please refer to the [source code](https://github.com/aritraghsh09/GaMReN/blob/trial_network/ggt/losses).
    * **mse**
    * **aleatoric**
    * **aleatoric_cov**  

* **expand_data** - This controls the factor by which the training
data is augmented.

* **cutout_size** (*int*) - Size of the fits image that the model takes as input. Default 167x167 pixels. 

* **channels** - 

* **n_workers** (*int*) - The number of workers to be used during the
data loading process.

* **batch_size** (*int*) - Mention the batch size to be used during training the model. This variable specifies how many images will be processed in a single batch. This is a hyperparameter. The default value is a good starting point.

* **epochs** (*int*) - Mention the number of epochs to be used during training the model.

* **lr** (*float*) - This is the learning rate to be used during the training process. This is a hyperparameter that should be tuned during the training process. The default value is a good starting point.

* **momentum** (*float*) - The value of the momentum to be used in the gradient descent optimizer that is used to train the model. This must always be ≥0. This accelerates the gradient descent process. This is a hyperparameter. The default value is a good starting point.

* **weight_decay** (*float*) - The amount of learning rate decay to be applied over each update.

* **parallel/ no-parallel** (*bool*) - The parallel argument controls whether or not
to use multiple GPUs when they are available. 

* **normalize/no-normalize** (*bool*) - The normalize argument controls whether or not, the
loaded images will be normalized using the `arcsinh` function. 

* **label_scaling** - The label scaling option controls whether to
standardize the labels or not. Set this to *std* for sklearn's
`StandardScaling()` and *minmax* for sklearn's `MinMaxScaler()`.
This is especially important when predicting multiple
outputs. For more information, visit [here](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html).

* **transform/no-transform** (*bool*) - If True, the training images are passed through a
series of random transformations.

* **crop/no-crop** (*bool*) - If True, all images are passed through a cropping
operation before being fed into the network. Images are cropped
to the cutout_size parameter.

* **nesterov/no-nesterov** (*bool*) - Whether to use Nesterov momentum or not. 

* **repeat_dims/no-repeat_dims** (*bool*) - In case of multi-channel data, whether to repeat a two
dimensional image as many times as the number of channels. 

* **dropout_rate** (*float*) - The dropout rate to use for all the layers in the
    model. If this is set to None, then the default dropout rate
    in the specific model is used.

### Returns:
    Trained GaMReN model.


### Return type: 
    PyTorch `Model` class. 

## Aleatoric covariant loss

```{eval-rst}  
.. autoapisummary::

   ggt.losses.aleatoric_cov_loss



.. py:function:: aleatoric_cov_loss(outputs, targets, num_var=3, average=True)

   Computes the Aleatoric Loss while including the full
   covariance matrix of the outputs.

   If you are predicting for n output variables, then the
   number of output neuros required for this loss is
   (3n + n^2)/2.

   Args:
       outputs: (tensor) - predicted outputs from the model
       targets: (tensor) - ground truth labels
       size_average: (bool) - if True, the losses are
              averaged over all elements of the batch
   Returns:
       aleatoric_cov_loss: (tensor) - aleatoric loss

   Formula:
       loss = 0.5 * [Y - Y_hat].T * cov_mat_inv
               * [Y - Y_hat] + 0.5 * log(det(cov_mat))
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




