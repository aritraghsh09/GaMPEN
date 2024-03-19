# Public Data Release Handbook

```{note}
We are still rolling out the full data release of HSC PDR2 morphological parameters. This page will be updated continuously through the Spring of 2023. If you can't find the portion of the data-release that you need, please drop us a line!
```

## FTP Server

All components of the public data release are hosted on the Yale Astronomy FTP server ``ftp.astro.yale.edu``. There are multiple ways you can access the FTP server, and we summarize some of the methods below.


### Using Unix Command Line

First, using the Unix terminal, navigate to the location where you want to download the files. Thereafter, connect to the FTP server using the following command

```bash
ftp ftp.astro.yale.edu
```
If prompted for a username, try `anonymous` and keep the password field blank. 

After connecting, navigate to the appropriate subdirectory (scroll down for locations) and download the relevant files using the ``get`` command. For example, to get the prediction tables for g-band HSC-Wide z < 0.25 galaxies, you should issue the following commands after connecting

```bash
cd /pub/hsc_morph/g_0_025/
get g_0_025_preds_summary.csv
```
This should download `g_0_025_preds_summary.csv` into the directory from which you initiated the FTP connection.

*Tip: Mac terminals don't come pre-installed with the `ftp` command. But, if you use [Homebrew](https://brew.sh/), you can install FTP using `brew install inetutils`*

### Using a Browser

On a browser, navigate to ``ftp://ftp.astro.yale.edu/pub/hsc_morph/``

Now, download the relevant files by navigating to the relevant subdirectory (see below).

*Tip: If you are using Google Chrome, make sure that you are not selecting the default Google Search option from the suggested links in the dropdown*

### Using Finder on MacOS

Open Finder, and then choose Go &rarr; Connect to Server (or command + K) and enter ``ftp://ftp.astro.yale.edu/pub/hsc_morph/``. Choose to connect as 
``Guest`` when prompted. 

Thereafter, navigate to the appropriate subdirectory to download the relevant files. 

## Hyper Suprime-Cam Wide PDR2 Morphology Directories

### Prediction Tables

The prediction tables are located at the following subdirectories on the FTP server:

* g-band HSC-Wide z < 0.25 galaxies &rarr; ``/pub/hsc_morph/g_0_025/g_0_025_preds_summary.csv``

* r-band HSC-Wide 0.25 < z < 0.50 galaxies &rarr; ``/pub/hsc_morph/r_025_050/r_025_050_preds_summary.csv``

* i-band HSC-Wide 0.50 < z < 0.75 galaxies &rarr; ``/pub/hsc_morph/i_050_075/i_050_075_preds_summary.csv``

The various columns in the prediction tables are described below:

* ``object_id``: The unique object ID for the galaxy. This is the same as the ``object_id`` in the HSC-Wide PDR2 catalog.

* ``ra``: The right ascension of the galaxy in degrees.

* ``dec``: The declination of the galaxy in degrees.

* ``z_best``: The redshift of the galaxy. This is the same as the ``z_best`` in the HSC-Wide PDR2 catalog.

* ``zmode``: The redshift mode of the galaxy. The two options are `specz` or `photz`. 

There are multiple columns for each of the three morphological parmaeters: effective radius (``R_e``) (in arcsec), bulge-to-total_light_ratio (``bt``), total flux (``total_flux``) (in ADUs), and mangitude (``total_mag``). In all the columns below `xx` refers to the column names mentioned in brackets.

* ``preds_xx_mode``: The mode of the posterior distribution for the morphological parameter. (**Recommended**)

* ``preds_xx_mean``: The mean of the posterior distribution of the morphological parameter.

* ``preds_xx_median``: The median of the posterior distribution of the morphological parameter.

* ``preds_xx_std``: The standard deviation of the posterior distribution of the morphological parameter.

* ``preds_xx_skew``: The skewness of the posterior distribution of the morphological parameter.

* ``preds_xx_kurtosis``: The kurtosis of the posterior distribution of the morphological parameter.

* ``preds_xx_sig_ci``: The 1-sigma confidence interval of the posterior distribution of the morphological parameter.

* ``preds_xx_twosig_ci``: The 2-sigma confidence interval of the posterior distribution of the morphological parameter.

* ``preds_xx_threesig_ci``: The 3-sigma confidence interval of the posterior distribution of the morphological parameter.


### Posterior Distribution Files for Individual Galaxies

The predicted posterior distributions for individual galaxies are available as ``.npy`` files. The files are named as ``zz.npy`` where zz is the ``object_id`` mentioned in the prediction tables. The files located at the following subdirectories on the FTP server:

* g-band HSC-Wide z < 0.25 galaxies &rarr; ``/pub/hsc_morph/g_0_025/posterior_arrays/``

* r-band HSC-Wide 0.25 < z < 0.50 galaxies &rarr; ``/pub/hsc_morph/r_025_050/posterior_arrays/``

* i-band HSC-Wide 0.50 < z < 0.75 galaxies &rarr; ``/pub/hsc_morph/i_050_075/posterior_arrays/``

You can load the `.npy` files using the `np.load` function in `Numpy`. The array dimensions are as follows:

* 0 &rarr; x of radius (in arcsec)
* 4 &rarr; y of radius (in arcsec)
* 1 &rarr; x of flux (in ADUs)
* 5 &rarr; y of flux (in ADUs)
* 2 &rarr; x of bulge-to-total_light_ratio
* 6 &rarr; y of bulge-to-total_light_ratio
* 3 &rarr; x of magnitude
* 7 &rarr; y of magnitude


### Trained GaMPEN Models

The trained GaMPEN models are available as  ``.pt`` PyTorch files. The models are at the following locations:-

**Real Data Models**
* g-band HSC-Wide z < 0.25 galaxies &rarr; ``/pub/hsc_morph/g_0_025/trained_model/g_0_025_model.pt``

* r-band HSC-Wide 0.25 < z < 0.50 galaxies &rarr; ``/pub/hsc_morph/r_025_050/trained_model/r_025_050_model.pt``

* i-band HSC-Wide 0.50 < z < 0.75 galaxies &rarr; ``/pub/hsc_morph/i_050_075/trained_model/i_050_075_model.pt``

**Simulated Data Models**
* Simulated g-band HSC-Wide z < 0.25 galaxies &rarr; ``/pub/hsc_morph/sim_g_0_025/trained_model/sim_g_0_025.pt`` 

* Simulated r-band HSC-Wide 0.25 < z < 0.50 galaxies &rarr; ``/pub/hsc_morph/sim_r_025_050/trained_model/sim_r_025_050.pt``

* Simulated i-band HSC-Wide 0.50 < z < 0.75 galaxies &rarr; ``/pub/hsc_morph/sim_i_050_075/trained_model/sim_i_050_075.pt``


#### Trained Model Parameters

We mention some of the finally tuned hyper-parameters that we used for the above models. Note that while performing inference using the above models, you will need to use some of these parameters.


##### Real Data Models

| Parameter Name | Low-z Real Data | Mid-z Real Data | High-z Real Data |
|----------------|-----------------|-----------------|------------------|
| `model_type` | `vgg16_w_stn_oc_drp`| `vgg16_w_stn_oc_drp` | `vgg16_w_stn_oc_drp` |
| `cutout_size` | 239 | 143 | 96 |
| `droput_rate` | 0.0004 | 0.0002 | 0.0002 |
| `label_scaling` | `std` | `std` | `std` |
| `loss` | `aleatoric_cov` | `aleatoric_cov` | `aleatoric_cov` |
| `lr` | 5e-8 | 5e-8 | 5e-6 |
| `momentum` | 0.99 | 0.99 | 0.99 |
| `nesterov` | False | False | False |
| `weight_decay` | 0.0001 | 0.0001 | 0.0001 |
| `parallel` | True | True | True |
| `target_metrics` | `custom_logit_bt`, `ln_R_e_asec`, `ln_total_flux_adus` | `custom_logit_bt`, `ln_R_e_asec`, `ln_total_flux_adus` | `custom_logit_bt`, `ln_R_e_asec`, `ln_total_flux_adus` |
| `split_slug` | `balanced-dev2` | `balanced-dev2` | `balanced-dev2` |


##### Simulated Data Models

| Parameter Name | Low-z Sims. | Mid-z Sims. | High-z Sims. |
|----------------|-----------------|-----------------|------------------|
| `model_type` | `vgg16_w_stn_oc_drp`| `vgg16_w_stn_oc_drp` | `vgg16_w_stn_oc_drp` |
| `cutout_size` | 239 | 143 | 96 |
| `droput_rate` | 0.0007 | 0.0007 | 0.0004 |
| `label_scaling` | `std` | `std` | `std` |
| `loss` | `aleatoric_cov` | `aleatoric_cov` | `aleatoric_cov` |
| `lr` | 5e-7 | 5e-7 | 5e-7 |
| `momentum` | 0.99 | 0.99 | 0.99 |
| `nesterov` | False | False | False |
| `weight_decay` | 0.0001 | 0.0001 | 0.0001 |
| `parallel` | True | True | True |
| `target_metrics` | `custom_logit_bt`, `ln_R_e_asec`, `ln_total_flux_adus` | `custom_logit_bt`, `ln_R_e_asec`, `ln_total_flux_adus` | `custom_logit_bt`, `ln_R_e_asec`, `ln_total_flux_adus` |
| `split_slug` | `balanced-dev` | `balanced-dev` | `balanced-dev` |


#### Scaling Data

Note that as mentioned in the [Predictions Tutorial](Tutorials.md#making-predictions), in order to unscale the predictions made using the above models, you need access to the training files. 

You can access these files at the following locations using `wget`:

`ftp://ftp.astro.yale.edu/pub/hsc_morph/xxxx/scaling_data_dir/info.csv`

and 

`ftp://ftp.astro.yale.edu/pub/hsc_morph/xxxx/scaling_data_dir/splits/`

where `xxxx` is `g_0_025`, `r_025_050`, or `i_050_075` for low-, mid-, and high-z real data models respectively; and `sim_g_0_025`, `sim_r_025_050`, or `sim_i_050_075` for low-, mid-, and high-z simulated data models respectively.


#### Custom Scaling Function

As mentioned in the [Tutorials](Tutorials.md), all the trained GaMPEN models first make predictions in the `logit(bulge-to-total light ratio)` space. The predictions are then scaled to the `bulge-to-total light ratio` space using the custom inverse-scaling function defined in [`/GaMPEN/ggt/modules/result_aggregator.py`](./../../ggt/modules/result_aggregator.py). 

Here, for completeness, we provide the custom scaling function that we used for the **forward** logit transformation while creating our `info.csv` files. The only way this is different from the standard logit transformation is that we prevent the function from blowing up for values of `bulge-to-total_light_ratio` that are very close to 0 or 1. 

```python
from scipy.special import logit 

def logit_custom(x_input):
    
    '''Handling for 0s and 1s while doing a
       logit transformation
       
       x_input should be the entire column/array
       in info.csv over which you are applying 
       the transformation'''
    
    x = np.array(x_input)
    
    if np.min(x) < 0 or np.max(x) > 1:
        raise ValueError("x must be between 0 and 1")

    if np.min(x) == 0:
        min_x = np.min(x[x != 0])
        add_epsilon = min_x/2.0
        x[np.where(x==0)[0]] = add_epsilon
        
    if np.max(x) == 1:
        max_x = np.max(x[x != 1])
        sub_epsilon = (1-max_x)/2.0
        x[np.where(x==1)[0]] = 1.0 - sub_epsilon
        
    return logit(x)
```




