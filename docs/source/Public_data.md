# Public Data Release Handbook

```{note}
We are still rolling out the full data release of HSC PDR2 morphological parameters. This page will be updated continuously through the Fall of 2022. Please check back for updates and contact us if you have any questions.
```

## FTP Server

All components of the public data release are hosted on the Yale Astronomy FTP server ``ftp.astro.yale.edu``. There are multiple ways you can access the FTP server, and we summarize some of the methods below.


### Using Unix Command Line

```bash
ftp ftp.astro.yale.edu
cd pub/aghosh/hsc_wide_pdr2_morph/<appropriate_subdirectory>
```

If prompted for a username, try `anonymous` and keep the password field blank. After connecting, you can download files using the ``get`` command. 

### Using a Browser

Navigate to ``ftp://ftp.astro.yale.edu/pub/<appropriate_subdirectory>``

### Using Finder on OSX

Open Finder, and then choose Go &rarr; Connect to Server (or command + K) and enter ``ftp://ftp.astro.yale.edu/pub/aghosh/``. Choose to connect as 
``Guest`` when prompted. 

Thereafter, navigate to the appropriate subdirectory. 

## Hyper Suprime-Cam Wide PDR2 Morphology

### Prediction Tables

The prediction tables are located at the following subdirectories on the FTP server:

* g-band HSC-Wide z < 0.25 galaxies &rarr; ``/pub/aghosh/hsc_wide_pdr2_morph/g_0_025/g_0_025_preds_summary.csv``

* r-band HSC-Wide 0.25 < z < 0.50 galaxies &rarr; ``/pub/aghosh/hsc_wide_pdr2_morph/r_025_050/r_025_050_preds_summary.csv``

* i-band HSC-Wide 0.50 < z < 0.75 galaxies &rarr; ``/pub/aghosh/hsc_wide_pdr2_morph/i_050_075/i_050_075_preds_summary.csv``

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

* g-band HSC-Wide z < 0.25 galaxies &rarr; ``/pub/aghosh/hsc_wide_pdr2_morph/g_0_025/posterior_arrays/``

* r-band HSC-Wide 0.25 < z < 0.50 galaxies &rarr; ``/pub/aghosh/hsc_wide_pdr2_morph/r_025_050/posterior_arrays/``

* i-band HSC-Wide 0.50 < z < 0.75 galaxies &rarr; ``/pub/aghosh/hsc_wide_pdr2_morph/i_050_075/posterior_arrays/``

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

Coming Soon! 





