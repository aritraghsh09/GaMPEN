# Getting Started

GaMPEN is written in Python and relies on the [PyTorch](https://pytorch.org/) deep learning library to perform all of its tensor operations.

## Installation
Training and inference for GaMPEN requires Python 3.7 or 3.8. Trained GaMPEN models can be run on a CPU to perform inference, but training a model requires access to a CUDA-enabled GPU for reasonable training times.

1. Create a new conda environment with Python 3.7 or 3.8. Extensive instructions on how to create a conda enviornment can be found [here](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-with-commands). Of course, you could use any other method of creating a virtual environment, but we will assume you are using conda for the rest of this guide.
```bash
conda create -n gampen python=3.7
```
2. Activate the new environment
```bash
conda activate gampen
```
3. Navigate to the directory where you want to install GaMPEN and then clone this repository with
```bash
git clone https://github.com/aritraghsh09/GaMPEN.git
```
4. Navigate into the root directory of GaMPEN 
```bash
cd GaMPEN
```
4. Install all the required dependencies with
```bash
make requirements
```
5. To confirm that the installation was successful, run
```bash
make check
```
It is okay if there are some warnings or some tests are skipped. The only thing you should look out for is errors produced by the `make check` command.

:::{tip}
If you get an error about specific `libcudas` libraries being absent while running `make check`, this has probably to do with the fact that you don't have the appropriate `CUDA` and `cuDNN` versions installed for the PyTorch version being used by GaMPEN. See [below](#gpu-support) for more details about GPU support.
:::

## GPU Support

GaMPEN can make use of multiple GPUs while training if you pass in the appropriate arguments to the `train.py` script.

To check whether the GaMPEN is able to detect GPUs, type `python` into the command line from the root directory and run the following command:
```python
from ggt.utils.device_utils import discover_devices
discover_devices()
```
The output should be `cuda`if GaMPEN can detect a GPU.

If the output is `cpu` then GaMPEN couldn't find a GPU. This could be because you don't have a GPU, or because you haven't installed the appropriate CUDA and cuDNN versions.

 If you are using an NVIDIA GPU, then you can use [this link](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch) for more details about specific CUDA and cuDNN versions that are compatible with different PyTorch versions. To check the version of PyTorch you are using, type `python` into the command line and then run the following code-block:

```python
import torch
print(torch.__version__)
```


## Quickstart

The core steps involved in running GaMPEN include :-

1. Placing your data in a specific directory structure
2. Using the `GaMPEN/ggt/data/make_splits.py` script to generate train/devel/test splits
3. Using the `GaMPEN/ggt/train/train.py` script to train a GaMPEN model
4. Using the MLFlow UI to monitor your model during and after training
5. Using the `GaMPEN/ggt/modules/inference.py` script to perform predictions using the trained model.
6. Using the `GaMPEN/ggt/modules/result_aggregator.py` script to aggregate the predictions into an easy-to-read pandas data-frame.

:::{attention}
We strongly recommend going through our [Tutorials](Tutorials.md) and [Using GaMPEN](Using_GaMPEN.md) pages to get an in-depth understanding of how to use GaMPEN, and an overview of all the steps above.

Here, we provide a quick-and-dirty demo to just get you started training your 1st GaMPEN model! This section is intentionally short, without much explanation.
:::

### Data preparation
Let's download some simulated Hyper Suprime-Cam (HSC) images from the Yale servers. To do this run from the root directory of this repository:

```bash
make demodir=./../hsc hsc_demo
```

This should create a directory called `hsc` at the specified `demodir` path with the following components

```text
- hsc
  - info.csv -- file names of the trianing images with labels
  - cutouts/ -- 67 images to be used for this demo
```

Now, let's split the data into train, devel, and test sets. To do this, run
```bash
python ./ggt/data/make_splits.py --data_dir=./../hsc/ --target_metric='bt'
```

This will create another folder called `splits` within `hsc` with the different data-splits for training, devel (validation), and testing.

### Running the trainer
Let's use the data we just downloaded to train a GaMPEN model. To do this, we will
use the `train.py` script:-

```bash
python ggt/train/train.py \
  --experiment_name='demo' \
  --data_dir='./../hsc/' \
  --split_slug='balanced-dev2' \
  --batch_size=16 \
  --epochs=2 \
  --lr=5e-7 \
  --momentum=0.99 \
  --crop \
  --cutout_size=239 \
  --target_metrics='custom_logit_bt,ln_R_e_asec,ln_total_flux_adus' \
  --repeat_dims \
  --no-nesterov \
  --label_scaling='std' \
  --dropout_rate=0.0004 \
  --loss='aleatoric_cov' \
  --weight_decay=0.0001 \
  --parallel
```
To list the all possible options along with explanations, head to the [Using GaMPEN](Using_GaMPEN.md) page or run
```bash
python ggt/train/train.py --help
```

### Launching the MLFlow UI

Although, this is an optional step, GaMPEN comes pre-installed with the [MLFlow UI](https://mlflow.org/) to help you 
monitor the different models that you are currently training and compare these with models you have trained in the past. 

To initialize MLFlow, open a separate shell and _activate the virtual environment_ that the model is training in. Then,
navigate to the directory from where you initiated the training run and execute the following command:-

```bash
mlflow ui
```
Now navigate to `http://localhost:5000/` to access the MLFlow UI which will show you the status of your model training.

:::{warning}
If you are running these commands on a server/remote machine, you will need to follow the additional instructions listed below
to access the MLFlow UI.
:::

#### MLFLow on a remote machine

First, on the server/HPC system, open a shell and _activate the virtual environment_ that the model is training in. Thereafter, navigate to the directory from where you initiated your GaMPEN run (you can do this on separate machine as well -- only the filesystem needs to tbe same). Then, execute the following command:-

```bash
mlflow ui --host 0.0.0.0
```

The `--host` option is important to make the MLFlow  server accept connections from other machines. 

Now, from your local machine, tunnel into the `5000` port of the server where you ran the above command. After forwarding, if you navigate to  `http://localhost:5000/` you should be able to access the MLFlow UI.

:::{tip}
For example, let's say you are working in an HPC environment, where the machine where you ran the above command is named `server1` and the login node to your HPC system is named `hpc.university.edu` and you have the username `astronomer`. Then, to establish port-forwarding, you should type the following command on your local machine:-

```bash
ssh -N -L 5000:server1:5000 astronomer@hpc.university.edu
```

If performing the above step without a login node (e.g., a server which has the IP `server1.university.edu`), you should be able to
establish port-forwarding simply with:- 

```bash
ssh -N -L 5000:localhost:5000 astronomer@server1.university.edu
```
:::



### Training on other datasets
In [Running the trainer](#running-the-trainer) section, we demonstrated how to train a GaMPEN model on the demo HSC
dataset. Below, we outline the core steps involved in training a model on your own data:-

1. Create the necessary directory structure with:-

```bash
mkdir -p dataset-name/cutouts
```
where `dataset-name` can be any name of your choosing.

2. GaMPEN expects input images to be in the `.fits` format, centered on the galaxy of interest. Make 
same-sized individual cutouts for all objects in your dataset; and place these files in `dataset-name/cutouts/`.

3. Place a file titled `info.csv` inside the `dataset-name` directory. This file should have (at least) a column titled `file_name` (corresponding to the names of the files in `dataset-name/cutouts`), a column titled `object_id` (with a unique ID for each file in `dataset-name/cutouts/`) and one column each for the parameters that you are trying to predict. For example, if you are trying to predict the radius and magnitude of a galaxy, you would have two columns titled `radius` and `magnitude`. 

:::{note}
Besides the `file_name` and `object_id` columns; all other columns in `info.csv` can be named according to your choosing. 
There are also no limitations on additional columns being present in `info.csv`.
:::

4. Next, separate your dataset into train, devel, and test splits with the following command:-

```bash
python ggt/data/make_splits.py --data_dir=/dataset-name/
```
:::{attention}
You should provide the full path of the `dataset-name` directory to the `data_dir` argument.
:::

The `make_splits.py` file splits the dataset according to a set of pre-determined fractions and you can choose to use any of these for your analysis. Details of the various splits are mentioned on the [Using GaMPEN](Using_GaMPEN.md#make-splits) page.

After generating the splits, the `dataset-name` directory should look like this:
```
- dataset_name
    - info.csv
    - cutouts/
    - splits/
```

:::{tip}
To change the fractions (of train/devel/test data) in the various splits; alter the `split_types` dictionary in `make_splits.py`
:::

5. Follow the instructions in [Running the trainer](#running-the-trainer).



