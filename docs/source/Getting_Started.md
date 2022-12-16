# Getting Started

GaMPEN is written in Python and relies on the [PyTorch](https://pytorch.org/) deep learning library to perform all of its tensor operations.

## Installation
Training and inference for GaMPEN requires Python 3.7+. Trained GaMPEN models can be run on a CPU to perform inference, but training a model requires access to a CUDA-enabled GPU for reasonable training times.

1. Create a new conda environment with Python 3.7+. Extensive instructions on how to create a conda enviornment can be found [here](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-with-commands). Of course, you could use any other method of creating a virtual environment, but we will assume you are using conda for the rest of this guide.
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

## GPU Support

GaMPEN can make use of multiple GPUs while training if you pass in the appropriate arguments to the `train.py` script.

To check whether the GaMPEN is able to detect GPUs, type `python` into the command line from the root directory and run the following command:
```python
from ggt.utils.device_utils import discover_devices
discover_devices()
```
The output should be *cuda*.

Note that if you are using an NVIDIA GPU, then you would need to make sure that the appropriate CUDA and cuDNN versions are installed. See [this link](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch) for more details.

If you get an error about specific `libcudas` libraries being absent while running `make check`, this has probably to do with the fact that you don't have the appropriate CUDA and cuDNN versions installed for the PyTorch version being used by GaMPEN. 


## Quickstart

The core parts of the GaMPEN ecosystem are :-

* Placing your data in a specific directory structure
* Using the `GaMPEN/ggt/data/make_splits.py` script to generate train/devel/test splits
* Using the `GaMPEN/ggt/train/train.py` script to train a GaMPEN model
* Using the MLFlow UI to monitor your model during and after training
* Using the `GaMPEN/ggt/modules/inference.py` script to perform predictions using the trained model.
* Using the `GaMPEN/ggt/modules/result_aggregator.py` script to aggregate the predictions into an easy-to-read pandas data-frame.

**We highly recommend going through all our [Tutorials](Tutorials.md) to get an in-depth understanding of how to use GaMPEN, and an overview of all the steps above.**

Here, we provide a quick-and-dirty demo to just get you started training your 1st GaMPEN model! (**This is intentionally short without much explanation**)


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
Run the trainer with
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
  --target_metrics='custom_logit_bt,log_R_e,log_total_flux' \
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

Open a separate shell and activate the virtual environment that the model is training in. Then, run
```bash
mlflow ui
```
Now navigate to `http://localhost:5000/` to access the MLFlow UI which will show you the status of your model training.

#### MLFLow on a remote machine

First, on the server/HPC, navigate to the directory from where you initiated your GaMPEN run (you can do this on separate machine as well -- only the filesystem needs to tbe same). Then execute the following command 

```bash
mlflow ui --host 0.0.0.0
```

The `--host` option is important to make the MLFlow  server accept connections from other machines. 

Now from your local machine tunnel into the `5000` port of the server where you ran the above command.
F
or example, let's say you are in an HPC environment, where the machine where you ran the above command is named `server1` and the login node to your HPC is named `hpc.university.edu` and you have the username `astronomer`. Then to forward the port you should type the following command in your local machine 

```bash
ssh -N -L 5000:server1:5000 astronomer@hpc.university.edu
```

If performing the above step without a login node (e.g., a server whhich has the IP `server1.university.edu`), you should be able to do 

```bash
ssh -N -L 5000:localhost:5000 astronomer@server1.university.edu
```

After forwarding, if you navigate to  `http://localhost:5000/` you should be able to access the MLFlow UI



### Training on other datasets
1. First create the necessary directories with
```bash
mkdir -p (dataset-name)/cutouts
```
2. Place FITS files in `(dataset-name)/cutouts`.
3. Provide a file titled `info.csv` at `(dataset-name)`. This file should have (at least) a column titled `file_name` (corresponding to the names of the files in `(dataset-name)/cutouts`), a column titled `object_id` (with a unique ID of each file in `(dataset-name)/cutouts`) and one column for each of the variables that you are trying to predict. For example, if you are trying to predict the radius of a galaxy, you would have a column titled `radius`. 

4. Generate train/devel/test splits with
```bash
python ggt/data/make_splits.py --data_dir=data/(dataset-name)/
```

The `make_splits.py` file splits the dataset into a variety of splits and you can choose to use any of these for your analysis. Details of the various splits are mentioned on the [Using GaMPEN](https://gampen.readthedocs.io/en/latest/Using_GaMPEN.html#make-splits) page.

After generating the splits, the `dataset-name` directory should look like this:
```
- dataset_name
    - info.csv
    - cutouts/
    - splits/
```
5. Follow the instructions under [Running the trainer](#running-the-trainer).



