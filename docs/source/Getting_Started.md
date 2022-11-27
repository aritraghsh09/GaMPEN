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
4. Install all the required dependencies. From the root directory of `GaMPEN` repository, run
```bash
make requirements
```
5. To confirm that the installation was successful, run
```bash
make check
```
It is okay if there are some warnings or some tests are skipped. The only thing you should look out for is errors. 

6. To check whether the GaMPEN is able to detect GPUs, type `Python` from the root directory and run the following command:
```python
from ggt.utils.device_utils import discover_devices
discover_devices()
```
The output should be *cuda*.


### Quickstart
To train a GaMPEN model, you need to place the training images and their corresponding labels in a specific directory structure. Here, we do a quick demonstration on some SDSS data. The backbone of GaMPEN's training is done by `train.py`. 

In order to have an easy GUI monitoring all your models during and after training, you can [launch the MLFlow UI](#launching-the-mlflow-ui) to view various statistics of the models being trained as well as the paths to the trained models.

Check out our [Tutorials](Tutorials.md) page for extensive details on how to train a GaMPEN model from scratch, how to perform transfer-learning/fine-tuning, and how to use GaMPEN to perform inference on your own data.

#### Data preparation
In this section, we will prepare and load the SDSS sample of Simard, et al. To load another dataset, see [Loading other datasets](#loading-other-datasets).

1. From the root directory of this repository, run
```bash
make sdss
```
2. Place all the relevant FITS files under `data/sdss/cutouts/`.
3. Generate train/devel/test splits with
```bash
python ggt/data/make_splits.py --data_dir=data/sdss/
```

#### Running the trainer
Run the trainer with
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
To list the all possible options along with explanations, head to the [Using GaMPEN](Using_GaMPEN.md) page or run
```bash
python ggt/train/train.py --help
```

#### Launching the MLFlow UI
Open a separate shell and activate the virtual environment that the model is training in. Then, run
```bash
mlflow ui
```

This launches the MLFlow UI on `localhost:5000`. If you are training on a remote machine, use SSH tunneling to access the UI by running the command on your _local_ system:
```bash
ssh -i <keyfile> <user>@<remote.com> -NL 5000:localhost:5000
```
No output will be shown if the connection was successful. Open a browser and navigate to `localhost:5000` to monitor your model. (For site-specific instructions, see this repository's [wiki](https://github.com/amritrau/ggt/wiki).)


### Loading other datasets
1. From the root directory of this repository, run
```bash
mkdir -p data/(dataset-name)/cutouts
```
2. Place FITS files in `data/(dataset-name)/cutouts`.
3. Provide a file titled `info.csv` at `data/(dataset-name)`. This file should have (at least) a column titled `file_name` (corresponding to the basenames of the files in `data/(dataset-name)/cutouts`) and one column for each of the variables that you are trying to predict. For example, if you are trying to predict the radius of a galaxy, you would have a column titled `radius`. The values in this column should be the target values of the radius of the galaxies in the dataset.

4. Generate train/devel/test splits with
```bash
python ggt/data/make_splits.py --data_dir=data/(dataset-name)/
```
After generating the splits, the subdirectory `data` should look like:
```
- data
    - __init__.py
    - dataset.py
    - make_splits.py
    - dataset_name
        - info.csv
        - cutouts 
        - splits
```
5. Follow the instructions under [Running the trainer](#running-the-trainer), replacing `data/sdss/` with `data/(dataset-name)/`.


## GPU Support

If you are using an NVIDIA GPU, then you would need to make sure that the appropriate CUDA and cuDNN versions are installed. See [this link](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch) for more details.
