# Getting Started

GaMPEN is written in Python and uses the PyTorch deep learning library to perform all of the machine learning operations. 

## Installation
Training and inference for GaMPEN requires a Python 3.6+ and a CUDA-friendly GPU.

1. Create and activate a new Python 3.6+ `virtualenv`. More details can be found [here](https://docs.python.org/3/library/venv.html).
2. Clone this repository with
```bash
git clone https://github.com/aritraghsh09/GaMReN.git
```
3. Install all the required dependencies. From the root directory of `GaMPEN` repository, run
```bash
make requirements
```
4. To confirm that the installation was successful, run
```bash
make check
```
5. To check whether the GaMPEN is able to detect GPUs, type `Python` from the root directory and run the following command:
```python
from ggt.utils.device_utils import discover_devices
discover_devices()
```
You should get the output as *cuda*. 
### Quickstart
To train a GaMPEN model, you need to prepare the dataset and run the module `train.py`. During and after training, you can [launch the MLFlow UI](#launching-the-mlflow-ui) to view the training logs and artifacts. Check out our [Tutorials](#Tutorials)section for 

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
To list additional options, run
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
3. Provide a file titled `info.csv` at `data/(dataset-name)`. This file should have (at least) a column titled `file_name` (corresponding to the basenames of the files in `data/(dataset-name)/cutouts`) and a column titled `bt_g` containing bulge-to-total ratios.

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

If you are using a GPU, then you would need to make sure that the appropriate CUDA and cuDNN versions are installed. The appropriate version is decided by the versions of your installed Python libraries. For detailed instructions on how to enable GPU support for Tensorflow, please see [this link](https://www.tensorflow.org/install/source#linux).