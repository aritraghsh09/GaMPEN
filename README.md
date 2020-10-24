Galaxy Group-Equivariant Transformer
===
This repository contains the source code for the Galaxy Group-Equivariant Transformer and its associated modules.

## Installation
Training and inference for GGT require a Python 3.6+ and a CUDA-friendly GPU.

1. Create and activate a new Python 3.6+ `virtualenv`. More details can be found [here](https://docs.python.org/3/library/venv.html).
2. Clone this repository with
```bash
git clone https://github.com/amritrau/ggt.git
```
3. Install all the required dependencies. From the root directory of the `ggt` repository, run
```bash
make requirements
```
4. To confirm that the installation was successful, run
```bash
make check
```

## Quickstart
To train a GGT model, you need to prepare the dataset and running the provided trainer. During and after training, you can launch the MLFlow UI to view the training logs and artifacts.

### Data preparation
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

### Running the trainer
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

### Launching the MLFlow UI
Open a separate shell and activate the virtual environment that the model is training in. Then, run
```bash
mlflow ui
```

This launches the MLFlow UI on `localhost:5000`. If you are training on a remote machine, use SSH tunneling to access the UI by running the command on your _local_ system:
```bash
ssh -i <keyfile> <user>@<remote.com> -NL 5000:localhost:5000
```
No output will be shown if the connection was successful. Open a browser and navigate to `localhost:5000` to monitor your model. (For site-specific instructions, see this repository's [wiki](https://github.com/amritrau/ggt/wiki).)

## Loading other datasets
1. From the root directory of this repository, run
```bash
mkdir -p data/(dataset-name)/cutouts
```
2. Place FITS files in `data/(dataset-name)/cutouts`.
3. Provide a file titled `info.csv` at `data/(dataset-name)`. This file should have (at least) a column titled `file_name` (corresponding to the basenames of the files in `data/(dataset-name)/cutouts`) and a column titled `bt_g` containing bulge-to-total ratios. See [here](http://amritrau.github.io/assets/data/info.csv) for an example CSV (7.3M).
4. Generate train/devel/test splits with
```bash
python ggt/data/make_splits.py --data_dir=data/(dataset-name)/
```
5. Follow the instructions under [Running the trainer](#running-the-trainer), replacing `data/sdss/` with `data/(dataset-name)/`.

## Modules
### Auto-cropping
The spatial transformer subnetwork that GGT learns can be extracted and used as an auto-cropping module to automatically focus on the region of interest of an image. After training a GGT model (or downloading a pretrained model), run the auto-cropping module with
```bash
python -m ggt.modules.autocrop \
  --model_path=models/<your_model>.pt \
  --image_dir=path/to/fits/cutouts/
```

The auto-cropping module automatically resizes and normalizes the provided FITS images to match GGT's required image format. Then, the auto-cropping module feeds the prepared image through the provided model's spatial transformer subnetwork to automatically crop the image. Results are written in `.png` form back to the provided image directory. An example with a Hyper Suprime-Cam image is shown below.


![Auto-cropping](/docs/assets/stn_figure.png)
