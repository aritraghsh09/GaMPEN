ggt
===
This repository contains the source code for the Galaxy Group-Equivariant Transformer and its associated modules.

## Installation
Training and inference for GGT require a Python 3.6+ and a CUDA-friendly GPU.

1. Create and activate a new Python 3.6+ `virtualenv`. More details can be found [here](https://docs.python.org/3/library/venv.html).
2. Clone this repository with
```bash
git clone https://github.com/amritrau/ggt.git
```
3. From the root directory of the `ggt` repository, run
```bash
make requirements
```
This installs all the required dependencies including PyTorch and Astropy.
4. To confirm that the installation was successful, run
```bash
make check
```

## Quickstart
To train a GGT model, you need to prepare the dataset and running the provided trainer. During and after training, you can launch the MLFlow UI to view the training logs and artifacts.

### Data preparation
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
No output will be shown if the connection was successful. Open a browser and navigate to `localhost:5000` to monitor your model.

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
