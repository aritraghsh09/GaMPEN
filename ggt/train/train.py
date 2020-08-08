# -*- coding: utf-8 -*-
import click
import logging
import math
from pathlib import Path
from functools import partial

import mlflow

import torch
import torch.nn as nn
import torch.optim as opt
import kornia.augmentation as K

from ggt.data import FITSDataset, get_data_loader
from ggt.models import model_factory, model_stats, save_trained_model
from ggt.train import create_trainer
from ggt.utils import discover_devices
from ggt.visualization.spatial_transform import visualize_spatial_transform


@click.command()
@click.option('--experiment_name', type=str, default='ggt-demo')
@click.option('--model_type',
              type=click.Choice(['ggt'], case_sensitive=False),
              default='ggt')
@click.option('--model_state', type=click.Path(exists=True), default=None)
@click.option('--data_dir', type=click.Path(exists=True), required=True)
@click.option('--split_slug', type=str, required=True)
@click.option('--cutout_size', type=int, default=167)
@click.option('--n_workers', type=int, default=16)
@click.option('--batch_size', type=int, default=32)
@click.option('--epochs', type=int, default=40)
@click.option('--lr', type=float, default=0.005)
@click.option('--momentum', type=float, default=0.9)
@click.option('--parallel/--no-parallel', default=False)
@click.option('--normalize/--no-normalize', default=True)
@click.option('--transform/--no-transform', default=True)
def train(**kwargs):
    """Runs the training procedure using MLFlow."""

    # Copy and log args
    args = {k: v for k, v in kwargs.items()}

    # Discover devices
    args['device'] = discover_devices()

    # Create the model given model_type
    cls = model_factory(args['model_type'])
    model = cls()
    model = nn.DataParallel(model) if args['parallel'] else model
    model = model.to(args['device'])

    # Load the model from a saved state if provided
    if args['model_state']:
        model.load_state_dict(torch.load(args['model_state']))

    # Define the optimizer and criterion
    optimizer = opt.SGD(model.parameters(), lr=args['lr'],
                        momentum=args['momentum'])
    criterion = nn.MSELoss()

    # Create a DataLoader factory based on command-line args
    loader_factory = partial(
        get_data_loader,
        batch_size=args['batch_size'],
        n_workers=args['n_workers']
    )

    # Select the desired transforms
    T = None
    if args['transform']:
        T = nn.Sequential(
            K.RandomHorizontalFlip(),
            K.RandomVerticalFlip(),
            K.RandomRotation(360),
        )

    # Generate the DataLoaders and log the train/devel/test split sizes
    splits = ('train', 'devel', 'test')
    datasets = {k: FITSDataset(
        data_dir=args['data_dir'],
        slug=args['split_slug'],
        normalize=args['normalize'],
        transform=T if k == 'train' else None,
        split=k) for k in splits}
    loaders = {k: loader_factory(v) for k, v in datasets.items()}
    args['splits'] = {k: len(v.dataset) for k, v in loaders.items()}

    # Start the training process
    mlflow.set_experiment(args['experiment_name'])
    with mlflow.start_run():
        # Write the parameters and model stats to MLFlow
        args = {**args, **model_stats(model)}  # py3.9: d1 |= d2
        for k, v in args.items():
            mlflow.log_param(k, v)

        # Set up trainer
        trainer = create_trainer(model, optimizer, criterion, loaders,
                                 args['device'])

        # Run trainer and save model state
        trainer.run(loaders['train'], max_epochs=args['epochs'])
        slug = (f"{args['experiment_name']}-{args['split_slug']}-"
                f"{mlflow.active_run().info.run_id}")
        model_path = save_trained_model(model, slug)

        # Log model as an artifact
        mlflow.log_artifact(model_path)

        # Visualize spatial transformation
        output_dir = Path("output") / slug
        output_dir.mkdir(parents=True, exist_ok=True)
        visualize_spatial_transform(model, loaders['devel'], output_dir,
                                    device=args['device'],
                                    nrow=round(math.sqrt(args['batch_size'])))

        # Log output directory as an artifact
        mlflow.log_artifacts(output_dir)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    train()
