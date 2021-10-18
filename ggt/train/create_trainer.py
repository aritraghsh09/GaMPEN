import mlflow

import torch.nn as nn

from ignite.engine import (
    Events,
    create_supervised_trainer,
    create_supervised_evaluator,
)
from ignite.metrics import MeanAbsoluteError, MeanSquaredError, Loss

from ggt.metrics import ElementwiseMae
from ggt.losses import AleatoricLoss
from ggt.utils import metric_output_transform


def create_trainer(model, optimizer, criterion, loaders, device):
    """Set up Ignite trainer and evaluator."""
    trainer = create_supervised_trainer(
        model, optimizer, criterion, device=device
    )

    if isinstance(criterion, AleatoricLoss):
        output_transform = metric_output_transform
    else:
        output_transform = nn.Identity()

    metrics = {
        "mae": MeanAbsoluteError(output_transform=output_transform),
        "elementwise_mae": ElementwiseMae(output_transform=output_transform),
        "mse": MeanSquaredError(output_transform=output_transform),
        "loss": Loss(criterion),
    }
    evaluator = create_supervised_evaluator(
        model, metrics=metrics, device=device
    )

    # Define training hooks
    @trainer.on(Events.STARTED)
    def log_results_start(trainer):
        for L, loader in loaders.items():
            evaluator.run(loader)
            metrics = evaluator.state.metrics
            for M in metrics.keys():
                if M == "elementwise_mae":
                    for i, val in enumerate(metrics[M].tolist()):
                        mlflow.log_metric(f"{L}-{M}-{i}", val, 0)
                else:
                    mlflow.log_metric(f"{L}-{M}", metrics[M], 0)

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_devel_results(trainer):
        evaluator.run(loaders["devel"])
        metrics = evaluator.state.metrics
        for M in metrics.keys():
            if M == "elementwise_mae":
                for i, val in enumerate(metrics[M].tolist()):
                    mlflow.log_metric(
                        f"devel-{M}-{i}", val, trainer.state.epoch
                    )
            else:
                mlflow.log_metric(
                    f"devel-{M}", metrics[M], trainer.state.epoch
                )

    @trainer.on(Events.COMPLETED)
    def log_results_end(trainer):
        for L, loader in loaders.items():
            evaluator.run(loader)
            metrics = evaluator.state.metrics
            for M in metrics.keys():
                if M == "elementwise_mae":
                    for i, val in enumerate(metrics[M].tolist()):
                        mlflow.log_metric(
                            f"{L}-{M}-{i}", val, trainer.state.epoch
                        )
                else:
                    mlflow.log_metric(
                        f"{L}-{M}", metrics[M], trainer.state.epoch
                    )

    return trainer
