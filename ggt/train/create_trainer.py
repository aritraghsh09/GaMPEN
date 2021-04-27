import mlflow

from ignite.engine import (
    Events,
    create_supervised_trainer,
    create_supervised_evaluator,
)
from ignite.metrics import MeanAbsoluteError, MeanSquaredError, Loss

from ggt.metrics import ElementwiseMae


def create_trainer(model, optimizer, criterion, loaders, device):
    """Set up Ignite trainer and evaluator."""
    trainer = create_supervised_trainer(
        model, optimizer, criterion, device=device
    )

    metrics = {
        "mae": MeanAbsoluteError(),
        "mse": MeanSquaredError(),
        "elementwise_mae": ElementwiseMae(),
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
                mlflow.log_metric(f"{L}-{M}", metrics[M], 0)

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_devel_results(trainer):
        evaluator.run(loaders["devel"])
        metrics = evaluator.state.metrics
        for M in metrics.keys():
            mlflow.log_metric(f"devel-{M}", metrics[M], trainer.state.epoch)

    @trainer.on(Events.COMPLETED)
    def log_results_end(trainer):
        for L, loader in loaders.items():
            evaluator.run(loader)
            metrics = evaluator.state.metrics
            for M in metrics.keys():
                mlflow.log_metric(f"{L}-{M}", metrics[M], trainer.state.epoch)

    return trainer
