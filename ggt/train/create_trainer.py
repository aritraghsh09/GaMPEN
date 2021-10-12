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
        "elementwise_mae": ElementwiseMae(),
        "mse": MeanSquaredError(),
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

#    @trainer.on(Events.EPOCH_COMPLETED)
#    def log_STN_weights(trainer):
#            if hasattr(model, "spatial_transform") or hasattr(
#                model.module, "spatial_transform"
#            ):
#                if hasattr(model, "spatial_transform"):
#                    fc_loc = model.fc_loc
#                else:
#                    fc_loc = model.module.fc_loc
#
#                for i, param in enumerate(fc_loc.parameters()):
#                    mlflow.log_param(f"STN_weights-{i}", param.data.tolist())

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
