import torch
from typing import Sequence, Union

from ignite.metrics import Metric
from ignite.metrics.metric import sync_all_reduce, reinit__is_reduced
from ignite.exceptions import NotComputableError


class ElementwiseMae(Metric):
    """
    Calculates the element-wise mean absolute error.
    """

    @reinit__is_reduced
    def reset(self) -> None:
        self._elementwise_sum_abs_errors = None
        self._num_examples = 0

    @reinit__is_reduced
    def update(self, output: Sequence[torch.Tensor]) -> None:
        # Unpack Sequence[Tensor] input
        y_pred, y = output

        # Initialize elementwise errors as zeros
        if self._elementwise_sum_abs_errors is None:
            self._elementwise_sum_abs_errors = torch.zeros_like(y_pred)

        # Sum absolute errors element-wise
        absolute_errors = torch.abs(y_pred - y.view_as(y_pred))
        self._elementwise_sum_abs_errors += absolute_errors
        self._num_examples += y.shape[0]

    @sync_all_reduce("_elementwise_sum_abs_errors", "_num_examples")
    def compute(self) -> Union[float, torch.Tensor]:
        if self._num_examples == 0:
            raise NotComputableError(
                "ElementwiseMae must have at least one "
                "example before it can be computed."
            )

        # Average by number of examples to produce elementwise MAE
        return self._elementwise_sum_abs_errors / self._num_examples
