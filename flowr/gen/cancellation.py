"""Cooperative cancellation for flow-matching generation.

Generation runs a multi-step ODE integration loop inside the model's
``_generate``. A single batch runs all integration steps before returning, so a
cancellation request issued mid-batch is not honoured until the whole batch
finishes — blocking the worker thread (and pinning its GPU/MPS memory) in the
meantime. Threading a ``should_cancel`` predicate down into the integration loop
lets generation abort between steps instead.
"""

from typing import Callable, Optional

ShouldCancel = Optional[Callable[[], bool]]


class GenerationCancelled(Exception):
    """Raised inside the integration loop when cancellation is requested."""


def raise_if_cancelled(should_cancel: ShouldCancel, step: Optional[int] = None) -> None:
    """Raise :class:`GenerationCancelled` if ``should_cancel`` returns truthy.

    Args:
        should_cancel: Zero-arg predicate returning ``True`` to abort. ``None``
            (the default) disables cancellation, so existing callers that don't
            pass a predicate are unaffected.
        step: Optional integration-step index, included in the exception message
            for diagnostics.
    """
    if should_cancel is not None and should_cancel():
        where = f" at integration step {step}" if step is not None else ""
        raise GenerationCancelled(f"Generation cancelled{where}")
