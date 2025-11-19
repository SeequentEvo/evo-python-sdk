#  Copyright © 2025 Bentley Systems, Incorporated
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

import functools
from collections.abc import Iterator, Sequence
from threading import Lock
from typing import Callable, TypeVar

from ..interfaces import IFeedback

__all__ = [
    "NoFeedback",
    "PartialFeedback",
    "iter_with_fb",
    "split_feedback",
]

_N_DIGITS = 4  # Let's maintain a user-friendly number of dp.


class _NoFeedback(IFeedback):
    def progress(self, progress: float, message: str | None = None) -> None:
        pass


NoFeedback = _NoFeedback()
"""A default feedback object that does nothing. Use this when no feedback is needed."""


class PartialFeedback(IFeedback):
    """A wrapper for IFeedback objects that subdivides the feedback range."""

    def __init__(self, parent: IFeedback, start: float, end: float) -> None:
        """
        :param parent: The parent feedback object to subdivide.
        :param start: The start of the partial feedback range.
        :param end: The end of the partial feedback range.
        """
        self.__offset = start
        self.__factor = end - start

        self.__lock = Lock()
        self.__parent = parent

    def progress(self, progress: float, message: str | None = None) -> None:
        """Update the progress of the feedback object.

        :param progress: The progress value, between 0 and 1.
        :param message: An optional message to display.
        """
        partial_progress = round(self.__offset + (progress * self.__factor), ndigits=_N_DIGITS)
        with self.__lock:
            self.__parent.progress(partial_progress, message)


T = TypeVar("T")


def iter_with_fb(elements: Sequence[T], feedback: IFeedback | None = None) -> Iterator[tuple[T, IFeedback]]:
    """Iterate over a sequence of elements, dividing feedback uniformly throughout the range.

    :param elements: The sequence of elements to iterate over.
    :param feedback: A feedback object to subdivide.

    :yields: A tuple containing a sequence element, and a new feedback object for that element.
    """
    if len(elements) == 0:
        return

    fb_part_size = 1 / len(elements)
    for i, element in enumerate(elements):
        if feedback is None:
            yield element, NoFeedback
        else:
            start = round(i * fb_part_size, ndigits=_N_DIGITS)
            end = round((i + 1) * fb_part_size, ndigits=_N_DIGITS)
            yield element, PartialFeedback(feedback, start, end)
            feedback.progress(end)


class _ConcurrentFeedback(IFeedback):
    """A feedback object that reports progress to a ConcurrentFeedbackGroup, allowing aggregation of progress from
    multiple sources concurrently.
    """

    def __init__(self, callback: Callable[[float, str | None], None]) -> None:
        self.__callback = callback
        self.__progress = 0.0

    def progress(self, progress: float, message: str | None = None) -> None:
        """Update the progress of the feedback object.

        :param progress: The progress value, between 0 and 1.
        :param message: An optional message to display.
        """
        diff = progress - self.__progress
        self.__progress = progress
        self.__callback(diff, message)


class _ConcurrentFeedbackGroup:
    """A group of feedbacks that can be updated concurrently, aggregating their progress."""

    def __init__(self, parent: IFeedback) -> None:
        self.__lock = Lock()
        self.__parent = parent
        self.__total_weight = 0.0
        self.__progress = 0.0
        self.__n_feedbacks = 0

    def _update(self, message: str | None = None) -> None:
        if self.__total_weight == 0:
            progress = self.__progress / self.__n_feedbacks
        else:
            progress = self.__progress / self.__total_weight
        self.__parent.progress(round(progress, _N_DIGITS), message)

    def _progress(self, weight: float, progress_change: float, message: str | None) -> None:
        with self.__lock:
            if self.__total_weight == 0:
                self.__progress += progress_change
            else:
                self.__progress += weight * progress_change
            self._update(message)

    def create_feedback(self, weight: float) -> IFeedback:
        with self.__lock:
            if self.__progress > 0:
                raise RuntimeError("Cannot create new feedback after progress has been reported.")
            self.__total_weight += weight
            self.__n_feedbacks += 1

        return _ConcurrentFeedback(functools.partial(self._progress, weight))


def split_feedback(feedback: IFeedback, weights: Sequence[float]) -> list[IFeedback]:
    """Split a feedback object into multiple feedback objects based on weights.

    Each of them can be updated concurrently, and their progress will be aggregated into the parent feedback object.

    Note, if all weights are zero, all feedback objects will be uniformly weighted.

    :param feedback: The parent feedback object to subdivide.
    :param weights: Weights for each partial feedback object.
    :returns: A list of ConcurrentFeedback objects.
    """
    group = _ConcurrentFeedbackGroup(feedback)
    return [group.create_feedback(weight) for weight in weights]
