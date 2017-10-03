from __future__ import division

import numpy as np

from rampwf.score_types.base import BaseScoreType

from ._circles import circle_map


def mask_detection(y_true, y_pred):
    """
    Score based on a matching by reprojection of craters on mask-map

    True craters are projected positively, predicted craters negatively,
    so they can cancel out. Then the sum of the absolute value of the
    residual map is taken.

    The best score value for a perfect match is 0.
    The worst score value for a given patch is the sum of all crater
    instances in both `y_true` and `y_pred`.

    Parameters
    ----------
    y_true : list of tuples (x, y, radius)
        List of coordinates and radius of actual craters in a patch
    y_pred : list of tuples (x, y, radius)
        List of coordinates and radius of craters predicted in the patch

    Returns
    -------
    float : score for a given patch, the higher the better

    """
    image = np.abs(circle_map(y_true, y_pred))

    # Sum all the pixels
    score = image.sum()

    return score


class MaskDetection(BaseScoreType):
    is_lower_the_better = True
    minimum = 0.0
    maximum = np.inf

    def __init__(self, name='mask_detection', precision=2):
        self.name = name
        self.precision = precision

    def __call__(self, y_true, y_pred):
        scores = [mask_detection(t, p) for t, p in zip(y_true, y_pred)]
        true_craters = [len(t) for t in y_true]
        pred_craters = [len(t) for t in y_pred]
        return np.sum(scores) / (np.sum(true_craters) + np.sum(pred_craters))
