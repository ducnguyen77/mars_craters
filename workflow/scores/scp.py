from __future__ import division

import numpy as np

from .detection_base import DetectionBaseScoreType
from ._circles import circle_map


def scp_single(y_true, y_pred, shape, minipatch=None):
    """
    L1 distance between superposing bounding box cylinder or prism maps.

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
    float : score for a given patch, the lower the better

    """
    image = np.abs(circle_map(y_true, y_pred, shape))
    if minipatch is not None:
        image = image[minipatch[0]:minipatch[1], minipatch[2]:minipatch[3]]
    # Sum all the pixels
    score = image.sum()

    return score


class SCP(DetectionBaseScoreType):
    is_lower_the_better = True
    minimum = 0.0
    maximum = 1.0

    def __init__(self, shape, name='scp', precision=2, conf_threshold=0.5,
                 minipatch=None):
        self.shape = shape
        self.name = name
        self.precision = precision
        self.conf_threshold = conf_threshold
        self.minipatch = minipatch

    def detection_score(self, y_true, y_pred):
        """
        Score based on a matching by reprojection of craters on mask-map.

        True craters are projected positively, predicted craters negatively,
        so they can cancel out. Then the sum of the absolute value of the
        residual map is taken.

        The best score value for a perfect match is 0.
        The worst score value for a given patch is the sum of all crater
        instances in both `y_true` and `y_pred`.

        Parameters
        ----------
        y_true : list of list of tuples (x, y, radius)
            List of coordinates and radius of actual craters for set of patches
        y_pred : list of list of tuples (x, y, radius)
            List of coordinates and radius of predicted craters for set of
            patches

        Returns
        -------
        float : score for a given patch, the lower the better

        """
        scores = [scp_single(t, p, self.shape, self.minipatch)
                  for t, p in zip(y_true, y_pred)]
        n_true_craters = np.sum([len(t) for t in y_true])
        n_pred_craters = np.sum([len(t) for t in y_pred])
        return np.sum(scores) / (n_true_craters + n_pred_craters)
