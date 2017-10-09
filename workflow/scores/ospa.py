from __future__ import division

import numpy as np

from .detection_base import DetectionBaseScoreType
from .precision_recall import _match_tuples


def score_craters_on_patch(y_true, y_pred):
    """
    Main OSPA score for single patch.

    Parameters
    ----------
    y_true : list of tuples (x, y, radius)
        List of coordinates and radius of actual craters in a patch
    y_pred : list of tuples (x, y, radius)
        List of coordinates and radius of craters predicted in the patch

    Returns
    -------
    float : score for a given path, the lower the better

    """
    y_true = np.atleast_2d(y_true).T
    y_pred = np.atleast_2d(y_pred).T
    score = ospa_single(y_true, y_pred)
    return score


def ospa_single(x_arr, y_arr, cut_off=1, minipatch=None):
    """
    OSPA score on single patch. See docstring of `ospa` for more info.

    Parameters
    ----------
    x_arr, y_arr : ndarray of shape (3, x)
        arrays of (x, y, radius)
    cut_off : float, optional (default is 1)
        penalizing value for wrong cardinality
    minipatch : list of int, optional
        Bounds of the internal scoring patch (default is None)

    Returns
    -------
    float: distance between input arrays

    """
    x_size = x_arr.size
    y_size = y_arr.size

    _, m = x_arr.shape
    _, n = y_arr.shape

    if m > n:
        return ospa_single(y_arr, x_arr, cut_off)

    # NO CRATERS
    # ----------
    # GOOD MATCH
    if x_size == 0 and y_size == 0:
        return 0

    # BAD MATCH
    if x_size == 0 or y_size == 0:
        return cut_off

    # minipatch cuts
    if minipatch is not None:
        row_min, row_max, col_min, col_max = minipatch

        x_arr_cut = ((x_arr[0] >= col_min) & (x_arr[0] < col_max) &
                     (x_arr[1] >= row_min) & (x_arr[1] < row_max))
        y_arr_cut = ((y_arr[0] >= col_min) & (y_arr[0] < col_max) &
                     (y_arr[1] >= row_min) & (y_arr[1] < row_max))

        x_arr = x_arr[x_arr_cut]
        y_arr = y_arr[y_arr_cut]

    # OSPA METRIC
    _, _, ious = _match_tuples(x_arr.T.tolist(), y_arr.T.tolist())
    iou_score = ious.sum()

    distance_score = m - iou_score
    cardinality_score = cut_off * (n - m)

    dist = 1 / n * (distance_score + cardinality_score)

    return dist


class OSPA(DetectionBaseScoreType):
    is_lower_the_better = True
    minimum = 0.0
    maximum = 1.0

    def __init__(self, name='ospa', precision=2, conf_threshold=0.5,
                 minipatch=None):
        self.name = name
        self.precision = precision
        self.conf_threshold = conf_threshold
        self.minipatch = minipatch

    def detection_score(self, y_true, y_pred):
        """Optimal Subpattern Assignment (OSPA) metric for IoU score.

        This metric provides a coherent way to compute the miss-distance
        between the detection and alignment of objects. Among all
        combinations of true/predicted pairs, if finds the best alignment
        to minimise the distance, and still takes into account missing
        or in-excess predicted values through a cardinality score.

        The lower the value the smaller the distance.

        Parameters
        ----------
        y_true, y_pred : list of list of tuples

        Returns
        -------
        float: distance between input arrays

        References
        ----------
        http://www.dominic.schuhmacher.name/papers/ospa.pdf

        """
        scores = [score_craters_on_patch(t, p, self.minipatch)
                  for t, p in zip(y_true, y_pred)]
        weights = [len(t) for t in y_true]
        return np.average(scores, weights=weights)
