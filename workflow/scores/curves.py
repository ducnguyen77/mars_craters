from __future__ import division

import numpy as np
import matplotlib.pyplot as plt

from rampwf.score_types.base import BaseScoreType


def precision_recall_curve(y_true, y_pred, conf_thresholds, iou_threshold=0.5):
    from .precision_recall import precision, recall

    ps = []
    rs = []

    for conf_threshold in conf_thresholds:
        y_pred_temp = [
            [(x, y, r) for (x, y, r, p) in y_pred_patch if p > conf_threshold]
            for y_pred_patch in y_pred]
        ps.append(precision(y_true, y_pred_temp, iou_threshold=iou_threshold))
        rs.append(recall(y_true, y_pred_temp, iou_threshold=iou_threshold))

    return np.array(ps), np.array(rs)


def mask_detection_curve(y_true, y_pred, conf_thresholds):
    from .mask import mask_detection

    ms = []

    for conf_threshold in conf_thresholds:
        y_pred_temp = [
            [(x, y, r) for (x, y, r, p) in y_pred_patch if p > conf_threshold]
            for y_pred_patch in y_pred]
        ms.append(mask_detection(y_true, y_pred_temp))

    return np.array(ms)


def ospa_curve(y_true, y_pred, conf_thresholds):
    from .ospa import ospa

    os = []

    for conf_threshold in conf_thresholds:
        y_pred_temp = [
            [(x, y, r) for (x, y, r, p) in y_pred_patch if p > conf_threshold]
            for y_pred_patch in y_pred]
        os.append(ospa(y_true, y_pred_temp))

    return np.array(os)


def average_precision_interpolated(ps, rs):
    ps = np.asarray(ps)
    rs = np.asarray(rs)

    p_at_r = []

    for r in np.arange(0, 1.1, 0.1):
        p = np.array(ps)[np.array(rs) >= r]
        if p.size:
            p_at_r.append(np.nanmax(p))
        else:
            p_at_r.append(0)

    ap = np.mean(p_at_r)
    return ap


def plot_precision_recall_curve(ps, rs):

    ap = average_precision_interpolated(ps, rs)

    fig, ax = plt.subplots()
    ax.plot(rs, ps, 'o-')
    ax.set_xlabel('Recall', fontsize=16)
    ax.set_ylabel('Precision', fontsize=16)
    ax.text(0.7, 0.9, 'AP = {:.2f}'.format(ap), fontsize=16)

    return fig, ax


class AveragePrecision(BaseScoreType):
    is_lower_the_better = False
    minimum = 0.0
    maximum = 1.0

    def __init__(self, name='average_precision', precision=2):
        self.name = name
        self.precision = precision

    def __call__(self, y_true, y_pred):
        ps, rs = precision_recall_curve(y_true, y_pred)
        return average_precision_interpolated(ps, rs)
