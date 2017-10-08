from __future__ import division

from rampwf.score_types.base import BaseScoreType


class DetectionBaseScoreType(BaseScoreType):
    """Common abstract base type for detection scores.

    Implements `__call__` by selecting all prediction detections with
    a confidence higher than `conf_threshold`. It assumes that the child
    class implements `detection_score`.
    """

    conf_threshold = 0.5

    def __call__(self, y_true, y_pred, conf_threshold=None):
        if conf_threshold is None:
            conf_threshold = self.conf_threshold
        y_pred_above_confidence = [
            [bounding_region for (bounding_region, p) in y_pred_patch
             if p > conf_threshold]
            for y_pred_patch in y_pred]
        return self.detection_score(y_true, y_pred_above_confidence)
