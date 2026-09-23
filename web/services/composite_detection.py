"""Web adapter: committed composite clustering with the validated web loaders.

Confidence is the winning raw model weight, not a calibrated probability.
The CLI constructor/inference path is deliberately not used: it silently skips
failures and fixes both the device and merging radius.
"""
import numpy as np


class CompositeDetector:
    def __init__(self, members, distance):
        self.members = members  # (id, particle label, loaded model)
        self.distance = distance

    def detect(self, image, params, detect_arrays):
        from composite_model import CompositeLodeSTAR

        points, maps = {}, {}
        for model_id, label, model in self.members:
            detections, weights, _, _ = detect_arrays(model, image, params, None)
            if weights is None or not np.isfinite(weights).all() or not np.isfinite(detections).all():
                raise ValueError(f"Non-finite/missing composite output for {label}")
            points[label], maps[label] = detections, weights
        merged = CompositeLodeSTAR._merge_detections(None, points, self.distance)
        labels, scores, ids = [], [], []
        for x, y in merged:
            weights = [weight[int(np.clip(round(y), 0, weight.shape[0]-1)),
                              int(np.clip(round(x), 0, weight.shape[1]-1))]
                       for weight in maps.values()]
            winner = int(np.argmax(weights))
            model_id, label, _ = self.members[winner]
            labels.append(label)
            ids.append(model_id)
            scores.append(float(weights[winner]))
        return merged, np.maximum.reduce(list(maps.values())), labels, scores, ids
