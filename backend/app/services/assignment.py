"""Maximum-weight one-to-one assignment with an explicit unmatched choice."""
import numpy as np


def match_scores(scores, threshold):
    # scipy is also a dependency of the installed Ultralytics runtime.
    from scipy.optimize import linear_sum_assignment

    scores = np.asarray(scores, dtype=float)
    if scores.ndim != 2 or not scores.size:
        return []
    rows, cols = scores.shape
    # Each row can choose its own dummy column instead of an invalid match.
    weights = np.zeros((rows, cols + rows))
    weights[:, :cols] = np.where(scores >= threshold, scores, -1e6)
    row_ids, col_ids = linear_sum_assignment(weights, maximize=True)
    return [(int(r), int(c)) for r, c in zip(row_ids, col_ids)
            if c < cols and scores[r, c] >= threshold]
