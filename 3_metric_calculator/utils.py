import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

def estimate_dimension_contribution_by_anomaly_score_with_a_buffer(
    y_score: np.ndarray,
    dimension_contribution: np.ndarray,
    smoothing_window: int
) -> np.ndarray:
    """
    Estimate the contribution of each dimension with a buffer.

    Args:
        y_score: 1D array of shape (T,) with anomaly scores.
        dimension_contribution: 2D array of shape (T, D) with contribution
            scores for each dimension at each timestamp.
        smoothing_window: number of timestamps to consider before and after
            each timestamp.

    Returns:
        2D array of shape (T, D) with smoothed contribution scores,
        weighted by anomaly scores via trapezoidal integration.
    """
    if smoothing_window == 0:
        return dimension_contribution

    T, D = dimension_contribution.shape
    w = smoothing_window
    window_len = 2 * w + 1
    # print('y_score.shape', y_score.shape)

    y_score =y_score.reshape(-1)

    # Pad along the time axis so every timestamp has a full window
    y_padded  = np.pad(y_score, (w,w), mode="edge")          # (T + 2w,)
    # print('y_padded.shape', y_padded.shape)
    dc_padded = np.pad(dimension_contribution,
                       ((w, w), (0, 0)), mode="edge")          # (T + 2w, D)

    # sliding_window_view on 1D score — shape (T, window_len)
    y_windows = sliding_window_view(y_padded, window_len,axis=0)      # (T, window_len)
    # print('dc_padded.shape', dc_padded.shape)

    # sliding_window_view on 2D array — apply along axis=0 only
    # result shape: (T, D, window_len) — note D and window_len are swapped
    dc_windows = sliding_window_view(
        dc_padded, window_len, axis=0
    )                                                          # (T, D, window_len)

    # Align axes: bring window axis next to time → (T, window_len, D)
    dc_windows = dc_windows.transpose(0, 2, 1)                # (T, window_len, D)

    # Weighted integrand and trapezoidal integration along window axis
    integrand = y_windows[:, :, np.newaxis] * dc_windows      # (T, window_len, D)
    estimated = np.trapz(integrand, axis=1)                   # (T, D)

    return estimated
