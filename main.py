import sys
import os

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


def generate_demo_data(
    T: int = 1000,
    d: int = 5,
    anomalous_areas: tuple[tuple[int, int], ...] = ((100, 150), (400, 470), (700, 760)),
    seed: int = 42,
) -> dict:
    """
    Generate synthetic demo data for evaluating the aVUSi metric.

    For each anomalous area, 1–3 dimensions are randomly selected as anomalous.
    Anomaly scores S are simulated to peak within anomalous areas (with noise).

    Returns a dict with keys: DL (T, d), L (T,), S (T,), anomalous_dims (per area).
    """
    rng = np.random.default_rng(seed)

    DL = np.zeros((T, d), dtype=np.int32)
    anomalous_dims_per_area = []

    for start, end in anomalous_areas:
        n_anom_dims = rng.integers(1, min(4, d + 1))  # 1, 2, or 3 (capped at d)
        chosen_dims = rng.choice(d, size=n_anom_dims, replace=False)
        DL[start:end, chosen_dims] = 1
        anomalous_dims_per_area.append((start, end, sorted(chosen_dims.tolist())))

    # L: a timestamp is anomalous if any dimension is anomalous
    L = (DL.sum(axis=1) > 0).astype(np.int32)

    # Per-dimension scores: base noise in [0, 0.3]; sine peak in [0.6, 1.0] only
    # for timestamps where that dimension is marked anomalous in DL.
    dim_values = rng.uniform(0.0, 0.3, size=(T, d))
    for dim in range(d):
        # find contiguous anomalous segments for this dimension
        anom = DL[:, dim]
        i = 0
        while i < T:
            if anom[i] == 1:
                j = i
                while j < T and anom[j] == 1:
                    j += 1
                length = j - i
                peak = np.sin(np.linspace(0, np.pi, length))
                dim_values[i:j, dim] = 0.6 + 0.4 * peak + rng.uniform(-0.05, 0.05, size=length)
                i = j
            else:
                i += 1
    dim_values = np.clip(dim_values, 0.0, 1.0)

    # S: aggregate per-dimension scores by taking the max across dimensions
    S = dim_values.max(axis=1)

    # DCM: softmax-normalize dim_values over the d dimension axis at each timestamp
    # so each row sums to 1, reflecting relative contribution of each dimension
    exp_vals = np.exp(dim_values - dim_values.max(axis=1, keepdims=True))  # numerically stable
    DCM = exp_vals / exp_vals.sum(axis=1, keepdims=True)

    return {
        "DL": DL,
        "L": L,
        "S": S,
        "dim_values": dim_values,
        "DCM": DCM,
        "anomalous_dims_per_area": anomalous_dims_per_area,
    }


def plot_demo_data(data: dict) -> None:
    """
    Plot generated demo data.

    Subplots (top to bottom):
      - One subplot per dimension: anomaly score S overlaid with red shading
        where that dimension is marked anomalous in DL.
      - One subplot for the univariate label L.
    """
    S = data["S"]
    DL = data["DL"]
    L = data["L"]
    dim_values = data["dim_values"]
    T, d = DL.shape
    t = np.arange(T)

    # layout: d dimension subplots + 1 for S + 1 for L
    n_rows = d + 2
    height_ratios = [2] * d + [2, 1]
    fig, axes = plt.subplots(
        n_rows, 1, figsize=(7, 1 * n_rows),
        sharex=True, gridspec_kw={"height_ratios": height_ratios}
    )
    fig.suptitle("aVUSi Demo Data", fontsize=13, fontweight="bold", y=1.01)

    anom_patch = mpatches.Patch(color="red", alpha=0.25, label="Anomalous region (DL)")

    def _shade_segments(ax, mask):
        """Shade contiguous True regions of mask on ax."""
        in_region = False
        seg_start = 0
        for i in range(T):
            if mask[i] and not in_region:
                seg_start = i
                in_region = True
            elif not mask[i] and in_region:
                ax.axvspan(seg_start, i, color="red", alpha=0.25)
                in_region = False
        if in_region:
            ax.axvspan(seg_start, T, color="red", alpha=0.25)

    # --- per-dimension subplots ---
    for dim in range(d):
        ax = axes[dim]
        ax.plot(t, dim_values[:, dim], color="steelblue", linewidth=0.9)
        _shade_segments(ax, DL[:, dim] == 1)
        ax.set_ylabel(f"Dim {dim}", fontsize=9)
        ax.set_ylim(-0.05, 1.1)
        ax.tick_params(labelsize=8)
        ax.yaxis.set_major_locator(plt.MultipleLocator(0.5))

    # --- anomaly score S subplot ---
    ax_s = axes[d]
    ax_s.plot(t, S, color="darkorange", linewidth=1.1)
    _shade_segments(ax_s, L == 1)
    ax_s.set_ylabel("S", fontsize=9)
    ax_s.set_ylim(-0.05, 1.1)
    ax_s.tick_params(labelsize=8)
    ax_s.yaxis.set_major_locator(plt.MultipleLocator(0.5))

    # --- univariate label subplot ---
    ax_l = axes[-1]
    ax_l.fill_between(t, L, step="post", color="tomato", alpha=0.7)
    ax_l.plot(t, L, drawstyle="steps-post", color="darkred", linewidth=0.8)
    ax_l.set_ylabel("L", fontsize=9)
    ax_l.set_ylim(-0.05, 1.3)
    ax_l.set_yticks([0, 1])
    ax_l.tick_params(labelsize=8)
    ax_l.set_xlabel("Timestamp", fontsize=9)

    # shared legend
    handles = [
        plt.Line2D([0], [0], color="steelblue", linewidth=1.5, label="Per-dim value"),
        plt.Line2D([0], [0], color="darkorange", linewidth=1.5, label="Anomaly score S"),
        anom_patch,
        mpatches.Patch(color="tomato", alpha=0.7, label="Univariate label L"),
    ]
    fig.legend(handles=handles, loc="upper right", fontsize=8, framealpha=0.9)

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    demo = generate_demo_data()

    print(f"T={len(demo['L'])}, d={demo['DL'].shape[1]}")
    print(f"Anomalous timestamps: {demo['L'].sum()}")
    print("Anomalous areas (start, end, dims):")
    for entry in demo["anomalous_dims_per_area"]:
        print(f"  [{entry[0]}:{entry[1]}] → dims {entry[2]}")
    print(f"S range: [{demo['S'].min():.3f}, {demo['S'].max():.3f}]")
    print(f"DL shape: {demo['DL'].shape}, L shape: {demo['L'].shape}")
    print(f"DCM shape: {demo['DCM'].shape}, row-sum check: {demo['DCM'].sum(axis=1)[:3].round(6)}")


    plot_demo_data(demo)

    # --- metric computation ---
    # Add 3_metric_calculator to path so its internal imports resolve
    # metric_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "3_metric_calculator")
    # if metric_dir not in sys.path:
    #     sys.path.insert(0, metric_dir)

    from metrics.ffvus.ffvus_metrics import FFVUS
    from metrics.interpretability_metrics import IndependentNDCG, AVUSI

    S = demo["S"].astype(np.float64)
    L = demo["L"].astype(np.float64)
    DL = demo["DL"].astype(np.float64)
    DCM = demo["DCM"].astype(np.float64)

    default_k = S.shape[0]//2
    default_smoothing_window_w = 5
    default_sensitivity_levels_m = 50

    # VUS-PR (accuracy)
    vus_pr_metric = FFVUS(slope=100)
    vus_pr_result = vus_pr_metric.score(L, S)
    print(f"\nVUS-PR   = {vus_pr_result['value']:.4f}")

    # IndepNDCG (interpretability), fixed k=5, w=10
    indep_ndcg_metric = IndependentNDCG(
        max_k=5,
        max_smoothing_window=10,
        k_slide=1,
        smoothing_window_slide=1,
        fix_k=True,
        fix_smoothing_window=True,
        default_selected_k=default_k,
        default_selected_smoothing_window=default_smoothing_window_w,
    )
    indep_ndcg_results = indep_ndcg_metric.score_for_different_k(
        y_score=S,
        y_true_multivariate=DL,
        dimension_contribution=DCM,
    )
    indep_ndcg_value = indep_ndcg_results[(default_k, default_smoothing_window_w)]["value"]
    print(f"IndepNDCG = {indep_ndcg_value:.4f}")

    # aVUSi (combined), fixed k=5, w=10, M=50
    avusi_metric = AVUSI(
        max_k=5,
        max_smoothing_window=5,
        max_n_interpretability_sensitivity_levels=50,
        k_slide=1,
        smoothing_window_slide=1,
        n_interpretability_sensitivity_levels_slide=1,
        fix_k=True,
        fix_smoothing_window=True,
        fix_n_interpretability_sensitivity_levels=True,
        default_selected_k=default_k,
        default_selected_smoothing_window=default_smoothing_window_w,
        default_selected_n_interpretability_sensitivity_levels=default_sensitivity_levels_m,
        vus_slope=5,
        vus_n_thresholds=150,
    )
    avusi_results = avusi_metric.score_for_different_k(
        y_true_univariate=L,
        y_score_univariate=S,
        y_true_multivariate=DL,
        dimension_contribution_multivariate=DCM,
    )
    avusi_value = avusi_results[(default_k, default_smoothing_window_w, default_sensitivity_levels_m)]["value"]
    print(f"aVUSi    = {avusi_value:.4f}")