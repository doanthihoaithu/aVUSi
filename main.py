import sys
import os

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from metrics.ffvus.ffvus_metrics import FFVUS
from metrics.interpretability_metrics import IndependentNDCG, AVUSI



def generate_demo_data(
    T: int = 1000,
    d: int = 5,
    anomalous_areas: tuple[tuple[int, int], ...] = ((100, 150), (400, 470), (700, 760)),
    normal_areas: tuple[tuple[int, int], ...] = ((250, 290), (580, 620)),
    seed: int = 42,
) -> dict:
    """
    Generate synthetic demo data for evaluating the aVUSi metric.

    For each anomalous area, 1–3 dimensions are randomly selected as anomalous.
    Per-dimension scores peak only in the windows where that dimension is anomalous.
    S additionally has high scores in normal_areas (false alarms — no dimension is
    truly anomalous there), simulating a detector that fires spuriously.

    Returns a dict with keys: DL, L, S, dim_values, DCM, anomalous_dims_per_area.
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

    # S: max over per-dimension scores, then overlay high scores in normal_areas
    # to simulate false alarms (high S despite L=0, DL=0 in those windows).
    S = dim_values.max(axis=1)
    for start, end in normal_areas:
        length = end - start
        peak = np.sin(np.linspace(0, np.pi, length))
        S[start:end] = np.maximum(S[start:end], 0.6 + 0.4 * peak + rng.uniform(-0.05, 0.05, size=length))
    S = np.clip(S, 0.0, 1.0)

    # DCM: softmax of dim_values + uniform noise, then re-normalize so each row sums to 1
    exp_vals = np.exp(dim_values - dim_values.max(axis=1, keepdims=True))
    # Reorder dimensions in exp_vals before forming DCM to simulate interpretability < 1 for anomalous areas
    exp_vals = exp_vals[:,[1,0,2,3,4]]
    # noise = rng.uniform(0.5, 1.0, size=(T, d))
    # exp_vals = exp_vals + noise
    DCM = exp_vals / exp_vals.sum(axis=1, keepdims=True)

    return {
        "DL": DL,
        "L": L,
        "S": S,
        "X": dim_values,
        "DCM": DCM,
        "anomalous_dims_per_area": anomalous_dims_per_area,
        "normal_areas": normal_areas,
    }


def plot_demo_data(data: dict, out_dir=None) -> None:
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
    X = data["X"]
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
        ax.plot(t, X[:, dim], color="steelblue", linewidth=0.9)
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
    if out_dir == None:
        out_dir = './'
    fig.savefig(os.path.join(out_dir, "demo_data.png"), dpi=150, bbox_inches="tight")
    plt.show()


def plot_vusi_curve(avusi_results: dict, k: int, w: int, M: int, vus_pr: float, indep_ndcg: float, out_dir=None) -> None:
    """
    Plot the VUSi(m) curve, shade the area under it (= aVUSi), and draw VUS-PR
    and IndepNDCG as horizontal reference lines.

    Args:
        avusi_results: dict returned by AVUSI.score_for_different_k().
        k: NDCG cutoff used when computing the results.
        w: smoothing window used when computing the results.
        M: number of sensitivity levels used when computing the results.
        vus_pr: VUS-PR score — plotted as a horizontal reference line.
        indep_ndcg: IndepNDCG score — plotted as a horizontal reference line.
    """
    m_levels = np.asarray(avusi_results[f"n_interpretability_sensitivity_levels_{M}_M_list"])
    vusi_values = np.asarray(avusi_results[(k, w, M)]["vus_pr_list"])
    avusi_value = avusi_results[(k, w, M)]["value"]

    fig, ax = plt.subplots(figsize=(8, 4))

    # VUSi(m) curve and shaded area representing aVUSi
    ax.plot(m_levels, vusi_values, color="steelblue", linewidth=2.0, label=r"VUSi$(m)$")
    ax.fill_between(m_levels, vusi_values, alpha=0.25, color="steelblue")

    # aVUSi annotation placed at the centre of the shaded area
    mid_idx = len(m_levels) // 2
    ax.annotate(
        f"aVUSi = {avusi_value:.4f}",
        xy=(m_levels[mid_idx], vusi_values[mid_idx] / 2),
        ha="center", va="center",
        fontsize=11, color="steelblue", fontweight="bold",
    )

    # VUS-PR horizontal reference line with inline label above
    ax.axhline(vus_pr, color="tomato", linewidth=1.5, linestyle="--",
               label=f"VUS-PR = {vus_pr:.4f}")
    ax.text(
        m_levels[-1], vus_pr, f"VUS-PR = {vus_pr:.4f}",
        ha="right", va="bottom",
        fontsize=10, color="tomato", fontweight="bold",
    )

    # IndepNDCG horizontal reference line with inline label above
    ax.axhline(indep_ndcg, color="seagreen", linewidth=1.5, linestyle="--",
               label=f"IndepNDCG = {indep_ndcg:.4f}")
    ax.text(
        m_levels[-1], indep_ndcg, f"IndepNDCG = {indep_ndcg:.4f}",
        ha="right", va="bottom",
        fontsize=10, color="seagreen", fontweight="bold",
    )

    ax.set_xlabel(r"Sensitivity level $m$", fontsize=11)
    ax.set_ylabel(r"VUSi$(m)$", fontsize=11)
    ax.set_title(
        rf"VUSi curve  —  $k={k}$, $w={w}$, $M={M}$",
        fontsize=12, fontweight="bold",
    )
    ax.set_xlim(m_levels[0], m_levels[-1])
    ax.set_ylim(0.0, 1.05)
    ax.legend(fontsize=10)
    ax.grid(True, linestyle=":", alpha=0.5)

    plt.tight_layout()
    if out_dir == None:
        out_dir = './'
    fig.savefig(os.path.join(out_dir, "vusi_curve.png"), dpi=150, bbox_inches="tight")
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

    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures", "usage_in_readme")
    os.makedirs(out_dir, exist_ok=True)
    #Visualize generated demo data
    plot_demo_data(demo, out_dir)

    #Start computing quality measures: VUS-PR (independent accuracy), IndepNDCG (independent interpretability), aVUSi (combined)
    X= demo["X"].astype(np.float64)
    S = demo["S"].astype(np.float64)
    L = demo["L"].astype(np.float64)
    DL = demo["DL"].astype(np.float64)
    DCM = demo["DCM"].astype(np.float64)

    default_k = X.shape[1]//2
    default_smoothing_window_w = 5
    default_sensitivity_levels_m = 50

    # VUS-PR (accuracy)
    vus_pr_metric = FFVUS(slope=10)
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
        vus_n_thresholds=50,
    )
    avusi_results = avusi_metric.score_for_different_k(
        y_true_univariate=L,
        y_score_univariate=S,
        y_true_multivariate=DL,
        dimension_contribution_multivariate=DCM,
    )

    avusi_value = avusi_results[(default_k, default_smoothing_window_w, default_sensitivity_levels_m)]["value"]
    print(f"aVUSi    = {avusi_value:.4f}")

    m_list = avusi_results[f'n_interpretability_sensitivity_levels_{default_sensitivity_levels_m}_M_list']
    vusi_list = avusi_results[(default_k, default_smoothing_window_w, default_sensitivity_levels_m)]['vus_pr_list']
    assert len(m_list) == len(vusi_list), "Length of m_list and vus_pr_list should be the same"
    plot_vusi_curve(avusi_results,
                    k=default_k,
                    w=default_smoothing_window_w,
                    M=default_sensitivity_levels_m,
                    vus_pr=vus_pr_result["value"],
                    indep_ndcg=indep_ndcg_value,
                    out_dir=out_dir)
