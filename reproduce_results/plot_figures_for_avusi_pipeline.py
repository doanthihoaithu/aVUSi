from __future__ import annotations

import os.path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from scipy.ndimage import uniform_filter1d

from metrics.ffvus.ffvus_metrics import FFVUS

# ── Output directory ──────────────────────────────────────────────────────────
CURRENT_DIR = os.path.dirname(__file__)
OUTPUT_DIR = os.path.join(CURRENT_DIR, 'combined_figures')
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ═══════════════════════════════════════════════════════════════════════
# PART 1 — DEMO DATA GENERATION
# ═══════════════════════════════════════════════════════════════════════

def generate_avusi_demo_data(
    T: int = 1000,
    anomaly_regions: list[tuple] = None,
    interp_levels: list[float] = None,
    seed: int = 42,
) -> dict:
    """
    Generate synthetic demo data for aVUSi illustration figure.

    Args:
        T:               Length of the time series.
        anomaly_regions: List of (start, end) tuples defining anomaly segments.
                         Defaults to three segments with varying interpretability.
        interp_levels:   Interpretability level per anomaly region, values in [0,1].
                         1.0 = fully interpretable, 0.5 = partial, 0.0 = missed.
        vus_pr_ref:      VUS-PR reference value to show on the VUSi curve.
        seed:            Random seed for reproducibility.

    Returns:
        Dictionary with keys:
            t           : time axis, shape (T,)
            L           : univariate labels, shape (T,)
            S           : anomaly scores, shape (T,)
            I           : interpretability scores, shape (T,)
            PS_m0       : penalized scores at m=0.0, shape (T,)
            PS_m05      : penalized scores at m=0.5, shape (T,)
            PS_m1       : penalized scores at m=1.0, shape (T,)
            m_vals      : sensitivity sweep values, shape (n_m,)
            VUSi        : VUSi curve values, shape (n_m,)
            aVUSi       : scalar aVUSi value
            vus_pr      : VUS-PR reference value
            regions     : anomaly region list
            interp_levels: interpretability level per region
    """
    rng = np.random.default_rng(seed)

    if anomaly_regions is None:
        anomaly_regions = [(120, 170), (380, 420), (640, 680)]
    if interp_levels is None:
        interp_levels = [1.0, 0.5, 0.0]

    assert len(interp_levels) == len(anomaly_regions), \
        "interp_levels must have one value per anomaly region."

    t = np.arange(T)

    # ── Univariate labels ─────────────────────────────────────────────
    L = np.zeros(T, dtype=np.float32)
    for s, e in anomaly_regions:
        L[s:e] = 1.0

    # ── Anomaly scores ────────────────────────────────────────────────
    # Normal: Beta(2, 6) → unimodal, low-valued but visibly fluctuating
    # Anomalous: Beta(4, 1.5) → left-skewed, mostly high
    S = rng.beta(2, 6, T).astype(np.float32) * 0.38
    for s, e in anomaly_regions:
        pad = int(0.1 * (e - s))
        S[s + pad:e - pad] = rng.beta(4, 1.5, (e - pad) - (s + pad)).astype(np.float32) * 0.85 + 0.1
    # Two false-positive bursts in normal regions (high score, no ground-truth label)
    false_positive_regions = [(240, 275), (520, 555)]
    for s, e in false_positive_regions:
        S[s:e] = rng.beta(4, 1.5, e - s).astype(np.float32) * 0.80 + 0.10
    # Smooth to mimic realistic detector output
    S = uniform_filter1d(S, size=6)
    S = np.clip(S, 0.0, 1.0).astype(np.float32)

    # ── Interpretability scores ───────────────────────────────────────
    # Nonzero only at anomalous timestamps; constant per region
    I = np.zeros(T, dtype=np.float32)
    for (s, e), level in zip(anomaly_regions, interp_levels):
        I[s:e] = level

    # ── Penalized score sequences PS^m ────────────────────────────────
    def penalized(S, I, L, m):
        """PS^m = S ⊙ S^m_Interp, where S^m_Interp = I at anomalies, m elsewhere."""
        S_interp = np.where(L == 1, I, m)
        return (S * S_interp).astype(np.float32)

    PS_m0  = penalized(S, I, L, 0.0)
    PS_m05 = penalized(S, I, L, 0.5)
    PS_m1  = penalized(S, I, L, 1.0)

    # ── VUSi curve as a function of m ─────────────────────────────────
    # Simulated: peaks near m=0.2, declines toward m=1
    # Reflects that a detector with mixed interpretability (levels 1.0, 0.5, 0.0)
    # performs best at low-to-moderate sensitivity and is penalized at high m
    m_vals = np.linspace(0, 1, 50)
    m_middle_index = len(m_vals)//2
    if 0.5 not in m_vals:
        m_vals = sorted(list(m_vals) + [0.5])  # Ensure m=0.5 is included for annotation
        m_middle_index = m_vals.index(0.5)

    def vusi_curve(m_vals, S, I, L):
        # peak  = 0.82 * np.exp(-2.5 * (m - 0.18) ** 2)
        # decay = 0.35 * (1 - m) * 0.3
        # base  = vus_pr_ref * m * 0.6
        # return peak + decay + base
        vusi = []
        for m in m_vals:
            # tmp_score = interpretability_scores.copy()
            # tmp_score[univariate_label == 0] = m
            penalized_scores = penalized(S, I, L, m)
            vus_pr_metric = FFVUS(slope=5)
            vusi.append(vus_pr_metric.score(L, penalized_scores)['value'])
        return vusi

    VUSi  = np.array(vusi_curve(m_vals, S, I, L)).astype(np.float32)
    aVUSi = float(np.trapz(VUSi, m_vals))
    vus_pr = FFVUS(slope=5).score(L, S)['value']
    # PS_m0 = VUSi[0]
    # PS_m05 = VUSi[len(VUSi)//2]
    # PS_m1 = VUSi[-1]
    shown_m_vals = [0,m_middle_index,len(VUSi)-1]

    return {
        "t":             t,
        "L":             L,
        "S":             S,
        "I":             I,
        "PS_m0":         PS_m0,
        "PS_m05":        PS_m05,
        "PS_m1":         PS_m1,
        "m_vals":        m_vals,
        "shown_m_vals_index": shown_m_vals,
        "VUSi":          VUSi,
        "aVUSi":         aVUSi,
        "vus_pr":        vus_pr,
        "regions":       anomaly_regions,
        "interp_levels": interp_levels,
    }


# ═══════════════════════════════════════════════════════════════════════
# PART 2 — FIGURE PLOTTING
# ═══════════════════════════════════════════════════════════════════════

# ── Color palette ─────────────────────────────────────────────────────
# Mirrors the palette in plot_figures_for_the paper.py so the same signal
# type uses the same color across all paper figures:
#   "label"  = C_GT      = "blue"
#   "score"  = C_S       = "red"
#   "interp" = C_I       = "green"
#   "anomaly"= C_ANOMALY = "#E24B4A"
#   "gray"   = C_GRAY    = "#888780"
COLORS = {
    "label":   "blue",      # C_GT      — ground-truth label
    "score":   "red",       # C_S       — anomaly score
    "interp":  "green",     # C_I       — interpretability score (fully interpretable)
    "partial": "orange",    #           — partially interpretable annotation
    "ps":      "orange",    #           — penalized score
    "vusi":    "#d62728",   #           — VUSi curve
    "fill":    "#f5aaaa",   #           — VUSi fill
    "anomaly": "#E24B4A",   # C_ANOMALY — anomaly shading / totally missed annotation
    "gray":    "#888780",   # C_GRAY    — reference lines
}

# Per-region interpretability annotation styles: (label, text color, bg fill, edge color)
# Matches the annotation style used in plot_figures_for_the paper.py
INTERP_ANNOT_STYLES = [
    ("Fully\nInterpretable",     COLORS["interp"],  "#EEF7EE", COLORS["interp"]),
    ("Partially\nInterpretable", COLORS["partial"], "#FFF5EE", COLORS["partial"]),
    ("Totally\nMissed",          COLORS["anomaly"], "#fce8e4", COLORS["anomaly"]),
]

FONT_SIZE = 11
TICK_FONT_SIZE = FONT_SIZE - 3
LEGEND_FONT_SIZE = FONT_SIZE - 2

# Region labels and interpretability annotations
REGION_LABELS  = ["Anomaly 1\n(Fully Interpretable)",
                   "Anomaly 2\n(Partially Interpretable)",
                   "Anomaly 3\n(Totally Missed)"]
INTERP_ANNOTS  = ["$I$=1.0", "$I$=0.5", "$I$=0.0"]


def _shade_anomalies(ax: plt.Axes, regions: list[tuple], T: int) -> None:
    """Shade anomaly regions and draw boundary dashes on a time series axis."""
    for s, e in regions:
        ax.axvspan(s, e, color=COLORS["anomaly"], alpha=0.12, zorder=0)
        for x in [s, e]:
            ax.axvline(x, color=COLORS["anomaly"],
                       lw=0.6, ls="--", alpha=0.5, zorder=1)


def _style_time_axis(
    ax: plt.Axes,
    ylabel: str,
    color: str,
    T: int,
    last: bool = False,
) -> None:
    """Apply consistent styling to a time series panel."""
    ax.set_xlim(0, T)
    ax.set_ylim(-0.05, 1.18)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["0", "1"], fontsize=TICK_FONT_SIZE)
    ax.set_ylabel(ylabel, fontsize=FONT_SIZE, color=color,
                  rotation=0, labelpad=38, va="center")
    ax.tick_params(axis="x", bottom=last, labelbottom=last,
                   length=3 if last else 0, labelsize=TICK_FONT_SIZE)
    ax.tick_params(axis="y", length=2, pad=1, labelsize=TICK_FONT_SIZE)
    for sp in ax.spines.values():
        sp.set_visible(True)
        sp.set_linewidth(0.5)
        sp.set_color("#cccccc")
    # if last:
    #     ax.set_xlabel("Timestamp", fontsize=FONT_SIZE)
    ax.set_facecolor("none")


def plot_avusi_pipeline(
    data: dict,
    detector_name: str = "HBOS",
    figsize: tuple[float, float] = (6, 7.5),
    save_path: str = None,
) -> plt.Figure:
    """
    Draw the aVUSi computation pipeline illustration figure.

    Args:
        data:          Output of generate_avusi_demo_data().
        detector_name: Name shown on the anomaly score panel.
        figsize:       Figure size in inches (width, height).
        save_path:     If provided, saves PDF and PNG to this base path
                       (e.g. "avusi_figure" → saves avusi_figure.pdf
                       and avusi_figure.png). If None, does not save.

    Returns:
        matplotlib Figure object.
    """
    # Unpack data
    t         = data["t"]
    T         = len(t)
    L         = data["L"]
    S         = data["S"]
    I         = data["I"]
    PS_m0     = data["PS_m0"]
    PS_m05    = data["PS_m05"]
    PS_m1     = data["PS_m1"]
    m_vals    = data["m_vals"]
    VUSi      = data["VUSi"]
    aVUSi     = data["aVUSi"]
    vus_pr    = data["vus_pr"]
    regions   = data["regions"]

    # ── Figure and gridspec ───────────────────────────────────────────
    fig = plt.figure(figsize=figsize)
    fig.subplots_adjust(left=0.22, right=0.93, top=0.95, bottom=0.08)

    outer = gridspec.GridSpec(
        2, 1, figure=fig,
        height_ratios=[2.5, 2.5],
        hspace=0.3,
    )
    # Split top section: signals (L, S, I) with tight spacing, PS panels with more room
    top_split = gridspec.GridSpecFromSubplotSpec(
        2, 1, subplot_spec=outer[0], height_ratios=[0.8, 1], hspace=0.50,
    )
    inner_signals = gridspec.GridSpecFromSubplotSpec(
        3, 1, subplot_spec=top_split[0], hspace=0.10,
    )
    inner_ps = gridspec.GridSpecFromSubplotSpec(
        3, 1, subplot_spec=top_split[1], hspace=0.38,
    )

    ax_L    = fig.add_subplot(inner_signals[0])
    ax_S    = fig.add_subplot(inner_signals[1])
    ax_I    = fig.add_subplot(inner_signals[2])
    ax_PS0  = fig.add_subplot(inner_ps[0])
    ax_PS05 = fig.add_subplot(inner_ps[1])
    ax_PS1  = fig.add_subplot(inner_ps[2])
    ax_vusi = fig.add_subplot(outer[1])

    # ── Panel: Univariate Label ───────────────────────────────────────
    _shade_anomalies(ax_L, regions, T)
    ax_L.step(t, L, color=COLORS["label"], lw=1.4,
              where="post", zorder=3)
    _style_time_axis(ax_L, "", COLORS["label"], T)
    ax_L.text(0.98, 0.6, "Univariate Label $L$",
              transform=ax_L.transAxes, fontsize=LEGEND_FONT_SIZE,
              ha="right", color=COLORS["label"])
    # ax_L.set_title("aVUSi Computation Pipeline",
    #                fontsize=9, fontweight="bold", pad=14)

    for (s, e), lbl in zip(regions, REGION_LABELS):
        ax_L.text((s + e) / 2, 1.20, lbl,
                  ha="center", va="bottom", fontsize=FONT_SIZE - 3,
                  color='black',
                  # fontweight="bold"
                  )

    # ── Panel: Anomaly Score ──────────────────────────────────────────
    _shade_anomalies(ax_S, regions, T)
    ax_S.plot(t, S, color=COLORS["score"], lw=0.9, zorder=3)
    _style_time_axis(ax_S, "", COLORS["score"], T)
    ax_S.text(0.98, 0.6, "Anomaly Score $S$",
              transform=ax_S.transAxes, fontsize=LEGEND_FONT_SIZE,
              ha="right", color=COLORS["score"])

    # ── Panel: Interpretability Score ────────────────────────────────
    _shade_anomalies(ax_I, regions, T)
    ax_I.step(t, I, color=COLORS["interp"],
              lw=1.2, where="post", zorder=4)
    _style_time_axis(ax_I, "", COLORS["interp"], T, last=False)
    ax_I.text(0.98, 0.6, "Interpretability Score $I$",
              transform=ax_I.transAxes, fontsize=LEGEND_FONT_SIZE,
              ha="right", color=COLORS["interp"])

    # for (s, e), (text, color, fc, ec) in zip(regions, INTERP_ANNOT_STYLES):
    #     ax_I.text((s + e) / 2, 0.60, text,
    #               ha="center", va="center", fontsize=5,
    #               color=color, fontweight="bold",
    #               bbox=dict(fc=fc, ec=ec, lw=0.6, boxstyle="round,pad=0.25"))

    # ── Panels: Penalized Scores ──────────────────────────────────────
    ps_panels = [
        (ax_PS0,  PS_m0,  "m=0.0", False),
        (ax_PS05,  PS_m05, "m=0.5", False),
        (ax_PS1,  PS_m1,  "m=1.0", True),
    ]
    for ax, PS, m_lbl, last in ps_panels:
        _shade_anomalies(ax, regions, T)
        ax.plot(t, PS, color=COLORS["ps"], lw=0.9, zorder=3)
        # ax.fill_between(t, 0, PS, color=COLORS["ps"], alpha=0.18)
        _style_time_axis(ax, "", COLORS["ps"], T, last=last)
        ax.text(0.98, 0.6, f"PS$^{{{m_lbl}}}$",
                transform=ax.transAxes, fontsize=LEGEND_FONT_SIZE,
                ha="right", color=COLORS["ps"])

    # ── "⋮" between PS subplots ───────────────────────────────────────
    fig.canvas.draw()
    p0   = ax_PS0.get_position()
    p05  = ax_PS05.get_position()
    p1   = ax_PS1.get_position()
    for y_mid in [(p0.y0 + p05.y1) / 2, (p05.y0 + p1.y1) / 2]:
        fig.text(0.55, y_mid, "⋮", ha="center", va="center",
                 fontsize=FONT_SIZE, color=COLORS["gray"],
                 transform=fig.transFigure)

    # ── "Set of m" bracket ────────────────────────────────────────────
    # p_top = ax_PS0.get_position()
    # p_bot = ax_PS1.get_position()
    # bx    = p_top.x0 - 0.09
    # by0   = p_bot.y0 + 0.01
    # by1   = p_top.y1 - 0.01
    #
    # fig.lines.append(plt.Line2D(
    #     [bx, bx], [by0, by1],
    #     transform=fig.transFigure,
    #     color=COLORS["gray"], lw=1.0, clip_on=False,
    # ))
    # for by in [by0, by1]:
    #     fig.lines.append(plt.Line2D(
    #         [bx, bx + 0.015], [by, by],
    #         transform=fig.transFigure,
    #         color=COLORS["gray"], lw=1.0, clip_on=False,
    #     ))
    # fig.text(bx - 0.02, (by0 + by1) / 2, "Set of $m$",
    #          ha="center", va="center", fontsize=7,
    #          color=COLORS["gray"], rotation=90,
    #          transform=fig.transFigure)

    # ── Arrow: time series → VUSi ─────────────────────────────────────
    p_ts  = ax_PS1.get_position()
    p_vu  = ax_vusi.get_position()
    arr_x = p_ts.x0 + p_ts.width / 2

    # ax_ov = fig.add_axes([0, 0, 1, 1], zorder=10)
    # ax_ov.set_xlim(0, 1)
    # ax_ov.set_ylim(0, 1)
    # ax_ov.axis("off")
    #
    # arrow = FancyArrowPatch(
    #     (arr_x, p_ts.y0 - 0.002),
    #     (arr_x, p_vu.y1 + 0.002),
    #     arrowstyle="-|>", color=COLORS["gray"],
    #     lw=1.0, mutation_scale=9,
    #     transform=fig.transFigure, figure=fig,
    # )
    # ax_ov.add_patch(arrow)
    # fig.text(arr_x + 0.04, (p_ts.y0 + p_vu.y1) / 2,
    #          "Compute\nVUSi$^m$",
    #          ha="left", va="center", fontsize=6.5,
    #          color=COLORS["gray"], transform=fig.transFigure)

    # ── VUSi curve panel ──────────────────────────────────────────────
    ax_vusi.fill_between(m_vals, 0, VUSi,
                         color=COLORS["fill"], alpha=0.7, zorder=2)
    ax_vusi.plot(m_vals, VUSi,
                 color=COLORS["vusi"], lw=2.0, zorder=3,
                 label="VUSi")
    ax_vusi.axhline(vus_pr, color=COLORS["gray"],
                    lw=1.0, ls="-.", zorder=2,
                    label=f"VUS-PR = {vus_pr:.3f}")
    ax_vusi.annotate(
        f"VUS-PR={vus_pr:.3f}",
        xy=(1.0, vus_pr), xycoords=("axes fraction", "data"),
        xytext=(-4, 4), textcoords="offset points",
        ha="right", va="bottom", fontsize=FONT_SIZE, color=COLORS["gray"],
    )

    # aVUSi label
    ax_vusi.text(0.50, 0.25,
                 f"aVUSi = {aVUSi:.3f}\n(Area under VUSi curve)",
                 ha="center", va="center", fontsize=FONT_SIZE,
                 color=COLORS["vusi"], fontweight="bold",
                 transform=ax_vusi.transAxes)


    # Boundary case annotations
    middle_idx   = data["shown_m_vals_index"][1]
    m_val_mid    = m_vals[middle_idx]
    vusi_m0      = float(VUSi[0])
    vusi_m_middle = float(VUSi[middle_idx])
    vusi_m1      = float(VUSi[-1])

    ax_vusi.annotate(
        f"VUSi$^{{m=0}}$={vusi_m0:.2f}",
        xy=(0.0, vusi_m0), xytext=(0.18, vusi_m0 + 0.10),
        fontsize=FONT_SIZE, color=COLORS["vusi"],
        arrowprops=dict(arrowstyle="-|>", color=COLORS["vusi"],
                        lw=0.7, mutation_scale=6,
                        connectionstyle="arc3,rad=0.3"),
        ha="center",
    )
    ax_vusi.annotate(
        f"VUSi$^{{m=0.5}}$={vusi_m_middle:.2f}",
        xy=(m_val_mid, vusi_m_middle),
        xytext=(m_val_mid + 0.18, vusi_m_middle + 0.08),
        fontsize=FONT_SIZE, color=COLORS["vusi"],
        arrowprops=dict(arrowstyle="-|>", color=COLORS["vusi"],
                        lw=0.7, mutation_scale=6,
                        connectionstyle="arc3,rad=0.3"),
        ha="center",
    )
    ax_vusi.annotate(
        f"VUSi$^{{m=1}}$={vusi_m1:.2f}",
        xy=(1.0, vusi_m1), xytext=(0.82, vusi_m1 + 0.12),
        fontsize=FONT_SIZE, color=COLORS["vusi"],
        arrowprops=dict(arrowstyle="-|>", color=COLORS["vusi"],
                        lw=0.7, mutation_scale=6,
                        connectionstyle="arc3,rad=-0.3"),
        ha="center",
    )

    # Mark peak m*
    # m_star_idx = int(np.argmax(VUSi))
    # ax_vusi.axvline(m_vals[m_star_idx], color=COLORS["vusi"],
    #                 lw=0.8, ls=":", alpha=0.7)
    # ax_vusi.scatter([m_vals[m_star_idx]], [VUSi[m_star_idx]],
    #                 color=COLORS["vusi"], s=30, zorder=4)
    shown_indices = data["shown_m_vals_index"]
    for i, m_index in enumerate(shown_indices):
        m_val  = m_vals[m_index]
        v_val  = VUSi[m_index]
        ax_vusi.scatter([m_val], [v_val], color=COLORS["vusi"], s=30, zorder=4)
        if i == 1:  # m=0.5 point only
            ax_vusi.plot([m_val, m_val], [0, v_val],
                         color=COLORS["vusi"], lw=0.8, ls="--", alpha=0.6, zorder=2)
            ax_vusi.plot([0, m_val], [v_val, v_val],
                         color=COLORS["vusi"], lw=0.8, ls="--", alpha=0.6, zorder=2)
        else:
            ax_vusi.plot([m_val, m_val], [0, v_val],
                         color=COLORS["vusi"], lw=0.8, ls="--", alpha=0.6, zorder=2)

    # Axes styling
    ax_vusi.set_xlim(-0.02, 1.02)
    ax_vusi.set_ylim(0, 1.0)
    ax_vusi.set_xlabel("Interpretability sensitivity level $m$", fontsize=FONT_SIZE)
    ax_vusi.set_ylabel("VUSi", fontsize=FONT_SIZE)
    # ax_vusi.set_title(
    #     "Compute VUSi$^m$ $\\rightarrow$ aVUSi",
    #     fontsize=FONT_SIZE, fontweight="bold", pad=5,
    # )
    # for sp in ["top", "right"]:
    #     ax_vusi.spines[sp].set_visible(False)
    ax_vusi.spines["left"].set_linewidth(0.5)
    ax_vusi.spines["bottom"].set_linewidth(0.5)
    ax_vusi.spines["top"].set_linewidth(0.5)
    ax_vusi.spines["right"].set_linewidth(0.5)
    ax_vusi.tick_params(labelsize=TICK_FONT_SIZE)

    # Add ticks at VUSi(m=0.5) coordinates and colour them
    x_ticks = sorted(set([0.0, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0, round(float(m_val_mid), 4)]))
    y_ticks = sorted(set([0.0, 0.2, 0.4, 0.6, 0.8, 1.0, round(float(vusi_m_middle), 2)]))
    ax_vusi.set_xticks(x_ticks)
    ax_vusi.set_yticks(y_ticks)
    ax_vusi.tick_params(labelsize=TICK_FONT_SIZE)
    # for lbl, val in zip(ax_vusi.get_xticklabels(), ax_vusi.get_xticks()):
    #     if abs(val - m_val_mid) < 1e-4:
    #         lbl.set_color(COLORS["vusi"])
    #         lbl.set_fontweight("bold")
    # for lbl, val in zip(ax_vusi.get_yticklabels(), ax_vusi.get_yticks()):
    #     if abs(val - vusi_m_middle) < 1e-4:
    #         lbl.set_color(COLORS["vusi"])
    #         lbl.set_fontweight("bold")

    ax_vusi.legend(fontsize=FONT_SIZE, frameon=False,
                   loc="upper right", handlelength=1.5)

    # ── Bottom legend ─────────────────────────────────────────────────
    legend_elements = [
        mpatches.Patch(color=COLORS["label"],
                       label="Univariate label $L$"),
        mpatches.Patch(color=COLORS["score"],
                       label="Anomaly score $S$"),
        mpatches.Patch(color=COLORS["interp"],
                       label="Interpretability score $I$"),
        mpatches.Patch(color=COLORS["ps"],
                       label="Penalized score $PS^m$"),
        mpatches.Patch(color=COLORS["anomaly"], alpha=0.3,
                       label="Anomaly region"),
    ]
    # fig.legend(
    #     handles=legend_elements,
    #     loc="lower center", ncol=3,
    #     fontsize=FONT_SIZE, frameon=False,
    #     bbox_to_anchor=(0.55, -0.05),
    #     handlelength=1.2, columnspacing=0.8,
    # )

    # ── Save ──────────────────────────────────────────────────────────
    if save_path is not None:
        os.makedirs(os.path.basename(save_path), exist_ok=True)
        fig.savefig(f"{save_path}.pdf", bbox_inches="tight", dpi=300)
        fig.savefig(f"{save_path}.png", bbox_inches="tight", dpi=300)
        print(f"Saved: {save_path}.pdf and {save_path}.png")

    return fig


# ═══════════════════════════════════════════════════════════════════════
# PART 3 — MAIN
# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":

    # Generate demo data
    data = generate_avusi_demo_data(
        T=1000,
        anomaly_regions=[(120, 170), (380, 420), (640, 680)],
        # interp_levels=[1.0, 0.5, 0.0],
        seed=42,
    )

    print(f"aVUSi  = {data['aVUSi']:.3f}")
    print(f"VUS-PR = {data['vus_pr']:.3f}")
    print(f"Anomaly ratio = {data['L'].mean():.3f}")

    save_path = os.path.join(OUTPUT_DIR,'avusi_illustration')

    # Plot and save
    fig = plot_avusi_pipeline(
        data=data,
        detector_name="Synthetic",
        figsize=(7, 6),
        save_path=save_path,
    )