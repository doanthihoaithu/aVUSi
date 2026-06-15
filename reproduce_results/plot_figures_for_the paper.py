"""
Combined figure script: produces side-by-side figures for both datasets
(settings_six / synthetic and smd), one combined figure per figure type.
"""

import itertools
import os
import re

import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.ticker import FormatStrFormatter
from omegaconf import OmegaConf
from scipy.interpolate import make_interp_spline

# ── Output directory ──────────────────────────────────────────────────────────
CURRENT_DIR = os.path.dirname(__file__)
OUTPUT_DIR = os.path.join(CURRENT_DIR, 'combined_figures')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Config ────────────────────────────────────────────────────────────────────
config = OmegaConf.load('../conf/config.yaml')

DATASET_NAMES = ['settings_six', 'smd']
DATASET_DISPLAY_NAMES = {'settings_six': 'Synthetic', 'smd': 'SMD'}
RESULTS_PATHS = {
    'settings_six': '../results/settings_six/results.csv',
    'smd': '../results/smd/results.csv',
}
SERIES_INFO_PATHS = {
    'settings_six': '../results/settings_six/series_information.csv',
    'smd': '../results/smd/series_information.csv',
}

VUSI_CURVE_DATA_PATHS = {
    'settings_six': '../results/settings_six/new_metrics',
    'smd': '../results/smd/new_metrics',
}




# ── Common algorithm maps ─────────────────────────────────────────────────────
based_detector_map = {
    'hbos': 'HBOS',
    'denoising_auto_encoder': 'DAE',
    'cblof': 'CBLOF',
    'random_black_forest': 'RBF',
    'auto_encoder': 'AE',
    'tran_ad': 'TranAD',
    'omni_anomaly': 'OmniAno.',
    'mtad_gat': 'Mtad-GAT',
    'copod': 'COPOD',
    'encdec_ad': 'EncDec',
}
scenario_map = {
    'min_acc_min_interp': 'MinAcc-\nMinInt',
    'min_acc_max_interp': 'MinAcc-\nMaxInt',
    'max_acc_min_interp': 'MaxAcc-\nMinInt',
    'max_acc_max_interp': 'MaxAcc-\nMaxInt',
    'random_acc_random_interp': 'RandAcc-\nRandInt',
    'median_acc_median_interp': 'MedAcc-\nMedInt',
}
ensemble_map = {'avg_ens': 'AvgEns', 'decision_tree_128_average_4': 'MSv2'}
single_selection_map = {'oracle': 'Oracle', 'decision_tree_256_preds': 'MSv1'}
alg_rename_map = {**based_detector_map, **scenario_map, **ensemble_map, **single_selection_map}

# ── Metric name maps ──────────────────────────────────────────────────────────
independent_metric_name_map = {
    f'INDEPENDENT_INTERPRETABILITY_NDCG_HIT_K_{k}_W_{w}_SCORE_WITH_SMOOTHING': f'IndepNDCG@k={k},w={w}'
    for (k, w) in itertools.product(range(1, 39), range(0, 31))
}
conditional_metric_name_map = {
    f'INTERPRETABILITY_CONDITIONAL_VOLUMN_PR_WITH_NDCG_HIT_K_{k}_W_{w}_M_{m}_SCORE_COMBINATION_WITH_SMOOTHING': f'aVUSi@k={k},w={w},M={m}'
    for (k, w, m) in itertools.product(range(1, 39), range(0, 31), [10,20,30,40,50])
}
metric_name_map = {
    'FFVUS_PR': 'VUS-PR',
    'sum_acc_interp': 'AddC',
    'mul_acc_interp': 'HarC',
    **independent_metric_name_map,
    **conditional_metric_name_map,
}

# ── Color / style maps ────────────────────────────────────────────────────────
color_map_dict = {
    'based_detector': '#CCCCCC',
    'scenario': '#C3B1E1',
    'ensemble': 'orange',
    'avg_ens': '#FF7133',
    'oracle': 'white',
    'decision_tree_128_average_4': 'green',
    'decision_tree_256_preds': 'blue',
}
label_map_dict = {
    'based_detector': 'black',
    'scenario': '#7247B8',
    'ensemble': 'orange',
    'avg_ens': 'orange',
    'oracle': 'white',
    'decision_tree_128_average_4': 'green',
    'decision_tree_256_preds': 'blue',
}
my_palette = {
    **{alg: color_map_dict['based_detector'] for alg in based_detector_map},
    **{alg: color_map_dict['scenario'] for alg in scenario_map},
    **{alg: color_map_dict['ensemble'] for alg in ensemble_map},
    'oracle': color_map_dict['oracle'],
    'decision_tree_128_average_4': color_map_dict['decision_tree_128_average_4'],
    'decision_tree_256_preds': color_map_dict['decision_tree_256_preds'],
    'avg_ens': color_map_dict['avg_ens'],
}
my_label_color = {
    **{alg: label_map_dict['based_detector'] for alg in based_detector_map},
    **{alg: label_map_dict['scenario'] for alg in scenario_map},
    **{alg: label_map_dict['ensemble'] for alg in ensemble_map},
    'oracle': color_map_dict['oracle'],
    'decision_tree_128_average_4': color_map_dict['decision_tree_128_average_4'],
    'decision_tree_256_preds': color_map_dict['decision_tree_256_preds'],
    'avg_ens': color_map_dict['avg_ens'],
}

random_markers = ["o", "s", "^", "D", "p", "h", "v", "<", ">", "*", "X"]
random_colors = [
    "#1f77b4", "#2ca02c", "#d62728", "#17becf", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#b33c00", "#ff7f0e",
]
FONT_SIZE = 14


# ── Helper: parse parameter values from metric names ──────────────────────────
def find_k_in_name(metric_name, max_value=100):
    match = re.search(r'K_(\d+)', metric_name)
    if match:
        v = int(match.group(1))
        return v if v <= max_value else -1
    return -1


def find_w_in_name(metric_name, max_value=100):
    match = re.search(r'W_(\d+)', metric_name)
    if match:
        v = int(match.group(1))
        return v if v <= max_value else -1
    return -1


def find_m_in_name(metric_name, max_value=100):
    match = re.search(r'M_(\d+)', metric_name)
    if match:
        v = int(match.group(1))
        return v if v <= max_value else -1
    return -1


def get_columns_by_parameter_sensitivity_config(df, cfg, investigated_parameter):
    assert investigated_parameter in ['k', 'w', 'm']
    max_k = cfg.max_k
    max_w = cfg.max_smoothing_window
    max_m = cfg.max_n_interpretability_sensitivity_levels

    if investigated_parameter == 'k':
        fix_w = cfg['default_selected_smoothing_window']
        fix_m = cfg['default_selected_n_interpretability_sensitivity_levels']
        indep = {find_k_in_name(f, max_k): f for f in df.columns
                 if f.startswith('INDEPENDENT_INTERPRETABILITY_NDCG_HIT')
                 and f'W_{fix_w}' in f and f.endswith('WITH_SMOOTHING')}
        indep.pop(-1, None)
        cond = {find_k_in_name(f, max_k): f for f in df.columns
                if f.startswith('INTERPRETABILITY_CONDITIONAL_VOLUMN_PR_WITH_NDCG_HIT')
                and f'W_{fix_w}' in f and f'M_{fix_m}' in f and f.endswith('WITH_SMOOTHING')}
        cond.pop(-1, None)

    elif investigated_parameter == 'w':
        fix_k = cfg['default_selected_k']
        fix_m = cfg['default_selected_n_interpretability_sensitivity_levels']
        indep = {find_w_in_name(f, max_w): f for f in df.columns
                 if f.startswith('INDEPENDENT_INTERPRETABILITY_NDCG_HIT')
                 and f'K_{fix_k}' in f and f.endswith('WITH_SMOOTHING')}
        indep.pop(-1, None)
        cond = {find_w_in_name(f, max_w): f for f in df.columns
                if f.startswith('INTERPRETABILITY_CONDITIONAL_VOLUMN_PR_WITH_NDCG_HIT')
                and f'K_{fix_k}' in f and f'M_{fix_m}' in f and f.endswith('WITH_SMOOTHING')}
        cond.pop(-1, None)

    else:  # 'm'
        fix_k = cfg['default_selected_k']
        fix_w = cfg['default_selected_smoothing_window']
        indep = {find_m_in_name(f, max_m): f for f in df.columns
                 if f.startswith('INDEPENDENT_INTERPRETABILITY_NDCG_HIT')
                 and f'K_{fix_k}' in f and f'W_{fix_w}' in f and f.endswith('WITH_SMOOTHING')}
        indep.pop(-1, None)
        cond = {find_m_in_name(f, max_m): f for f in df.columns
                if f.startswith('INTERPRETABILITY_CONDITIONAL_VOLUMN_PR_WITH_NDCG_HIT')
                and f'K_{fix_k}' in f and f'W_{fix_w}' in f and f.endswith('WITH_SMOOTHING')}
        cond.pop(-1, None)

    return (
        list(dict(sorted(indep.items())).values()),
        list(dict(sorted(cond.items())).values()),
    )


# ── Data loading ──────────────────────────────────────────────────────────────
def load_dataset(dataset_name):
    cfg = config['parameter_sensitivity_analysis']['datasets'][dataset_name]
    df_full = pd.read_csv(RESULTS_PATHS[dataset_name])

    selected_k = cfg.default_selected_k
    selected_w = cfg.default_selected_smoothing_window
    selected_m = cfg.default_selected_n_interpretability_sensitivity_levels

    visualized_metrics = [
        'FFVUS_PR',
        f'INDEPENDENT_INTERPRETABILITY_NDCG_HIT_K_{selected_k}_W_{selected_w}_SCORE_WITH_SMOOTHING',
        f'INTERPRETABILITY_CONDITIONAL_VOLUMN_PR_WITH_NDCG_HIT_K_{selected_k}_W_{selected_w}_M_{selected_m}_SCORE_COMBINATION_WITH_SMOOTHING',
    ]

    visualized_algs = list(based_detector_map.keys()) + ['avg_ens']
    df_filtered = df_full[df_full['algorithm'].isin(visualized_algs)].copy()

    return df_full, df_filtered, visualized_metrics, cfg


# ── Shared boxplot label styling ──────────────────────────────────────────────
def _style_boxplot_yticklabels(ax):
    for label in ax.get_yticklabels():
        label.set_fontsize(17)
        if label.get_text() != 'oracle':
            label.set_color(my_label_color.get(label.get_text(), 'black'))
        if label.get_text() in scenario_map or label.get_text() in ensemble_map:
            label.set_weight('bold')
        if label.get_text() in scenario_map:
            label.set_fontsize(15)
    new_labels = [alg_rename_map.get(t.get_text(), t.get_text()) for t in ax.get_yticklabels()]
    ax.set_yticklabels(new_labels)


def _style_boxplot_xticklabels(ax):
    for label in ax.get_xticklabels():
        label.set_fontsize(12)
        if label.get_text() != 'oracle':
            label.set_color(my_label_color.get(label.get_text(), 'black'))
        if label.get_text() in scenario_map or label.get_text() in ensemble_map:
            label.set_weight('bold')
            if label.get_text() in scenario_map:
                label.set_fontsize(12)
    new_labels = [alg_rename_map.get(t.get_text(), t.get_text()) for t in ax.get_xticklabels()]
    ax.set_xticklabels(new_labels, rotation=60)


def _add_oracle_and_merge(df_filtered, metric):
    oracle_df = (
        df_filtered[df_filtered['algorithm'].isin(based_detector_map.keys())]
        [['test_batch_id', metric]]
        .groupby('test_batch_id').max()
    )
    oracle_df['algorithm'] = 'oracle'
    return pd.concat([df_filtered, oracle_df], ignore_index=True)


_BOXPLOT_MEAN_PROPS = {
    "marker": "o",
    "markerfacecolor": "white",
    "markeredgecolor": "black",
    "markersize": "4",
}


# ── Figure 1: Combined three-metrics boxplot (vertical, 2 rows × 3 cols) ─────
def plot_combined_three_metrics_boxplot(datasets_data, save_path):
    n_datasets = len(datasets_data)
    fig, axes = plt.subplots(
        nrows=n_datasets, ncols=3,
        figsize=(20, 12 * n_datasets),
        squeeze=False,
    )

    for row_idx, (dataset_name, (df_full, df_filtered, visualized_metrics, cfg)) in enumerate(datasets_data.items()):
        for metric_index, metric in enumerate(visualized_metrics):
            ax = axes[row_idx][metric_index]
            merged_df = _add_oracle_and_merge(df_filtered, metric)
            alg_order = merged_df.groupby('algorithm')[metric].median().sort_values(ascending=False).index

            sns.boxplot(
                x=metric, y='algorithm', data=merged_df, ax=ax,
                order=alg_order, dodge=False, showfliers=False,
                palette=my_palette, showmeans=True, meanprops=_BOXPLOT_MEAN_PROPS,
            )
            _style_boxplot_yticklabels(ax)
            ax.set_xlabel(metric_name_map.get(metric, metric), fontsize=20)
            ax.set_ylabel('Detector/Ensembles', fontsize=20)

        for ax in axes[row_idx][1:]:
            ax.set_ylabel('')

        axes[row_idx][0].set_title(
            DATASET_DISPLAY_NAMES[dataset_name], fontsize=22, fontweight='bold', loc='left'
        )

    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches='tight', format=save_path.split('.')[-1])
    print(f'Saved: {save_path}')
    # plt.close(fig)
    return fig


# ── Figure 2: Combined three-metrics boxplot (horizontal, 3 rows × 2 cols) ───
def plot_combined_three_metrics_boxplot_horizontal(datasets_data, save_path):
    n_metrics = 3
    n_datasets = len(datasets_data)
    fig, axes = plt.subplots(
        nrows=n_metrics, ncols=n_datasets,
        figsize=(4 * n_datasets, 8),
        squeeze=False,
        sharey='row',
    )

    for col_idx, (dataset_name, (df_full, df_filtered, visualized_metrics, cfg)) in enumerate(datasets_data.items()):
        for metric_index, metric in enumerate(visualized_metrics):
            ax = axes[metric_index][col_idx]
            merged_df = _add_oracle_and_merge(df_filtered, metric)
            alg_order = (
                merged_df.groupby('algorithm')[metric].median()
                .sort_values(ascending=False).index.tolist()
            )

            sns.boxplot(
                ax=ax, data=merged_df, y=metric, x='algorithm',
                hue='algorithm', order=alg_order, dodge=False, showfliers=False,
                palette=my_palette, showmeans=True, meanprops=_BOXPLOT_MEAN_PROPS,
            )
            _style_boxplot_xticklabels(ax)
            short_label = _format_metric_label(metric_name_map.get(metric, metric))
            if col_idx == 0:
                ax.set_ylabel(short_label, fontsize=16)
            else:
                ax.set_ylabel('')
            ax.set_xlabel('')

        axes[0][col_idx].set_title(
            DATASET_DISPLAY_NAMES[dataset_name], fontsize=16, fontweight='bold'
        )

    fig.supxlabel('Detector', fontsize=16)
    fig.tight_layout(w_pad=0.05)
    fig.savefig(save_path, dpi=300, bbox_inches='tight', format=save_path.split('.')[-1])
    print(f'Saved: {save_path}')
    # plt.close(fig)
    return fig


# ── Figure 3: Combined average ranking (1 row × 2 cols) ──────────────────────
def _reconstruct_metric_matrix(df_filtered, visualized_metrics, cfg):
    selected_k = cfg.default_selected_k
    selected_w = cfg.default_selected_smoothing_window
    accuracy_metric = 'FFVUS_PR'
    interp_metric = f'INDEPENDENT_INTERPRETABILITY_NDCG_HIT_K_{selected_k}_W_{selected_w}_SCORE_WITH_SMOOTHING'

    all_algs = list(based_detector_map.keys()) + ['avg_ens']
    only_base = df_filtered[df_filtered['algorithm'].isin(all_algs)][
        ['test_batch_id', 'algorithm'] + visualized_metrics
    ]

    metric_df = pd.DataFrame({'test_batch_id': only_base['test_batch_id'].unique()})
    for test_batch_id in metric_df['test_batch_id'].unique():
        for sel_metric in visualized_metrics:
            for alg in all_algs:
                vals = only_base.loc[
                    (only_base['test_batch_id'] == test_batch_id) & (only_base['algorithm'] == alg),
                    sel_metric,
                ].values
                if len(vals) > 0:
                    metric_df.loc[metric_df['test_batch_id'] == test_batch_id, f'{sel_metric}/{alg}'] = vals[0]

    metric_df.set_index('test_batch_id', inplace=True)

    for alg in all_algs:
        acc = metric_df.get(f'{accuracy_metric}/{alg}')
        interp = metric_df.get(f'{interp_metric}/{alg}')
        if acc is not None and interp is not None:
            denom = (acc + interp).replace(0, np.nan)
            metric_df[f'sum_acc_interp/{alg}'] = (acc + interp) / 2
            metric_df[f'mul_acc_interp/{alg}'] = 2 * acc * interp / denom

    return metric_df


def _compute_average_ranking(ranking_dfs_dict, metrics):
    avg_df = pd.DataFrame(index=list(based_detector_map.keys()) + ['avg_ens'])
    for metric in metrics:
        avg_df.loc[:, metric] = ranking_dfs_dict[metric].mean(axis=0).values.reshape(-1)
    return avg_df


def _format_metric_label(label):
    if '@' in label:
        name, _ = label.split('@', 1)
        return name
    return label


def _draw_ranking_to_ax(ax, ranking_data, metrics_ordered, bold_detectors=()):
    model_styles = {
        det: dict(color=random_colors[idx], marker=random_markers[idx], label=alg_rename_map.get(det, det))
        for idx, det in enumerate(list(based_detector_map.keys()) + ['avg_ens'])
    }
    y_pos = {m: i for i, m in enumerate(metrics_ordered)}

    for model, style in model_styles.items():
        is_bold = model in bold_detectors
        lw = 2.5 if is_bold else 0.8
        alpha = 1.0 if is_bold else 0.5
        ms = 150 if is_bold else 80
        zorder_line = 4 if is_bold else 2

        xs, ys = [], []
        for metric in metrics_ordered:
            rank = ranking_data[model].get(metric)
            if rank is not None:
                xs.append(rank)
                ys.append(y_pos[metric])
        if len(xs) >= 4:
            y_arr = np.array(ys, dtype=float)
            x_arr = np.array(xs, dtype=float)
            spl = make_interp_spline(y_arr, x_arr, k=3)
            y_fine = np.linspace(y_arr.min(), y_arr.max(), 300)
            ax.plot(spl(y_fine), y_fine, color=style['color'],
                    linewidth=lw, alpha=alpha, zorder=zorder_line)
        elif len(xs) >= 2:
            ax.plot(xs, ys, color=style['color'],
                    linewidth=lw, alpha=alpha, zorder=zorder_line)
        ax.scatter(xs, ys, color=style['color'], marker=style['marker'],
                   s=ms, zorder=zorder_line + 1, label=style['label'])

    ax.set_yticks(range(len(metrics_ordered)))
    ax.set_yticklabels(
        [_format_metric_label(metric_name_map.get(m, m)) for m in metrics_ordered],
        fontsize=15, linespacing=1.2,
    )
    ax.set_xlabel('Average Ranking', fontsize=15)
    ax.set_ylabel('Evaluation Metrics', fontsize=15)
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, linestyle='--', color='gray', alpha=0.5)
    ax.xaxis.grid(True, linestyle='--', color='gray', alpha=0.5)
    ax.spines['top'].set_visible(True)
    ax.spines['right'].set_visible(True)
    for label in ax.get_yticklabels():
        if label.get_text().startswith('aVUSi'):
            label.set_weight('bold')


def plot_combined_average_ranking(datasets_data, save_path):
    n_datasets = len(datasets_data)
    fig, axes = plt.subplots(
        nrows=1, ncols=n_datasets, figsize=(4.5 * n_datasets, 4.5), squeeze=False,
        sharey=True,
    )

    for col_idx, (dataset_name, (df_full, df_filtered, visualized_metrics, cfg)) in enumerate(datasets_data.items()):
        cache_dir = f'{dataset_name}_data'
        cache_path = os.path.join(cache_dir, 'metric_matrix.csv')
        if os.path.exists(cache_path):
            metric_df = pd.read_csv(cache_path, index_col=0)
            print(f'Loaded cached metric_matrix: {cache_path}')
        else:
            metric_df = _reconstruct_metric_matrix(df_filtered, visualized_metrics, cfg)
            os.makedirs(cache_dir, exist_ok=True)
            metric_df.to_csv(cache_path)
            print(f'Saved metric_matrix: {cache_path}')

        metrics_with_trivial = visualized_metrics + ['sum_acc_interp', 'mul_acc_interp']
        main_metrics = [m for m in metrics_with_trivial if 'CONDITIONAL_VOLUMN_PR_WITH_NDCG' in m]
        for m in main_metrics:
            metrics_with_trivial.remove(m)
        metrics_with_trivial.extend(main_metrics)

        ranking_dfs_dict = {}
        for metric in metrics_with_trivial:
            cols = metric_df.columns[metric_df.columns.str.startswith(metric + '/')]
            ranking_df = metric_df[cols].rank(axis=1, method='average', ascending=False)
            ranking_df.columns = [c.split('/')[-1] for c in ranking_df.columns]
            ranking_dfs_dict[metric] = ranking_df

        avg_ranking_df = _compute_average_ranking(ranking_dfs_dict, metrics_with_trivial)
        ranking_data = avg_ranking_df.to_dict(orient='index')

        bold_detectors_map = {
            'settings_six': ('tran_ad', 'avg_ens'),
            'smd': ('copod', 'avg_ens'),
        }
        bold_dets = bold_detectors_map.get(dataset_name, ())

        ax = axes[0][col_idx]
        _draw_ranking_to_ax(ax, ranking_data, metrics_with_trivial, bold_detectors=bold_dets)
        ax.set_xlabel('')

        x_margin, y_margin = 0.4, 0.35
        n_metrics = len(metrics_with_trivial)
        all_x = [
            v for model in ranking_data
            for metric in metrics_with_trivial
            for v in [ranking_data[model].get(metric)]
            if v is not None
        ]
        ax.set_xlim(min(all_x) - x_margin, max(all_x) + x_margin)
        ax.set_ylim(-y_margin, n_metrics - 1 + y_margin)

        ax.set_title(DATASET_DISPLAY_NAMES[dataset_name], fontsize=FONT_SIZE + 2, fontweight='bold')

        if col_idx > 0:
            plt.setp(ax.get_yticklabels(), visible=False)
            ax.set_ylabel('')

    handles = [
        mlines.Line2D([], [], color=random_colors[idx], marker=random_markers[idx],
                      linestyle='None', markersize=9, label=alg_rename_map.get(det, det))
        for idx, det in enumerate(list(based_detector_map.keys()) + ['avg_ens'])
    ]
    plt.tight_layout(rect=[0, 0, 1, 1])
    plt.subplots_adjust(wspace=0.04)
    ax_y0 = axes[0][0].get_position().y0
    xlabel_y = ax_y0 - 0.06
    legend_y = ax_y0 - 0.1
    fig.text(0.5, xlabel_y, 'Average Ranking', ha='center', va='top', fontsize=15)
    fig.legend(handles=handles, loc='upper right', bbox_to_anchor=(1.0, legend_y),
               ncol=len(handles)//2+1, frameon=True, fontsize=13, columnspacing=0.8)
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f'Saved: {save_path}')
    # plt.close(fig)
    return fig

def _draw_cd_panel(ax, avg_ranks, cd, all_algs):
    """Draw one Demsar-style CD panel onto ax.

    avg_ranks : pd.Series  alg -> average rank (lower = better)
    cd        : float       critical difference value
    all_algs  : list        canonical algorithm order for display names
    """
    NEMENYI_Q = {
        2: 1.960, 3: 2.344, 4: 2.569, 5: 2.728, 6: 2.850,
        7: 2.949, 8: 3.031, 9: 3.102, 10: 3.164, 11: 3.219, 12: 3.268,
    }

    order = np.argsort(avg_ranks.values)
    sorted_algs    = [avg_ranks.index[i] for i in order]
    sorted_display = [alg_rename_map.get(a, a) for a in sorted_algs]
    sorted_ranks   = [float(avg_ranks.iloc[i]) for i in order]

    k = len(sorted_ranks)
    left_count = (k + 1) // 2
    row_h      = 0.5
    axis_y     = 0.0
    rank_lo, rank_hi = sorted_ranks[0], sorted_ranks[-1]
    label_x_left  = rank_lo - 1.8
    label_x_right = rank_hi + 1.8

    ax.axis('off')
    ax.set_xlim(label_x_left - 0.3, label_x_right + 0.3)
    ax.set_ylim(-max(left_count, k - left_count) * row_h - 1.0, 1.8)

    # Rank axis
    ax.plot([rank_lo - 0.3, rank_hi + 0.3], [axis_y, axis_y], 'k-', lw=2, zorder=5)
    for r in sorted_ranks:
        ax.plot([r, r], [axis_y - 0.12, axis_y + 0.12], 'k-', lw=1.5, zorder=5)

    # CD bracket
    cd_y, cd_x0 = 1.3, rank_lo - 0.2
    ax.plot([cd_x0, cd_x0 + cd], [cd_y, cd_y], 'k-', lw=2)
    for cx in (cd_x0, cd_x0 + cd):
        ax.plot([cx, cx], [cd_y - 0.1, cd_y + 0.1], 'k-', lw=2)
    ax.text(cd_x0 + cd / 2, cd_y + 0.15, f'CD = {cd:.2f}',
            ha='center', va='bottom', fontsize=10, fontweight='bold')

    # Left-half connectors (best-ranked)
    for pos, alg_i in enumerate(range(left_count)):
        rank, name = sorted_ranks[alg_i], sorted_display[alg_i]
        y = -(pos + 1) * row_h
        ax.plot([rank, rank],         [axis_y, y], color='#444', lw=1, zorder=3)
        ax.plot([rank, label_x_left], [y, y],      color='#444', lw=1, zorder=3)
        ax.text(label_x_left - 0.1, y, name, ha='right', va='center', fontsize=9)

    # Right-half connectors (worst-ranked)
    for pos, alg_i in enumerate(range(left_count, k)):
        rank, name = sorted_ranks[alg_i], sorted_display[alg_i]
        y = -(pos + 1) * row_h
        ax.plot([rank, rank],          [axis_y, y], color='#444', lw=1, zorder=3)
        ax.plot([rank, label_x_right], [y, y],      color='#444', lw=1, zorder=3)
        ax.text(label_x_right + 0.1, y, name, ha='left', va='center', fontsize=9)

    # Clique bars (maximal non-significant groups)
    raw_cliques = []
    for start in range(k):
        end = start
        while end + 1 < k and sorted_ranks[end + 1] - sorted_ranks[start] < cd:
            end += 1
        if end > start:
            raw_cliques.append((start, end))
    maximal = []
    for c in sorted(raw_cliques, key=lambda x: x[1] - x[0], reverse=True):
        if not any(m[0] <= c[0] and m[1] >= c[1] for m in maximal):
            maximal.append(c)
    bar_y0 = axis_y - 0.28
    for bar_i, (s, e) in enumerate(sorted(maximal, key=lambda x: x[0])):
        bar_y = bar_y0 - bar_i * 0.18
        ax.plot([sorted_ranks[s] - 0.05, sorted_ranks[e] + 0.05],
                [bar_y, bar_y], 'k-', lw=4, solid_capstyle='butt', zorder=4)


def plot_critical_difference(datasets_data, save_path):
    """Critical Difference diagrams (Demsar 2006, Nemenyi post-hoc, α=0.05).

    Layout: n_metrics rows × n_datasets cols.
    Each panel shows the CD diagram for one (metric, dataset) combination.
    Metrics: VUS-PR, IndepNDCG, aVUSi  (the three in visualized_metrics).
    N = number of test_batch_ids in metric_df (one rank observation per problem).
    """
    NEMENYI_Q = {
        2: 1.960, 3: 2.344, 4: 2.569, 5: 2.728, 6: 2.850,
        7: 2.949, 8: 3.031, 9: 3.102, 10: 3.164, 11: 3.219, 12: 3.268,
    }
    all_algs = list(based_detector_map.keys()) + ['avg_ens']

    # Use visualized_metrics from the first dataset as the canonical metric list
    first_entry = next(iter(datasets_data.values()))
    visualized_metrics_ref = first_entry[2]  # (df_full, df_filtered, visualized_metrics, cfg)

    n_metrics  = len(visualized_metrics_ref)
    n_datasets = len(datasets_data)
    fig, axes  = plt.subplots(n_metrics, n_datasets,
                              figsize=(4 * n_datasets, 3 * n_metrics),
                              squeeze=False)

    for col_idx, (dataset_name, (df_full, df_filtered, visualized_metrics, cfg)) in enumerate(datasets_data.items()):
        # Load or compute metric matrix (reuse cache)
        cache_dir  = f'{dataset_name}_data'
        cache_path = os.path.join(cache_dir, 'metric_matrix.csv')
        if os.path.exists(cache_path):
            metric_df = pd.read_csv(cache_path, index_col=0)
        else:
            metric_df = _reconstruct_metric_matrix(df_filtered, visualized_metrics, cfg)
            os.makedirs(cache_dir, exist_ok=True)
            metric_df.to_csv(cache_path)

        # Column title (dataset name) on first row
        axes[0][col_idx].set_title(DATASET_DISPLAY_NAMES[dataset_name],
                                   fontsize=FONT_SIZE + 2, fontweight='bold')

        for row_idx, metric in enumerate(visualized_metrics):
            ax = axes[row_idx][col_idx]

            cols = metric_df.columns[metric_df.columns.str.startswith(metric + '/')]
            if len(cols) == 0:
                ax.axis('off')
                continue

            ranking_df = metric_df[cols].rank(axis=1, method='average', ascending=False)
            ranking_df.columns = [c.split('/')[-1] for c in ranking_df.columns]
            algs_present = [a for a in all_algs if a in ranking_df.columns]
            ranking_df   = ranking_df[algs_present]

            N      = len(ranking_df)
            k      = len(algs_present)
            avg_ranks = ranking_df.mean(axis=0)

            q_alpha = NEMENYI_Q.get(k, 3.268)
            cd      = q_alpha * np.sqrt(k * (k + 1) / (6 * N))

            _draw_cd_panel(ax, avg_ranks, cd, all_algs)

            # Row label: metric short name on left column only
            if col_idx == 0:
                short = _format_metric_label(metric_name_map.get(metric, metric))
                ax.annotate(short, xy=(-0.2, 0.5), xycoords='axes fraction',
                            ha='right', va='center', fontsize=FONT_SIZE,
                            fontweight='bold', rotation=90)

    plt.tight_layout(h_pad=0.05)
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f'Saved: {save_path}')
    # plt.close(fig)
    return fig

# ── Figure 4: Combined hyper-parameter sensitivity (2 rows × N cols) ─────────
def _build_sensitivity_panels(hyper_parameters):
    panels = []
    for param in hyper_parameters:
        if param != 'm':
            panels.append((param, False))   # Independent NDCG
        panels.append((param, True))        # aVUSi
    return panels


def plot_combined_hyper_parameter_sensitivity(datasets_data, hyper_parameters, save_path):
    panels = _build_sensitivity_panels(hyper_parameters)
    ncols = len(panels)
    n_datasets = len(datasets_data)

    fig, axes = plt.subplots(
        nrows=n_datasets, ncols=ncols,
        figsize=(3 * ncols, 2.5 * n_datasets),
        squeeze=False,
    )

    for row_idx, (dataset_name, (df_full, df_filtered, visualized_metrics, cfg)) in enumerate(datasets_data.items()):
        max_k = cfg.max_k
        max_w = cfg.max_smoothing_window
        max_m = cfg.max_n_interpretability_sensitivity_levels
        fix_k = cfg.default_selected_k
        fix_w = cfg.default_selected_smoothing_window
        fix_m = cfg.default_selected_n_interpretability_sensitivity_levels

        for ax_idx, (investigated_parameter, is_avusi) in enumerate(panels):
            ax = axes[row_idx][ax_idx]
            indep_metrics, cond_metrics = get_columns_by_parameter_sensitivity_config(
                df_full, cfg, investigated_parameter
            )
            metrics = cond_metrics if is_avusi else indep_metrics

            if investigated_parameter == 'k':
                x_vals = [find_k_in_name(m, max_k) for m in metrics]
            elif investigated_parameter == 'w':
                x_vals = [find_w_in_name(m, max_w) for m in metrics]
            else:
                x_vals = [find_m_in_name(m, max_m) for m in metrics]

            if investigated_parameter == 'k' and dataset_name == 'smd':
                pairs = [(x, m) for x, m in zip(x_vals, metrics) if x != 10]
                x_vals, metrics = (list(z) for z in zip(*pairs)) if pairs else ([], [])

            for alg_idx, alg in enumerate(list(based_detector_map.keys()) + ['avg_ens']):
                alg_mean = df_full[df_full['algorithm'] == alg][metrics].mean()
                ax.plot(
                    x_vals, alg_mean,
                    label=alg_rename_map.get(alg, alg),
                    marker=random_markers[alg_idx],
                    color=random_colors[alg_idx],
                    linewidth=1.8, markersize=3.5,
                )

            metric_label = 'aVUSi' if is_avusi else 'IndepNDCG'
            xlabel = investigated_parameter.upper() if investigated_parameter == 'm' else investigated_parameter

            ax.set_title(metric_label if row_idx == 0 else '', fontsize=FONT_SIZE-2, pad=8, fontweight='bold')
            ax.set_xlabel(xlabel, fontsize=FONT_SIZE, fontweight='bold')
            ax.xaxis.set_major_formatter(FormatStrFormatter('%d'))
            ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
            ax.tick_params(axis='both', labelsize=10)
            ax.grid(axis='y', linestyle='--', alpha=0.6)
            ax.grid(axis='x', linestyle='--', alpha=0.6)

        axes[row_idx][0].annotate(
            f'{DATASET_DISPLAY_NAMES[dataset_name]}\nFixed:k={fix_k}, w={fix_w}, M={fix_m}',
            xy=(-0.3, 0.5), xycoords='axes fraction',
            fontsize=FONT_SIZE-2, fontweight='bold', ha='center', va='center', rotation=90,
        )

    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(
        handles, labels,
        loc='center left', bbox_to_anchor=(0.81, 0.5),
        ncol=1,
        frameon=True, fontsize=FONT_SIZE - 3,
    )
    plt.tight_layout(rect=[0, 0, 0.82, 1])

    # Vertical separator lines between parameter groups
    separator_cols = []
    prev_param = None
    for ax_idx, (param, _) in enumerate(panels):
        if param != prev_param and prev_param is not None:
            separator_cols.append(ax_idx)
        prev_param = param
    y_bottom = axes[-1][0].get_position().y0
    y_top    = axes[0][0].get_position().y1
    for sep_idx in separator_cols:
        # Place just past the right spine of the previous group so the line
        # stays clear of the y-tick labels on the next group's first column.
        x_line = axes[0][sep_idx - 1].get_position().x1 + 0.007
        fig.add_artist(
            plt.Line2D([x_line, x_line], [y_bottom, y_top],
                       transform=fig.transFigure,
                       color='gray', linewidth=1.5, linestyle='--', zorder=10)
        )

    # Horizontal separator lines between dataset rows
    # Placed just above the top spine of the lower row to stay clear of
    # the x-tick labels / xlabel that appear below the upper row's axes.
    x_left  = axes[0][0].get_position().x0
    x_right = axes[0][-1].get_position().x1
    for row in range(n_datasets - 1):
        y_line = axes[row + 1][0].get_position().y1 + 0.03
        fig.add_artist(
            plt.Line2D([x_left, x_right], [y_line, y_line],
                       transform=fig.transFigure,
                       color='gray', linewidth=1.5, linestyle='--', zorder=10)
        )

    ext = save_path.rsplit('.', 1)[-1] if '.' in save_path else 'pdf'
    fig.savefig(save_path, dpi=300, bbox_inches='tight', format=ext)
    print(f'Saved: {save_path}')
    # plt.close(fig)
    return fig


# ── Figure 5: Computation time scalability analysis ──────────────────────────
def _get_time_cols_by_param(df, cfg, investigated_parameter):
    """Return sorted (indep_xs, indep_cols, cond_xs, cond_cols) for the given parameter."""
    max_k = cfg.max_k
    max_w = cfg.max_smoothing_window
    max_m = cfg.max_n_interpretability_sensitivity_levels
    fix_k = cfg.default_selected_k
    fix_w = cfg.default_selected_smoothing_window
    fix_m = cfg.default_selected_n_interpretability_sensitivity_levels

    all_time_cols = [c for c in df.columns if c.endswith('_computation_time')]

    if investigated_parameter == 'k':
        indep = {
            find_k_in_name(c, max_k): c for c in all_time_cols
            if c.startswith('INDEPENDENT_INTERPRETABILITY_NDCG_HIT')
            and f'W_{fix_w}' in c
        }
        cond = {
            find_k_in_name(c, max_k): c for c in all_time_cols
            if c.startswith('INTERPRETABILITY_CONDITIONAL_VOLUMN_PR_WITH_NDCG_HIT')
            and f'W_{fix_w}' in c and f'M_{fix_m}' in c
        }
    elif investigated_parameter == 'w':
        indep = {
            find_w_in_name(c, max_w): c for c in all_time_cols
            if c.startswith('INDEPENDENT_INTERPRETABILITY_NDCG_HIT')
            and f'K_{fix_k}' in c
        }
        cond = {
            find_w_in_name(c, max_w): c for c in all_time_cols
            if c.startswith('INTERPRETABILITY_CONDITIONAL_VOLUMN_PR_WITH_NDCG_HIT')
            and f'K_{fix_k}' in c and f'M_{fix_m}' in c
        }
    else:  # 'm'
        indep = {}
        cond = {
            find_m_in_name(c, max_m): c for c in all_time_cols
            if c.startswith('INTERPRETABILITY_CONDITIONAL_VOLUMN_PR_WITH_NDCG_HIT')
            and f'K_{fix_k}' in c and f'W_{fix_w}' in c
        }

    indep.pop(-1, None)
    cond.pop(-1, None)
    indep_sorted = dict(sorted(indep.items()))
    cond_sorted = dict(sorted(cond.items()))
    return (
        list(indep_sorted.keys()), list(indep_sorted.values()),
        list(cond_sorted.keys()), list(cond_sorted.values()),
    )


# Consistent colors/markers for the three metrics across all panels
_METRIC_STYLE = {
    'ffvus':  dict(color='#9467bd', marker='o', linestyle='-'),
    'indep':  dict(color='#1f77b4', marker='s', linestyle='-'),
    'cond':   dict(color='#d62728', marker='^', linestyle='-'),
}
_FLAT_STYLE = dict(linewidth=2, linestyle='--')
_LINE_STYLE = dict(linewidth=2, markersize=4)


def plot_combined_computation_time(datasets_data, parameters, save_path):
    """Plot avg_ens computation time vs. hyperparameter for VUS-PR, IndepNDCG, aVUSi.

    parameters: list of investigated parameters, e.g. ['k', 'w', 'm'].
    Layout: n_datasets rows × n_parameters cols.
    Each panel shows three series:
      - VUS-PR      — always flat (no parameter dependence)
      - IndepNDCG   — varies with k/w; flat when varying m
      - aVUSi       — varies with k, w, and m
    Flat series are drawn as dashed horizontal lines.
    """
    n_datasets = len(datasets_data)
    n_params = len(parameters)

    fig, axes = plt.subplots(
        nrows=n_datasets, ncols=n_params,
        figsize=(2.5 * n_params, 2.5 * n_datasets),
        squeeze=False,
    )

    # Share y-axis between k and w columns per row (but not m)
    kw_indices = [i for i, p in enumerate(parameters) if p in ('k', 'w')]
    shared_col_indices = set(kw_indices[1:]) if len(kw_indices) >= 2 else set()
    if len(kw_indices) >= 2:
        for row_idx in range(n_datasets):
            ref_ax = axes[row_idx][kw_indices[0]]
            for col_idx in kw_indices[1:]:
                axes[row_idx][col_idx].sharey(ref_ax)

    for row_idx, (dataset_name, (df_full, df_filtered, visualized_metrics, cfg)) in enumerate(datasets_data.items()):
        fix_k = cfg.default_selected_k
        fix_w = cfg.default_selected_smoothing_window
        fix_m = cfg.default_selected_n_interpretability_sensitivity_levels

        df_avg = df_full[df_full['algorithm'] == 'avg_ens']

        # VUS-PR: mean and std are constant regardless of parameter
        ffvus_mean = ffvus_std = None
        if 'FFVUS_PR_computation_time' in df_avg.columns:
            ffvus_mean = df_avg['FFVUS_PR_computation_time'].mean()
            ffvus_std  = df_avg['FFVUS_PR_computation_time'].std()

        for col_idx, param in enumerate(parameters):
            ax = axes[row_idx][col_idx]
            indep_xs, indep_cols, cond_xs, cond_cols = _get_time_cols_by_param(df_full, cfg, param)

            if param == 'k' and dataset_name == 'smd':
                # exclude k=10 (boundary artefact in SMD)
                indep_pairs = [(x, c) for x, c in zip(indep_xs, indep_cols) if x != 10]
                cond_pairs  = [(x, c) for x, c in zip(cond_xs,  cond_cols)  if x != 10]
                indep_xs, indep_cols = ([p[0] for p in indep_pairs], [p[1] for p in indep_pairs])
                cond_xs,  cond_cols  = ([p[0] for p in cond_pairs],  [p[1] for p in cond_pairs])

            if param == 'k':
                x_range = cond_xs or indep_xs
                xlabel = 'k'
                title = f'w={fix_w}, M={fix_m}'
            elif param == 'w':
                x_range = cond_xs or indep_xs
                xlabel = 'w'
                title = f'k={fix_k}, M={fix_m}'
            else:
                x_range = cond_xs
                xlabel = 'M'
                title = f'k={fix_k}, w={fix_w}'

            if not x_range:
                continue

            # ── VUS-PR: flat (only for k/w panels) ───────────────────
            if param != 'm' and ffvus_mean is not None:
                ax.axhline(ffvus_mean, label='VUS-PR',
                           color=_METRIC_STYLE['ffvus']['color'],
                           linewidth=2, linestyle='--')

            # ── IndepNDCG: varying for k/w only ──────────────────────
            if param != 'm' and indep_cols:
                means = df_avg[indep_cols].mean().values
                stds  = df_avg[indep_cols].std().values
                color = _METRIC_STYLE['indep']['color']
                ax.plot(indep_xs, means, label='IndepNDCG',
                        **{**_METRIC_STYLE['indep'], **_LINE_STYLE})
                ax.fill_between(indep_xs,
                                np.maximum(means - stds, 1e-9),
                                means + stds,
                                color=color, alpha=0.15)

            # ── aVUSi: always varying ─────────────────────────────────
            if cond_cols:
                means = df_avg[cond_cols].mean().values
                stds  = df_avg[cond_cols].std().values
                color = _METRIC_STYLE['cond']['color']
                ax.plot(cond_xs, means, label='aVUSi',
                        **{**_METRIC_STYLE['cond'], **_LINE_STYLE})
                ax.fill_between(cond_xs,
                                np.maximum(means - stds, 1e-9),
                                means + stds,
                                color=color, alpha=0.15)
                ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

            if param != 'm':
                ax.set_yscale('log')
                ax.set_xlim(x_range[0], x_range[-1])
                ylabel = 'Time (s): [log]'
            else:
                ax.set_xlim(10, 50)
                ylabel = 'Time (s): [linear]'
            ax.annotate(title, xy=(0.05, 0.85), xycoords='axes fraction',
                        ha='left', va='top', fontsize=FONT_SIZE - 2,
                        bbox=dict(boxstyle='round,pad=0.11', fc='#FDE992', alpha=0.9, ec='#A08C3A', linewidth=1.0))
            if row_idx == 0:
                ax.set_xlabel(xlabel, fontsize=FONT_SIZE, fontweight='bold')
                ax.xaxis.set_label_position('top')
                # ax.xaxis.tick_top()
            else:
                ax.set_xlabel('')
            if col_idx == 0:
                ax.set_ylabel(ylabel, fontsize=FONT_SIZE)
            elif col_idx in shared_col_indices:
                ax.set_ylabel('')
                plt.setp(ax.get_yticklabels(), visible=False)
            else:
                ax.set_ylabel(ylabel, fontsize=FONT_SIZE, rotation=270, labelpad=20)
                ax.yaxis.set_label_position('right')
                ax.yaxis.tick_right()
            ax.tick_params(axis='both', labelsize=10)
            ax.xaxis.set_major_formatter(FormatStrFormatter('%d' if param in ('k', 'm', 'w') else '%.1f'))
            # ax.grid(True, which='both', linestyle='--', alpha=0.5)

        axes[row_idx][0].annotate(
            DATASET_DISPLAY_NAMES[dataset_name],
            xy=(-0.48, 0.5), xycoords='axes fraction',
            fontsize=FONT_SIZE + 2, fontweight='bold', va='center', rotation=90,
        )

    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(
        handles, labels,
        loc='lower center', bbox_to_anchor=(0.5, -0.02),
        ncol=3, frameon=True, fontsize=FONT_SIZE,
    )
    plt.tight_layout(rect=(0, 0.06, 1, 1))
    plt.subplots_adjust(wspace=0.09)
    ext = save_path.rsplit('.', 1)[-1] if '.' in save_path else 'pdf'
    fig.savefig(save_path, dpi=300, bbox_inches='tight', format=ext)
    print(f'Saved: {save_path}')
    # plt.close(fig)
    return fig


# ── Figure 5b: Compact computation time (3 rows × 2 cols, single-column) ─────
def plot_compact_computation_time(datasets_data, parameters, save_path):
    """3 rows (parameters) × 2 cols (datasets) layout sized for one column
    in a double-column paper (~3.5 in wide).

    Rows: k, w, m parameters.  Columns: Synthetic / SMD.
    k/w rows share y-axis across columns; m row is independent (linear scale).
    """
    _SF = 7    # small font size for tick labels
    _MF = 8    # medium font size for axis labels / titles

    n_params   = len(parameters)
    n_datasets = len(datasets_data)
    datasets_list = list(datasets_data.items())

    fig, axes = plt.subplots(
        nrows=n_params, ncols=n_datasets,
        figsize=(5, 2 * n_params),
        squeeze=False,
    )

    # Share y within each row for k and w (same parameter, both datasets)
    for row_idx, param in enumerate(parameters):
        if param in ('k', 'w'):
            ref = axes[row_idx][0]
            for col_idx in range(1, n_datasets):
                axes[row_idx][col_idx].sharey(ref)

    _bullets = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)']
    _row_labels = {
        'k': 'Varying ranking cutoff k',
        'w': 'Varying smoothing window w',
        'm': 'Varying interpretability sensitivity M',
    }

    for row_idx, param in enumerate(parameters):
        for col_idx, (dataset_name, (df_full, df_filtered, visualized_metrics, cfg)) in enumerate(datasets_list):
            ax = axes[row_idx][col_idx]
            fix_k = cfg.default_selected_k
            fix_w = cfg.default_selected_smoothing_window
            fix_m = cfg.default_selected_n_interpretability_sensitivity_levels

            df_avg = df_full[df_full['algorithm'] == 'avg_ens']

            ffvus_mean = None
            if 'FFVUS_PR_computation_time' in df_avg.columns:
                ffvus_mean = df_avg['FFVUS_PR_computation_time'].mean()

            indep_xs, indep_cols, cond_xs, cond_cols = _get_time_cols_by_param(df_full, cfg, param)

            if param == 'k' and dataset_name == 'smd':
                indep_pairs = [(x, c) for x, c in zip(indep_xs, indep_cols) if x != 10]
                cond_pairs  = [(x, c) for x, c in zip(cond_xs,  cond_cols)  if x != 10]
                indep_xs, indep_cols = ([p[0] for p in indep_pairs], [p[1] for p in indep_pairs])
                cond_xs,  cond_cols  = ([p[0] for p in cond_pairs],  [p[1] for p in cond_pairs])

            if param == 'k':
                x_range, xlabel = cond_xs or indep_xs, 'k'
            elif param == 'w':
                x_range, xlabel = cond_xs or indep_xs, 'w'
            else:
                x_range, xlabel = cond_xs, 'M'

            if not x_range:
                continue

            lw, ms = 1.2, 3

            # VUS-PR flat (k/w rows only)
            if param != 'm' and ffvus_mean is not None:
                ax.axhline(ffvus_mean, label='VUS-PR',
                           color=_METRIC_STYLE['ffvus']['color'],
                           linewidth=lw, linestyle='--')

            # IndepNDCG varying (k/w rows only)
            if param != 'm' and indep_cols:
                means = df_avg[indep_cols].mean().values
                stds  = df_avg[indep_cols].std().values
                color = _METRIC_STYLE['indep']['color']
                ax.plot(indep_xs, means, label='IndepNDCG',
                        color=color, marker='s', linewidth=lw, markersize=ms)
                ax.fill_between(indep_xs,
                                np.maximum(means - stds, 1e-9), means + stds,
                                color=color, alpha=0.15)

            # aVUSi always varying
            if cond_cols:
                means = df_avg[cond_cols].mean().values
                stds  = df_avg[cond_cols].std().values
                color = _METRIC_STYLE['cond']['color']
                ax.plot(cond_xs, means, label='aVUSi',
                        color=color, marker='^', linewidth=lw, markersize=ms)
                ax.fill_between(cond_xs,
                                np.maximum(means - stds, 1e-9), means + stds,
                                color=color, alpha=0.15)

            if param != 'm':
                ax.set_yscale('log')
                ax.set_xlim(x_range[0], x_range[-1])
            else:
                ax.set_xlim(10, 50)

            # Dataset name as column title (first row only)
            if row_idx == 0:
                ax.set_title(DATASET_DISPLAY_NAMES[dataset_name],
                             fontsize=_MF, fontweight='bold', pad=4)

            ax.set_xlabel(xlabel, fontsize=_MF)
            ax.tick_params(axis='both', labelsize=_SF)

            # y-label on first column only, with scale info
            if col_idx == 0:
                scale_tag = 'log' if param != 'm' else 'linear'
                ax.set_ylabel(f'Time (s) [{scale_tag}]', fontsize=_MF)
            else:
                ax.set_ylabel('')

    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(
        handles, labels,
        loc='lower center', bbox_to_anchor=(0.5, 0.0),
        ncol=3, frameon=True, fontsize=_SF,
    )
    plt.tight_layout(rect=(0, 0.05, 1, 1))

    # Row labels centered across the full row, placed after layout so
    # axes positions are finalised and fig.text can use figure coordinates.
    # for row_idx, param in enumerate(parameters):
    #     bbox = axes[row_idx][0].get_position()
    #     bullet = _bullets[row_idx] if row_idx < len(_bullets) else ''
    #     row_label = _row_labels.get(param, f'Varying {param}')
    #     fig.text(
    #         0.5, bbox.y0 - 0.06,
    #         f'{bullet} {row_label}',
    #         fontsize=_SF, fontweight='bold',
    #         ha='center', va='top',
    #         transform=fig.transFigure,
    #     )
    ext = save_path.rsplit('.', 1)[-1] if '.' in save_path else 'pdf'
    fig.savefig(save_path, dpi=300, bbox_inches='tight', format=ext)
    print(f'Saved: {save_path}')
    # plt.close(fig)
    return fig


# ── Figure 6: Time complexity vs. series length (synthetic / avg_ens) ────────
def _scatter_with_trend(ax, x, y, label, color, marker):
    mask = ~(np.isnan(x) | np.isnan(y))
    xc, yc = x[mask], y[mask]
    yc = np.log(yc)
    ax.scatter(xc, yc, color=color, marker=marker, alpha=0.35, s=18, label=label)
    coeffs = np.polyfit(xc.astype(float), yc.astype(float), 1)
    x_line = np.linspace(xc.min(), xc.max(), 300)
    ax.plot(x_line, np.polyval(coeffs, x_line), color=color, linewidth=2, linestyle='--')


def plot_time_complexity_vs_series_length(save_path):
    """Side-by-side figure: VUS-PR, IndepNDCG and aVUSi computation time vs. series length.

    One column per dataset (Synthetic / SMD), both using avg_ens.
    Each panel plots scatter + linear trend line for all three metrics.
    """
    fig, axes = plt.subplots(1, len(DATASET_NAMES), figsize=(9 * len(DATASET_NAMES), 5),
                             squeeze=False)

    for col_idx, dataset_name in enumerate(DATASET_NAMES):
        ds_cfg = config['parameter_sensitivity_analysis']['datasets'][dataset_name]
        selected_k = ds_cfg.default_selected_k
        selected_w = ds_cfg.default_selected_smoothing_window
        selected_m = ds_cfg.default_selected_n_interpretability_sensitivity_levels

        df_results = pd.read_csv(RESULTS_PATHS[dataset_name])
        df_info = pd.read_csv(SERIES_INFO_PATHS[dataset_name])

        df_avg = df_results[df_results['algorithm'] == 'avg_ens'].copy()
        df = df_avg.merge(df_info[['test_batch_id', 'length']], on='test_batch_id', how='inner')

        time_specs = [
            (
                'FFVUS_PR_computation_time',
                'VUS-PR',
                random_colors[4], '^',
            ),
            (
                f'INDEPENDENT_INTERPRETABILITY_NDCG_HIT_K_{selected_k}_W_{selected_w}'
                f'_SCORE_WITH_SMOOTHING_computation_time',
                f'IndepNDCG (k={selected_k}, w={selected_w})',
                random_colors[0], 'o',
            ),
            (
                f'INTERPRETABILITY_CONDITIONAL_VOLUMN_PR_WITH_NDCG_HIT_K_{selected_k}_W_{selected_w}'
                f'_M_{selected_m}_SCORE_COMBINATION_WITH_SMOOTHING_computation_time',
                f'aVUSi (k={selected_k}, w={selected_w}, M={selected_m})',
                random_colors[2], 's',
            ),
        ]

        ax = axes[0][col_idx]
        x = df['length'].values
        for time_col, label, color, marker in time_specs:
            if time_col in df.columns:
                _scatter_with_trend(ax, x, df[time_col].values, label, color, marker)

        ax.set_xlabel('Time Series Length', fontsize=FONT_SIZE)
        ax.set_ylabel('Computation Time (s)', fontsize=FONT_SIZE)
        ax.set_title(DATASET_DISPLAY_NAMES[dataset_name], fontsize=FONT_SIZE + 1, fontweight='bold')
        ax.legend(fontsize=FONT_SIZE - 2, frameon=True, loc='upper left')
        ax.grid(linestyle='--', alpha=0.5)
        ax.tick_params(axis='both', labelsize=10)

    fig.suptitle('Scalability Analysis — AvgEns', fontsize=FONT_SIZE + 2, fontweight='bold')
    fig.tight_layout()
    ext = save_path.rsplit('.', 1)[-1] if '.' in save_path else 'pdf'
    fig.savefig(save_path, dpi=300, bbox_inches='tight', format=ext)
    print(f'Saved: {save_path}')
    # plt.close(fig)
    return fig

def plot_metrics_for_a_single_time_series(datasetname, time_series_id, save_path=None):
    """Table of all-detector × three-metric scores for one time series.

    Rows: detectors (base detectors + avg_ens).
    Columns: VUS-PR, IndepNDCG, aVUSi at default k/w/M.
    Best value per column highlighted green; avg_ens row highlighted gold.
    """
    df_full, df_filtered, visualized_metrics, cfg = load_dataset(datasetname)
    all_algs = [
        'hbos', 'denoising_auto_encoder', 'cblof', 'auto_encoder', 'omni_anomaly',
        'random_black_forest', 'mtad_gat', 'copod', 'tran_ad', 'encdec_ad',
    ]

    ts_df = df_filtered[df_filtered['test_batch_id'] == time_series_id]
    if ts_df.empty:
        print(f'Warning: time_series_id "{time_series_id}" not found in {datasetname}')
        return

    # Column 0 = Detector name; columns 1-4 = metric values
    col_labels = ['Detector', 'VUS-\nPR', 'Indep\nNDCG', 'Avg.\n(i.e. AddC)', 'aVUSi']
    n_metric_cols = len(col_labels) - 1  # 4

    # Build value matrix (strings for display, floats for comparison)
    cell_vals  = []
    cell_floats = []
    for alg in all_algs:
        alg_row = ts_df[ts_df['algorithm'] == alg]
        str_row, flt_row = [], []
        for metric in visualized_metrics:
            if len(alg_row) > 0 and metric in alg_row.columns:
                raw = alg_row[metric].values[0]
                if pd.notna(raw):
                    str_row.append(f'{float(raw):.2f}')
                    flt_row.append(float(raw))
                else:
                    str_row.append('—')
                    flt_row.append(float('-inf'))
            else:
                str_row.append('—')
                flt_row.append(float('-inf'))
        # Avg = mean of VUS-PR (index 0) and IndepNDCG (index 1)
        v0, v1 = flt_row[0], flt_row[1]
        if v0 != float('-inf') and v1 != float('-inf'):
            avg_val = (v0 + v1) / 2
            str_row.append(f'{avg_val:.2f}')
            flt_row.append(avg_val)
        else:
            str_row.append('—')
            flt_row.append(float('-inf'))
        # Reorder to: VUS-PR, IndepNDCG, Avg, aVUSi; prepend detector name
        str_row = [str_row[0], str_row[1], str_row[3], str_row[2]]
        flt_row = [flt_row[0], flt_row[1], flt_row[3], flt_row[2]]
        cell_vals.append([alg_rename_map.get(alg, alg)] + str_row)
        cell_floats.append(flt_row)

    fig, ax = plt.subplots(figsize=(5, len(all_algs) * 0.42))
    ax.axis('off')

    tbl = ax.table(
        cellText=cell_vals,
        colLabels=col_labels,
        cellLoc='center',
        loc='center',
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1.0, 1.4)

    # Double the header row height to fit two-line labels
    data_h = tbl[1, 0].get_height()
    for col_idx in range(len(col_labels)):
        tbl[0, col_idx].set_height(data_h * 1.5)

    # Style header row
    for col_idx in range(len(col_labels)):
        tbl[0, col_idx].set_facecolor('#BDD7EE')
        tbl[0, col_idx].set_text_props(color='black', fontweight='bold')

    # Bold detector-name cell for highlighted detectors
    bold_algs = {'tran_ad', 'random_black_forest', 'denoising_auto_encoder'}
    for row_idx, alg in enumerate(all_algs):
        if alg in bold_algs:
            tbl[row_idx + 1, 0].set_text_props(fontweight='bold')

    # White background and black text for all cells
    for row_idx in range(len(all_algs)):
        for col_idx in range(len(col_labels)):
            tbl[row_idx + 1, col_idx].set_facecolor('white')
            tbl[row_idx + 1, col_idx].set_text_props(color='black')

    # Highlight best group (green 0.6), 2nd-best group (green 0.3), worst (red) per metric column
    for metric_idx in range(n_metric_cols):
        tbl_col = metric_idx + 1  # skip Detector column
        col_vals = [cell_floats[r][metric_idx] for r in range(len(all_algs))]
        valid = [(v, r) for r, v in enumerate(col_vals) if v != float('-inf')]
        if not valid:
            continue
        sorted_valid = sorted(valid, key=lambda x: x[0], reverse=True)
        _, worst_row = min(valid, key=lambda x: x[0])

        # All rows sharing the best rounded value → alpha 0.6
        best_val = round(sorted_valid[0][0], 2)
        top_group = [(v, r) for v, r in sorted_valid if round(v, 2) == best_val]
        # Only highlight 2nd-best group when there is exactly one top-1 detector
        remaining = [(v, r) for v, r in sorted_valid if round(v, 2) != best_val]
        second_group = []
        if len(top_group) == 1 and remaining:
            second_val = round(remaining[0][0], 2)
            second_group = [(v, r) for v, r in remaining if round(v, 2) == second_val]

        for _, row_idx in top_group:
            tbl[row_idx + 1, tbl_col].set_facecolor((0/256, 128/256, 0/256, 0.6))
            tbl[row_idx + 1, tbl_col].set_text_props(color='black')
        for _, row_idx in second_group:
            tbl[row_idx + 1, tbl_col].set_facecolor((0/256, 128/256, 0/256, 0.3))
            tbl[row_idx + 1, tbl_col].set_text_props(color='black')
        tbl[worst_row + 1, tbl_col].set_facecolor((1, 0, 0, 0.2))
        tbl[worst_row + 1, tbl_col].set_text_props(color='black')

    # Color legend annotation
    from matplotlib.patches import Patch
    legend_handles = [
        Patch(facecolor=(1, 0, 0, 0.2), edgecolor='#aaaaaa', linewidth=0.5,
              label='Worst'),
        Patch(facecolor=(0, 128/256, 0, 0.6), edgecolor='#aaaaaa', linewidth=0.5,
              label='Top-1'),
        Patch(facecolor=(0, 128/256, 0, 0.3), edgecolor='#aaaaaa', linewidth=0.5,
              label='Top-2'),
    ]
    ax.legend(
        handles=legend_handles,
        loc='lower right',
        bbox_to_anchor=(0.55, 0.92),
        ncol=3,
        fontsize=8,
        frameon=True,
        edgecolor='#cccccc',
        handlelength=1.2,
        handleheight=1.0,
    )

    if save_path is None:
        save_path = f'{OUTPUT_DIR}/metrics_table_{datasetname}_{time_series_id}.png'
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f'Saved: {save_path}')
    # plt.close(fig)
    return fig


def plot_vusi_curve(datasetname, time_series_id, visualized_detectors, save_path=None):
    """Plot aVUSi curves (VUS-PR vs recall threshold M) for selected detectors.

    For each detector, also draws horizontal lines for the scalar VUS-PR (dashed)
    and aVUSi (dotted) values loaded from results.csv.
    """
    cfg = config['parameter_sensitivity_analysis']['datasets'][datasetname]
    selected_k = cfg.default_selected_k
    selected_w = cfg.default_selected_smoothing_window
    selected_m = cfg.default_selected_n_interpretability_sensitivity_levels

    x_prefix = f'n_interpretability_sensitivity_levels_{selected_m}_M_list_M_'
    y_prefix  = f'n_interpretability_sensitivity_levels_{selected_m}_vus_pr_list_M_'

    avusi_col = (f'INTERPRETABILITY_CONDITIONAL_VOLUMN_PR_WITH_NDCG_HIT_K_{selected_k}'
                 f'_W_{selected_w}_M_{selected_m}_SCORE_COMBINATION_WITH_SMOOTHING')
    vuspr_col = 'FFVUS_PR'

    # Load results.csv and filter for this time series
    df_results = pd.read_csv(RESULTS_PATHS[datasetname])
    ts_id_bare = time_series_id.replace('.out', '')
    ts_mask = df_results['test_batch_id'].astype(str).str.replace('.out', '', regex=False) == ts_id_bare
    df_ts = df_results[ts_mask]

    n_det = len(visualized_detectors)

    # (2.5 * n_params, 2.5 * n_datasets)
    fig, axes = plt.subplots(1, n_det, figsize=( 2.5*n_det, 2.5*1), squeeze=False, sharey='row')
    base_dir     = VUSI_CURVE_DATA_PATHS[datasetname]
    all_det_keys = list(based_detector_map.keys())
    dataset_label = DATASET_DISPLAY_NAMES.get(datasetname, datasetname)

    for col_idx, detector in enumerate(visualized_detectors):
        ax = axes[0][col_idx]
        det_idx = all_det_keys.index(detector) if detector in all_det_keys else 0
        color   = random_colors[det_idx % len(random_colors)]
        marker  = random_markers[det_idx % len(random_markers)]
        label   = alg_rename_map.get(detector, detector)

        # ── aVUSi curve from new_metrics CSV ─────────────────────────
        x_vals_plot = None
        csv_path = os.path.join(base_dir, detector, 'vus_pr_list.csv')
        if os.path.exists(csv_path):
            df_curve = pd.read_csv(csv_path, index_col=0)
            row = None
            for candidate in [time_series_id, ts_id_bare, ts_id_bare + '.out']:
                if candidate in df_curve.index:
                    row = df_curve.loc[candidate]
                    break
            if row is not None:
                x_cols = sorted([c for c in df_curve.columns if c.startswith(x_prefix)],
                                key=lambda c: int(c.split('_M_')[-1]))
                y_cols = sorted([c for c in df_curve.columns if c.startswith(y_prefix)],
                                key=lambda c: int(c.split('_M_')[-1]))
                x_vals_plot = row[x_cols].astype(float).values
                y_vals = row[y_cols].astype(float).values
                ax.fill_between(x_vals_plot, 0, y_vals, color=color, alpha=0.25,
                                # label='aVUSi (area)'
                                )
                ax.plot(x_vals_plot, y_vals, color=color, marker=marker,
                        label=f'{label} VUSi', linewidth=1.8, markersize=4, markevery=5)

        # ── VUS-PR horizontal line and aVUSi annotation ───────────────
        det_row = df_ts[df_ts['algorithm'] == detector]
        if not det_row.empty and vuspr_col in det_row.columns:
            vuspr_val = float(det_row[vuspr_col].values[0])
            ax.axhline(vuspr_val, color=color, linestyle='--',
                       linewidth=1.5, label=f'{label} VUS-PR = {vuspr_val:.2f}')
            ax.annotate(f'VUS-PR={vuspr_val:.2f}',
                        xy=(0.9, vuspr_val*0.95),
                        xycoords=('axes fraction', 'data'),
                        ha='right', va='top', fontsize=10, color='black',
                        bbox=dict(boxstyle='round,pad=0.1', fc='white', ec='none', alpha=0.7))
        if not det_row.empty and avusi_col in det_row.columns:
            avusi_val = float(det_row[avusi_col].values[0])
            if x_vals_plot is not None:
                x_center = (x_vals_plot[0] + x_vals_plot[-1]) / 2
                ax.text(x_center, avusi_val * 0.21, f'aVUSi={avusi_val:.2f}',
                        ha='center', va='center', fontsize=10,
                        color=color, fontweight='bold')


        ax.set_title(f'{label}', fontsize=FONT_SIZE, fontweight='bold')
        if col_idx == 0:
            ax.set_ylabel('VUSi', fontsize=FONT_SIZE - 2)
        ax.grid(axis='both', linestyle='--', alpha=0.5)
        ax.tick_params(axis='both', labelsize=10)
        ax.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))

    # fig.suptitle(f'{dataset_label} — {time_series_id}',
    #              fontsize=FONT_SIZE, fontweight='bold')
    fig.supxlabel('Interpretability sensitivity level $m$', fontsize=FONT_SIZE - 2, y=0.12)

    # Collect handles/labels from all subplots, deduplicate by label
    seen, handles_all, labels_all = set(), [], []
    for ax in axes[0]:
        for h, l in zip(*ax.get_legend_handles_labels()):
            if l not in seen:
                seen.add(l)
                handles_all.append(h)
                labels_all.append(l)

    # Key by detector display name so both rows stay aligned column-by-column
    det_to_vusi  = {}
    det_to_vuspr = {}
    for h, l in zip(handles_all, labels_all):
        if 'VUSi' in l:
            det_to_vusi[l.replace(' VUSi', '')] = (h, l)
        elif 'VUS-PR' in l:
            det_to_vuspr[l.split(' VUS-PR')[0]] = (h, l)

    # Build both rows in the same detector order; blank placeholders keep columns aligned
    # blank = mlines.Line2D([], [], color='none')
    row1_h, row1_l, row2_h, row2_l = [], [], [], []
    for detector in visualized_detectors:
        det_label = alg_rename_map.get(detector, detector)
        if det_label in det_to_vusi:
            row1_h.append(det_to_vusi[det_label][0])
            row1_l.append(det_to_vusi[det_label][1])

        if det_label in det_to_vuspr:
            row2_h.append(det_to_vuspr[det_label][0])
            row2_l.append(det_to_vuspr[det_label][1])

    combined_handles = []
    combined_labels = []
    for h1, h2 in zip(row1_h, row2_h):
        combined_handles.append(h1)
        combined_handles.append(h2)
    for l1,l2 in zip(row1_l, row2_l):
        combined_labels.append(l1)
        combined_labels.append(l2)

    ncols = len(visualized_detectors)

    # tight_layout first, then place legend so it isn't overridden
    fig.tight_layout(rect=[0, 0.01, 1, 1])
    print('number of legends', len(combined_handles))
    fig.legend(combined_handles, combined_labels,
               loc='lower center', bbox_to_anchor=(0.5, -0.12),
               ncol=ncols, fontsize=FONT_SIZE -3, frameon=True)

    if save_path is None:
        save_path = f'{OUTPUT_DIR}/vusi_curve_{datasetname}_{time_series_id}.png'
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f'Saved: {save_path}')
    # plt.close(fig)
    return fig


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    sns.set_style('white')

    datasets_data = {}
    for dataset_name in DATASET_NAMES:
        df_full, df_filtered, visualized_metrics, cfg = load_dataset(dataset_name)
        datasets_data[dataset_name] = (df_full, df_filtered, visualized_metrics, cfg)
        print(f'Loaded {dataset_name}: {len(df_full)} rows, k={cfg.default_selected_k}, '
              f'w={cfg.default_selected_smoothing_window}, m={cfg.default_selected_n_interpretability_sensitivity_levels}')

    # # Figure 1: Combined boxplot (vertical, 2 rows × 3 cols)
    plot_combined_three_metrics_boxplot(
        datasets_data,
        save_path=f'{OUTPUT_DIR}/combined_three_metrics_boxplot.png',
    )

    # Figure 2: Combined boxplot (horizontal, 3 rows × 2 cols)
    plot_combined_three_metrics_boxplot_horizontal(
        datasets_data,
        save_path=f'{OUTPUT_DIR}/combined_three_metrics_boxplot_horizontal.png',
    )

    # Figure 4: Combined hyper-parameter sensitivity (2 rows × 5 cols)
    plot_combined_hyper_parameter_sensitivity(
        datasets_data,
        hyper_parameters=['k', 'w', 'm'],
        save_path=f'{OUTPUT_DIR}/combined_all_hyper_parameters_sensitivity.png',
    )

    # Figure 5: Computation time scalability analysis (2 rows × 3 cols)
    plot_combined_computation_time(
        datasets_data,
        parameters=['k', 'w', 'm'],
        save_path=f'{OUTPUT_DIR}/combined_computation_time_scalability.png',
    )

    # # Figure 3: Combined average ranking (1 row × 2 cols)
    plot_combined_average_ranking(
        datasets_data,
        save_path=f'{OUTPUT_DIR}/combined_average_ranking_plot.png',
    )

    plot_critical_difference(
        datasets_data,
        save_path=f'{OUTPUT_DIR}/combined_critical_difference_diagram.png',
    )

    # Figure 5b: Compact computation time (3 rows × 2 cols, single-column)
    plot_compact_computation_time(
        datasets_data,
        parameters=['k', 'w', 'm'],
        save_path=f'{OUTPUT_DIR}/compact_computation_time_scalability.png',
    )

    # Figure 6: Time complexity vs. series length (synthetic only)
    plot_time_complexity_vs_series_length(
        save_path=f'{OUTPUT_DIR}/time_complexity_vs_series_length.png',
    )

    visualized_detectors = ['denoising_auto_encoder',
                            'tran_ad',
                            'random_black_forest',
                            # 'hbos',
                            # 'auto_encoder',
                            # 'cblof',
                            # 'omni_anomaly',
                            # 'copod',
                            # 'mtad_gat',
                            # 'encdec_ad'
                            # 'avg_ens',
                            ]

    time_series_id = 'synthetic_batch_999.out'
    dataset_name = 'settings_six'
    plot_metrics_for_a_single_time_series(datasetname=dataset_name,
                                          time_series_id=time_series_id,
                                          save_path=f'{OUTPUT_DIR}/metrics_for_time_series_{time_series_id}_in_{dataset_name}.png',)
    #
    plot_vusi_curve(datasetname=dataset_name,
                                          time_series_id=time_series_id,visualized_detectors=visualized_detectors,
                                          save_path=f'{OUTPUT_DIR}/vusi_curve_for_time_series_{time_series_id}_in_{dataset_name}.png', )

    time_series_id = 'machine-3-1.out'
    dataset_name = 'smd'
    plot_metrics_for_a_single_time_series(datasetname=dataset_name,
                                          time_series_id=time_series_id,
                                          save_path=f'{OUTPUT_DIR}/metrics_for_time_series_{time_series_id}_in_{dataset_name}.png',)

    plot_vusi_curve(datasetname=dataset_name,
                    time_series_id=time_series_id, visualized_detectors=visualized_detectors,
                    save_path=f'{OUTPUT_DIR}/vusi_curve_for_time_series_{time_series_id}_in_{dataset_name}.png', )
    print(f'\nAll combined figures saved to {OUTPUT_DIR}/')