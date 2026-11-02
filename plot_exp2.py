import csv
import glob
import os
import warnings
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import plotly.graph_objects as go
import yaml
from plotly.subplots import make_subplots

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
FIGURES_DIR = os.path.join(os.path.dirname(__file__), "figures")


MODELS = ["gvp", "mpnn", "pifold"]
MODEL_LABELS = {"gvp": "GVP", "mpnn": "MPNN", "pifold": "PiFold"}

TRAIN_SETS = ["inv", "afcath", "affull"]
DATASET_ALIASES = {
    "alphafold": "affull",
}

EVAL_SETS = ["inv", "afcath"]
SET_LABELS = {
    "inv": "CATH",
    "afcath": "AF-CATH",
    "affull": "AF-SwissProt",
}

FEATURES = ["gvp", "mpnn", "pifold"]
FEATURE_LABELS = {
    "gvp": "GVP",
    "mpnn": "MPNN",
    "pifold": "PiFold",
}

METRIC = "Recovery"

# Three shades of green, ordered GVP, MPNN, PiFold (left to right)
FEATURE_COLORS = {
    "gvp": "rgba(161, 217, 155, 1.0)",
    "mpnn": "rgba(116, 196, 118, 1.0)",
    "pifold": "rgba(49, 163, 84, 1.0)",
}


def metric_display_values(values: List[float]) -> np.ndarray:
    return np.asarray(values, dtype=float) * 100.0


def normalize_dataset(dataset: str) -> str:
    return DATASET_ALIASES.get(dataset, dataset)


def parse_experiment_name(exp_name: str) -> Tuple[str, str, str, int]:
    """
    Parse names like exp2_gvp_mpnn_inv_1 into (model, feature, train_set, seed).
    """
    parts = exp_name.split("_")
    if len(parts) != 5 or parts[0] != "exp2":
        raise ValueError(f"Unexpected experiment name format: {exp_name}")

    model, feature = parts[1], parts[2]
    train_set = normalize_dataset(parts[3])
    seed = int(parts[4])

    if model not in MODELS:
        raise ValueError(f"Unknown model '{model}' in experiment name '{exp_name}'")
    if train_set not in TRAIN_SETS:
        raise ValueError(f"Unknown train set '{train_set}' in experiment name '{exp_name}'")
    if feature not in FEATURES:
        raise ValueError(f"Unknown feature '{feature}' in experiment name '{exp_name}'")

    return model, feature, train_set, seed


def experiment_name(model: str, feature: str, train_set: str, seed: int) -> str:
    return f"exp2_{model}_{feature}_{train_set}_{seed}"


def expected_experiment_names(seeds: Tuple[int, ...] = (1,)) -> List[str]:
    return [
        experiment_name(model, feature, train_set, seed)
        for seed in seeds
        for model in MODELS
        for train_set in TRAIN_SETS
        for feature in FEATURES
    ]


def eval_result_path(exp_name: str, eval_set: str) -> str:
    """
    CATH eval stays at the historical test_results.yaml path. AF-CATH eval is
    written to results/<exp>/eval_afcath/test_results.yaml so it does not
    overwrite the CATH numbers. A flat test_results_afcath.yaml is also accepted.
    """
    exp_dir = os.path.join(RESULTS_DIR, exp_name)
    if eval_set == "inv":
        return os.path.join(exp_dir, "test_results.yaml")
    if eval_set == "afcath":
        nested = os.path.join(exp_dir, "eval_afcath", "test_results.yaml")
        if os.path.isfile(nested):
            return nested
        return os.path.join(exp_dir, "test_results_afcath.yaml")
    raise ValueError(f"Unknown eval set '{eval_set}'")


def read_recovery(path: str) -> Optional[float]:
    if not os.path.isfile(path):
        return None
    with open(path, "r") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict) or METRIC not in data:
        return None
    try:
        return float(data[METRIC])
    except (TypeError, ValueError):
        return None


def collect_results() -> Tuple[
    Dict[str, Dict[str, Dict[str, Dict[str, List[float]]]]],
    Dict[str, List[str]],
    int,
]:
    """
    Aggregate Recovery values:
    results[train_set][eval_set][model][feature] = list of values over seeds

    Also return missing experiment names per eval set.
    """
    results: Dict[str, Dict[str, Dict[str, Dict[str, List[float]]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    )

    candidates = set(expected_experiment_names())
    pattern = os.path.join(RESULTS_DIR, "exp2_*")
    for exp_dir in glob.glob(pattern):
        if not os.path.isdir(exp_dir):
            continue
        exp_name = os.path.basename(exp_dir)
        try:
            parse_experiment_name(exp_name)
        except ValueError:
            continue
        candidates.add(exp_name)

    missing: Dict[str, List[str]] = {eval_set: [] for eval_set in EVAL_SETS}
    for exp_name in sorted(candidates):
        try:
            model, feature, train_set, seed = parse_experiment_name(exp_name)
        except ValueError:
            continue

        for eval_set in EVAL_SETS:
            val = read_recovery(eval_result_path(exp_name, eval_set))
            if val is None:
                missing[eval_set].append(exp_name)
                continue
            results[train_set][eval_set][model][feature].append(val)

    return results, missing, len(candidates)


def ensure_figures_dir() -> None:
    os.makedirs(FIGURES_DIR, exist_ok=True)


def export_source_data(
    results: Dict[str, Dict[str, Dict[str, Dict[str, List[float]]]]],
) -> None:
    """
    Export the per-run Recovery values used in exp2 plots to a long-form CSV
    for source data archiving.
    """
    out_path = os.path.join(FIGURES_DIR, "source_data_plot2.csv")
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["train", "eval", "model", "feature", "value"])
        for train_set in TRAIN_SETS:
            for eval_set in EVAL_SETS:
                for model in MODELS:
                    for feature in FEATURES:
                        values = results[train_set][eval_set][model].get(feature, [])
                        for v in values:
                            writer.writerow(
                                [
                                    train_set,
                                    eval_set,
                                    model,
                                    feature,
                                    float(v),
                                ]
                            )


def panel_title(train_set: str, eval_set: str) -> str:
    return f"<b>{SET_LABELS[train_set]} → {SET_LABELS[eval_set]}</b>"


def build_figure(
    results: Dict[str, Dict[str, Dict[str, Dict[str, List[float]]]]],
) -> go.Figure:
    subplot_titles = [
        panel_title(train_set, eval_set)
        for eval_set in EVAL_SETS
        for train_set in TRAIN_SETS
    ]
    fig_height = 700
    margin_top = 90
    legend_y = 1.0 - (margin_top / fig_height) / 2

    fig = make_subplots(
        rows=len(EVAL_SETS),
        cols=len(TRAIN_SETS),
        shared_yaxes=True,
        horizontal_spacing=0.06,
        vertical_spacing=0.14,
        subplot_titles=subplot_titles,
    )

    fig.update_layout(
        template="plotly_white",
        width=1800,
        height=fig_height,
        font=dict(
            family="DejaVu Sans Mono, Menlo, Monaco, Consolas, monospace", size=12
        ),
        legend=dict(
            title="Features",
            orientation="h",
            xref="container",
            xanchor="center",
            x=0.5,
            yref="container",
            yanchor="middle",
            y=legend_y,
        ),
        margin=dict(l=70, r=20, t=margin_top, b=40),
        barmode="overlay",
    )
    fig.update_annotations(font=dict(size=13))

    model_positions = {model: idx for idx, model in enumerate(MODELS)}
    tickvals = [model_positions[m] for m in MODELS]
    ticktext = [MODEL_LABELS[m] for m in MODELS]
    bar_width = 0.18
    group_spacing = bar_width
    feature_offsets = {
        feature: (idx - (len(FEATURES) - 1) / 2) * group_spacing
        for idx, feature in enumerate(FEATURES)
    }

    plotted: List[float] = []

    for feature in FEATURES:
        fig.add_trace(
            go.Bar(
                x=[None],
                y=[None],
                name=FEATURE_LABELS[feature],
                marker=dict(
                    color=FEATURE_COLORS[feature],
                    line=dict(color=FEATURE_COLORS[feature], width=1),
                ),
                opacity=0.85,
                showlegend=True,
                hoverinfo="skip",
            ),
            row=1,
            col=1,
        )

    for row_idx, eval_set in enumerate(EVAL_SETS, start=1):
        for col_idx, train_set in enumerate(TRAIN_SETS, start=1):
            fig.add_trace(
                go.Scatter(
                    x=[None],
                    y=[None],
                    mode="markers",
                    showlegend=False,
                    hoverinfo="skip",
                ),
                row=row_idx,
                col=col_idx,
            )
            for feature in FEATURES:
                feature_label = FEATURE_LABELS[feature]
                for model in MODELS:
                    values = results[train_set][eval_set][model].get(feature, [])
                    if not values:
                        continue

                    arr = metric_display_values(values)
                    plotted.extend(arr.tolist())
                    x_pos = model_positions[model] + feature_offsets[feature]
                    mean = float(np.mean(arr))
                    n = len(arr)
                    sem = (
                        float(np.std(arr, ddof=1) / np.sqrt(n)) if n > 1 else 0.0
                    )
                    if n > 1:
                        plotted.append(mean + sem)

                    bar_kwargs = dict(
                        x=[x_pos],
                        y=[mean],
                        name=feature_label,
                        marker=dict(
                            color=FEATURE_COLORS[feature],
                            line=dict(color=FEATURE_COLORS[feature], width=1),
                        ),
                        opacity=0.85,
                        width=bar_width,
                        showlegend=False,
                    )
                    if n > 1:
                        bar_kwargs["error_y"] = dict(
                            type="data",
                            array=[sem],
                            color="black",
                            thickness=1,
                            width=3,
                        )
                    fig.add_trace(go.Bar(**bar_kwargs), row=row_idx, col=col_idx)

                    if n > 1:
                        fig.add_trace(
                            go.Scatter(
                                x=[x_pos] * n,
                                y=arr.tolist(),
                                mode="markers",
                                name=feature_label,
                                marker=dict(color="black", size=3),
                                showlegend=False,
                            ),
                            row=row_idx,
                            col=col_idx,
                        )

            fig.update_xaxes(
                tickmode="array",
                tickvals=tickvals,
                ticktext=ticktext,
                tickangle=0,
                range=[-0.5, 2.5],
                row=row_idx,
                col=col_idx,
            )

    if plotted:
        y_max = max(plotted)
        y_range = [0.0, y_max * 1.05 if y_max > 0 else 1.0]
    else:
        y_range = [0.0, 50.0]

    for row_idx in range(1, len(EVAL_SETS) + 1):
        for col_idx in range(1, len(TRAIN_SETS) + 1):
            fig.update_yaxes(
                title_text="Recovery (%)" if col_idx == 1 else "",
                title_font=dict(weight="normal"),
                showticklabels=col_idx == 1,
                range=y_range,
                row=row_idx,
                col=col_idx,
            )

    fig.update_xaxes(
        showline=True, mirror=True, linewidth=1, linecolor="#333"
    )
    fig.update_yaxes(
        showline=True, mirror=True, linewidth=1, linecolor="#333"
    )

    return fig


def main() -> None:
    ensure_figures_dir()
    results, missing, n_total = collect_results()
    for eval_set, names in missing.items():
        if names:
            warnings.warn(
                f"Missing {SET_LABELS[eval_set]} Recovery for "
                f"{len(names)}/{n_total} exp2 runs; plotting available data only.",
                stacklevel=1,
            )

    export_source_data(results)

    fig = build_figure(results)

    base_path = os.path.join(FIGURES_DIR, "exp2_overview")
    fig.write_image(base_path + ".png", scale=3)
    fig.write_image(base_path + ".svg", scale=3)
    fig.write_image(base_path + ".pdf", scale=3)


if __name__ == "__main__":
    main()
