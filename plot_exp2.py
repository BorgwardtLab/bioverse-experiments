import csv
import glob
import os
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np
import plotly.graph_objects as go
import yaml
from plotly.subplots import make_subplots

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
FIGURES_DIR = os.path.join(os.path.dirname(__file__), "figures")


MODELS = ["gvp", "mpnn", "pifold"]
MODEL_LABELS = {"gvp": "GVP", "mpnn": "MPNN", "pifold": "PiFold"}

DATASETS = ["inv", "srf"]
DATASET_LABELS = {"inv": "Inverse Design (inv)", "srf": "Surface Design (srf)"}

DATASET_SUBPLOT_TITLES = {
    "inv": "CATH 4.2 Dataset",
    "srf": "Surface-only Dataset",
}

FEATURES = ["", "unified", "meso"]
FEATURE_LABELS = {
    "": "Original",
    "unified": "Unified",
    "meso": "Mesoscopic",
}

METRICS = ["Recovery", "Top-2 Recovery"]

# Three shades of green, ordered Original, Unified, Mesoscopic
FEATURE_COLORS = {
    "": "rgba(229, 245, 224, 1.0)",  # very light green
    "unified": "rgba(161, 217, 155, 1.0)",  # light green
    "meso": "rgba(49, 163, 84, 1.0)",  # darker green
}


def parse_experiment_name(exp_name: str) -> Tuple[str, str, str, int]:
    """
    Parse names like:
      exp2_gvp_inv_1
      exp2_gvp_unified_inv_2
      exp2_gvp_meso_srf_1
    into (model, feature, dataset, seed).
    """
    parts = exp_name.split("_")
    if not parts or parts[0] != "exp2":
        raise ValueError(f"Unexpected experiment name format: {exp_name}")

    # exp2_<model>_<dataset>_<seed>
    # exp2_<model>_unified_<dataset>_<seed>
    # exp2_<model>_meso_<dataset>_<seed>
    if len(parts) == 4:
        _, model, dataset, seed_str = parts
        feature = ""
    elif len(parts) == 5:
        _, model, feature, dataset, seed_str = parts
    else:
        raise ValueError(f"Unexpected experiment name format: {exp_name}")

    if model not in MODELS:
        raise ValueError(f"Unknown model '{model}' in experiment name '{exp_name}'")
    if dataset not in DATASETS:
        raise ValueError(f"Unknown dataset '{dataset}' in experiment name '{exp_name}'")
    if feature not in FEATURES:
        raise ValueError(f"Unknown feature '{feature}' in experiment name '{exp_name}'")

    seed = int(seed_str)
    return model, feature, dataset, seed


def collect_results_from_source_csv() -> Dict[str, Dict[str, Dict[str, Dict[str, List[float]]]]]:
    """
    Fallback loader for archived source data when raw experiment outputs
    are not available locally.
    """
    path = os.path.join(FIGURES_DIR, "source_data_plot2.csv")
    results: Dict[str, Dict[str, Dict[str, Dict[str, List[float]]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    )

    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            dataset = row["dataset"]
            metric = row["metric"]
            model = row["model"]
            feature = row["feature"]
            value = float(row["value_percent"]) / 100.0
            results[dataset][metric][model][feature].append(value)

    return results


def collect_results() -> Dict[str, Dict[str, Dict[str, Dict[str, List[float]]]]]:
    """
    Aggregate metric values:
    results[dataset][metric][model][feature] = list of values over seeds
    """
    results: Dict[str, Dict[str, Dict[str, Dict[str, List[float]]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    )

    pattern = os.path.join(RESULTS_DIR, "exp2_*", "val_results.yaml")
    for path in glob.glob(pattern):
        exp_dir = os.path.basename(os.path.dirname(path))
        try:
            model, feature, dataset, seed = parse_experiment_name(exp_dir)
        except ValueError:
            # Skip folders that do not match the expected exp2 naming
            continue

        with open(path, "r") as f:
            data = yaml.safe_load(f)

        if not isinstance(data, dict):
            continue

        for metric in METRICS:
            if metric in data:
                val = float(data[metric])
                results[dataset][metric][model][feature].append(val)

    return results


def ensure_figures_dir() -> None:
    os.makedirs(FIGURES_DIR, exist_ok=True)


def export_source_data(
    results: Dict[str, Dict[str, Dict[str, Dict[str, List[float]]]]],
) -> None:
    """
    Export the per-run metric values (converted to percent) used in exp2 plots
    to a long-form CSV for source data archiving.
    """
    out_path = os.path.join(FIGURES_DIR, "source_data_plot2.csv")
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["dataset", "metric", "model", "feature", "value_percent"])
        for dataset in DATASETS:
            for metric in METRICS:
                for model in MODELS:
                    for feature in FEATURES:
                        values = results[dataset][metric][model].get(feature, [])
                        for v in values:
                            writer.writerow(
                                [
                                    dataset,
                                    metric,
                                    model,
                                    feature,
                                    float(v) * 100.0,
                                ]
                            )


def build_figure(
    results: Dict[str, Dict[str, Dict[str, Dict[str, List[float]]]]],
) -> go.Figure:
    fig = make_subplots(
        rows=len(DATASETS),
        cols=len(METRICS),
        shared_yaxes=True,
        horizontal_spacing=0.08,
        vertical_spacing=0.16,
    )

    fig.update_layout(
        template="plotly_white",
        font=dict(
            family="DejaVu Sans Mono, Menlo, Monaco, Consolas, monospace", size=12
        ),
        legend=dict(
            title="Features",
            orientation="h",
            yanchor="bottom",
            y=1.05,
            xanchor="center",
            x=0.5,
        ),
        margin=dict(l=60, r=20, t=80, b=60),
    )

    show_legend_for_feature = {f: True for f in FEATURES}

    # Numeric x-positions for models so we can precisely align boxes and points
    model_positions = {model: idx for idx, model in enumerate(MODELS)}
    tickvals = [model_positions[m] for m in MODELS]
    ticktext = [MODEL_LABELS[m] for m in MODELS]
    # Horizontal offsets for each feature within a model group
    group_spacing = 0.22
    feature_offsets = {
        "": -group_spacing,
        "unified": 0.0,
        "meso": group_spacing,
    }
    # Keep boxes narrower than the spacing so grouped features do not overlap
    box_width = 0.16

    # Track global min/max (in percent) across all panels to set shared y-axis
    global_min_value = float("inf")
    global_max_value = float("-inf")

    for row_idx, dataset in enumerate(DATASETS, start=1):
        for col_idx, metric in enumerate(METRICS, start=1):
            for feature in FEATURES:
                feature_label = FEATURE_LABELS[feature]
                for model in MODELS:
                    values = results[dataset][metric][model].get(feature, [])
                    if not values:
                        continue

                    arr = np.asarray(values, dtype=float) * 100.0
                    local_min = float(arr.min())
                    local_max = float(arr.max())
                    if local_min < global_min_value:
                        global_min_value = local_min
                    if local_max > global_max_value:
                        global_max_value = local_max

                    x_pos = model_positions[model] + feature_offsets[feature]
                    fig.add_trace(
                        go.Box(
                            x=[x_pos] * len(arr),
                            y=arr.tolist(),
                            name=feature_label,
                            fillcolor=FEATURE_COLORS[feature],
                            line_color=FEATURE_COLORS[feature],
                            opacity=0.85,
                            width=box_width,
                            boxpoints=False,
                            showlegend=show_legend_for_feature[feature],
                        ),
                        row=row_idx,
                        col=col_idx,
                    )
                    fig.add_trace(
                        go.Scatter(
                            x=[x_pos] * len(arr),
                            y=arr.tolist(),
                            mode="markers",
                            name=feature_label,
                            marker=dict(
                                color="rgba(196, 164, 132, 1.0)", size=4
                            ),
                            showlegend=False,
                        ),
                        row=row_idx,
                        col=col_idx,
                    )

                    show_legend_for_feature[feature] = False

            # Axis styling for each subplot (shared y-range set after loop)
            if col_idx == 1:
                yaxis_title = "Recovery (%)"
            else:
                yaxis_title = "Top-2 Recovery (%)"

            fig.update_yaxes(
                title_text=yaxis_title,
                row=row_idx,
                col=col_idx,
            )
            if dataset == "inv":
                xaxis_title = "Model trained on CATH 4.2 Dataset"
            else:
                xaxis_title = "Model trained on Surface-only Dataset"

            fig.update_xaxes(
                title_text=xaxis_title,
                tickmode="array",
                tickvals=tickvals,
                ticktext=ticktext,
                row=row_idx,
                col=col_idx,
            )

    # After all data processed, set a shared y-range from global min-max with margin
    if global_max_value > global_min_value:
        data_range = global_max_value - global_min_value
        margin = data_range * 0.05
        ymin = global_min_value - margin
        ymax = global_max_value + margin
    else:
        ymin = global_min_value - 1.0
        ymax = global_max_value + 1.0
    fig.update_yaxes(range=[ymin, ymax])

    return fig


def main() -> None:
    ensure_figures_dir()
    results = collect_results()
    if not results:
        results = collect_results_from_source_csv()

    if not results:
        raise RuntimeError(
            f"No results found under {RESULTS_DIR} or {FIGURES_DIR}/source_data_plot2.csv"
        )

    export_source_data(results)

    fig = build_figure(results)

    base_path = os.path.join(FIGURES_DIR, "exp2_overview")
    # Scale 3x for raster export; vector formats ignore scale but we keep consistent API
    fig.write_image(base_path + ".png", scale=3)
    fig.write_image(base_path + ".svg", scale=3)
    fig.write_image(base_path + ".pdf", scale=3)


if __name__ == "__main__":
    main()
