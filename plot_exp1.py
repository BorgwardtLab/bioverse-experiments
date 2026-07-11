import csv
import glob
import os
from collections import defaultdict
from typing import DefaultDict, Dict, List, Optional, Tuple

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
FIGURES_DIR = os.path.join(os.path.dirname(__file__), "figures")


# Frameworks and display labels
FRAMEWORKS = ["torch", "tensorflow"]
FRAMEWORK_LABELS: Dict[str, str] = {
    "torch": "PyTorch",
    "tensorflow": "TensorFlow",
}

# Representations and display labels
REPRS = ["voxel", "graph", "point"]
REPR_LABELS: Dict[str, str] = {
    "voxel": "Voxel",
    "graph": "Graph",
    "point": "Point",
}

# Prediction tasks: up to six subplots in a 2x3 grid.
# We identify tasks from the benchmark name used in exp1 (qm9, md17, cath42, ...).
# Known titles for current benchmarks; fall back to the raw name if unknown.
TASK_TITLES: Dict[str, str] = {
    "qm9": "QM9",
    "md17": "MD17",
    "cath42": "ProteinInvBench",
}
# Preferred ordering for known tasks; any additional tasks are appended afterwards.
DEFAULT_TASK_ORDER: List[str] = ["qm9", "md17", "cath42"]
MAX_TASK_SLOTS = 6

# Two shades for the two frameworks, in the spirit of the exp2 color palette
FRAMEWORK_COLORS: Dict[str, str] = {
    "torch": "rgba(161, 217, 155, 1.0)",  # light green
    "tensorflow": "rgba(49, 163, 84, 1.0)",  # darker green
}


def ensure_figures_dir() -> None:
    os.makedirs(FIGURES_DIR, exist_ok=True)


def parse_experiment_name(exp_name: str) -> Tuple[str, str, str, int]:
    """
    Parse names like:
      exp1_torch_graph_qm9_1
      exp1_tensorflow_point_md17_2
      exp1_torch_voxel_cath42_1
    into (framework, repr, task, seed).
    """
    parts = exp_name.split("_")
    if not parts or parts[0] != "exp1":
        raise ValueError(f"Unexpected experiment name format: {exp_name}")

    if len(parts) != 5:
        raise ValueError(f"Unexpected experiment name format: {exp_name}")

    _, framework, repr_name, task_name, seed_str = parts

    if framework not in FRAMEWORKS:
        raise ValueError(f"Unknown framework '{framework}' in experiment '{exp_name}'")
    if repr_name not in REPRS:
        raise ValueError(
            f"Unknown representation '{repr_name}' in experiment '{exp_name}'"
        )

    seed = int(seed_str)
    return framework, repr_name, task_name, seed


def read_metric_from_csv(path: str) -> Tuple[Optional[str], Optional[float]]:
    """
    Read the default metric (first column in test_results.csv) from the last
    available row. Returns (metric_name, value) where value is None if no
    valid number could be read.
    """
    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            return None, None
        metric_field = reader.fieldnames[0]

        last_value: Optional[float] = None
        for row in reader:
            raw = row.get(metric_field, "")
            if raw is None or raw == "":
                continue
            try:
                last_value = float(raw)
            except ValueError:
                continue

    return metric_field, last_value


def metric_direction_symbol(metric_name: Optional[str]) -> str:
    """
    Heuristic to decide whether higher or lower is better for a given metric name.
    Returns an arrow symbol.
    """
    if not metric_name:
        return "?"

    name = metric_name.lower()

    # Lower is better
    if any(
        key in name
        for key in ["mae", "mse", "rmse", "loss", "error", "nll", "l1", "l2"]
    ):
        return "↓"

    # Higher is better
    if any(
        key in name
        for key in [
            "recovery",
            "accuracy",
            "acc",
            "auc",
            "f1",
            "precision",
            "recall",
            "r2",
        ]
    ):
        return "↑"

    return "?"


def collect_results() -> Tuple[
    Dict[str, Dict[str, Dict[str, List[float]]]],
    Dict[str, str],
]:
    """
    Aggregate test metric values:

    results[task][repr][framework] = list of metric values over seeds
    metric_names[task] = name of the metric (first column of the CSV)
    """
    results: Dict[str, Dict[str, Dict[str, List[float]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(list))
    )
    metric_names: Dict[str, str] = {}

    pattern = os.path.join(RESULTS_DIR, "exp1_*", "test_results.csv")
    for path in glob.glob(pattern):
        exp_dir = os.path.basename(os.path.dirname(path))
        try:
            framework, repr_name, task_name, _seed = parse_experiment_name(exp_dir)
        except ValueError:
            # Skip folders that do not match the expected exp1 naming
            continue

        metric_name, value = read_metric_from_csv(path)
        if metric_name is None or value is None:
            continue

        results[task_name][repr_name][framework].append(value)
        # Record the metric name used for this task (assumes consistency per task)
        metric_names.setdefault(task_name, metric_name)

    return results, metric_names


def export_source_data(
    results: Dict[str, Dict[str, Dict[str, List[float]]]],
    metric_names: Dict[str, str],
) -> None:
    """
    Export the per-run metric values used in exp1 plots to a long-form CSV
    for source data archiving.
    """
    out_path = os.path.join(FIGURES_DIR, "source_data_plot1.csv")
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "task",
                "framework",
                "representation",
                "metric",
                "value",
            ]
        )
        for task in sorted(results.keys()):
            metric = metric_names.get(task, "")
            for repr_name in REPRS:
                for framework in FRAMEWORKS:
                    values = results[task][repr_name].get(framework, [])
                    for v in values:
                        writer.writerow(
                            [
                                task,
                                framework,
                                repr_name,
                                metric,
                                float(v),
                            ]
                        )


def build_figure(
    results: Dict[str, Dict[str, Dict[str, List[float]]]],
    metric_names: Dict[str, str],
) -> go.Figure:
    # Determine which tasks are present and how to place them into the 2x3 grid
    tasks_present = list(results.keys())

    # First place known tasks in the preferred order, then any additional tasks
    ordered_tasks: List[str] = []
    for t in DEFAULT_TASK_ORDER:
        if t in tasks_present:
            ordered_tasks.append(t)
    for t in sorted(tasks_present):
        if t not in ordered_tasks:
            ordered_tasks.append(t)

    # Truncate to available data and then pad with empty slots to 6 total
    ordered_tasks = ordered_tasks[:MAX_TASK_SLOTS]
    task_slots: List[Optional[str]] = list(ordered_tasks)
    while len(task_slots) < MAX_TASK_SLOTS:
        task_slots.append(None)

    subplot_titles = [
        TASK_TITLES.get(task, task.upper()) if task is not None else ""
        for task in task_slots
    ]

    fig = make_subplots(
        rows=2,
        cols=3,
        shared_yaxes=False,
        horizontal_spacing=0.08,
        vertical_spacing=0.16,
        subplot_titles=subplot_titles,
    )

    # Configure bar mode and layout (matching the style of plot_exp2.py)
    fig.update_layout(
        barmode="group",
        template="plotly_white",
        font=dict(
            family="DejaVu Sans Mono, Menlo, Monaco, Consolas, monospace",
            size=12,
        ),
        legend=dict(
            title="Framework",
            orientation="h",
            yanchor="bottom",
            y=1.05,
            xanchor="center",
            x=0.5,
        ),
        margin=dict(l=60, r=20, t=80, b=60),
    )

    # Numeric x-positions for representations so we can align bars and points
    repr_positions: Dict[str, int] = {
        repr_name: idx for idx, repr_name in enumerate(REPRS)
    }
    tickvals = [repr_positions[r] for r in REPRS]
    ticktext = [REPR_LABELS[r] for r in REPRS]

    # Horizontal offsets for each framework within a representation group
    bar_half_width = 0.25
    framework_offsets: Dict[str, float] = {
        "torch": -bar_half_width / 2.0,
        "tensorflow": bar_half_width / 2.0,
    }

    # Only show each legend entry once for the whole figure
    show_legend_for_framework: Dict[str, bool] = {fw: True for fw in FRAMEWORKS}

    # Fill each task slot (some may be empty if not yet run)
    for slot_idx, task in enumerate(task_slots):
        row_idx = slot_idx // 3 + 1
        col_idx = slot_idx % 3 + 1

        if task is None or task not in results:
            # Leave subplot blank (no data yet)
            # Still configure axes for consistent appearance
            fig.update_xaxes(
                title_text="Representation",
                tickmode="array",
                tickvals=tickvals,
                ticktext=ticktext,
                row=row_idx,
                col=col_idx,
            )
            fig.update_yaxes(
                title_text="Metric",
                row=row_idx,
                col=col_idx,
            )
            continue

        metric_name = metric_names.get(task, "")
        direction_symbol = metric_direction_symbol(metric_name)
        yaxis_title = metric_name if metric_name else "Metric"
        if direction_symbol != "?":
            yaxis_title = f"{yaxis_title} {direction_symbol}"

        # Add bars and overlaid scatter points for each framework
        for framework in FRAMEWORKS:
            means: List[float] = []
            stds: List[float] = []
            values_by_repr: List[Tuple[str, np.ndarray]] = []

            for repr_name in REPRS:
                values = results[task][repr_name].get(framework, [])
                if values:
                    arr = np.asarray(values, dtype=float)
                    means.append(float(arr.mean()))
                    stds.append(float(arr.std(ddof=1)) if arr.size > 1 else 0.0)
                    values_by_repr.append((repr_name, arr))
                else:
                    means.append(np.nan)
                    stds.append(0.0)
                    values_by_repr.append((repr_name, np.asarray([], dtype=float)))

            # x-positions for bars for this framework across representations
            x_bar = [
                repr_positions[repr_name] + framework_offsets[framework]
                for repr_name in REPRS
            ]

            fig.add_trace(
                go.Bar(
                    x=x_bar,
                    y=means,
                    error_y=dict(
                        type="data",
                        array=stds,
                        visible=True,
                    ),
                    name=FRAMEWORK_LABELS[framework],
                    marker_color=FRAMEWORK_COLORS[framework],
                    showlegend=show_legend_for_framework[framework],
                ),
                row=row_idx,
                col=col_idx,
            )

            # Overlay strip/point plot of individual data points (per seed)
            xs: List[float] = []
            ys: List[float] = []
            for repr_name, arr in values_by_repr:
                if arr.size == 0:
                    continue
                x_pos = repr_positions[repr_name] + framework_offsets[framework]
                xs.extend([x_pos] * len(arr))
                ys.extend(arr.tolist())

            if xs:
                fig.add_trace(
                    go.Scatter(
                        x=xs,
                        y=ys,
                        mode="markers",
                        name=FRAMEWORK_LABELS[framework],
                        marker=dict(color="rgba(196, 164, 132, 1.0)", size=4),
                        showlegend=False,
                    ),
                    row=row_idx,
                    col=col_idx,
                )

            # Legend entry already added once
            show_legend_for_framework[framework] = False

        # Axis styling for this subplot
        fig.update_yaxes(
            title_text=yaxis_title,
            row=row_idx,
            col=col_idx,
        )
        fig.update_xaxes(
            title_text="Representation",
            tickmode="array",
            tickvals=tickvals,
            ticktext=ticktext,
            row=row_idx,
            col=col_idx,
        )

    return fig


def main() -> None:
    ensure_figures_dir()
    results, metric_names = collect_results()

    if not results:
        raise RuntimeError(f"No exp1 results found under {RESULTS_DIR}")

    export_source_data(results, metric_names)

    fig = build_figure(results, metric_names)

    base_path = os.path.join(FIGURES_DIR, "exp1_overview")
    # Scale 3x for raster export; vector formats ignore scale but we keep consistent API
    fig.write_image(base_path + ".png", scale=3)
    fig.write_image(base_path + ".svg", scale=3)
    fig.write_image(base_path + ".pdf", scale=3)


if __name__ == "__main__":
    main()
