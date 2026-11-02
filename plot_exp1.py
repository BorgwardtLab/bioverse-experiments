import csv
import glob
import os
from collections import defaultdict
from typing import DefaultDict, Dict, List, Optional, Tuple

import numpy as np
import plotly.graph_objects as go
import yaml
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

# Prediction tasks: eight subplots in a 2x4 grid.
# We identify tasks from the benchmark name used in exp1 (qm9, md17, cath42, ...).
# Known titles for current benchmarks; fall back to the raw name if unknown.
TASK_TITLES: Dict[str, str] = {
    "qm9": "QM9",
    "md17": "MD17",
    "cath42": "ProteinInvBench",
    "dock": "BioLiP",
    "gym": "ProteinGym",
    "tmalign": "TM-align",
    "enzyme": "Enzyme Class",
    "rna": "ARES",
}
# Preferred ordering for known tasks; any additional tasks are appended afterwards.
DEFAULT_TASK_ORDER: List[str] = [
    "qm9",
    "md17",
    "dock",
    "tmalign",
    "rna",
    "enzyme",
    "gym",
    "cath42",
]
GRID_ROWS = 2
GRID_COLS = 4
MAX_TASK_SLOTS = GRID_ROWS * GRID_COLS

# Expected primary metric per task (used for axis labels when results are missing).
TASK_EXPECTED_METRICS: Dict[str, str] = {
    "qm9": "MAE",
    "md17": "Angular MAE",
    "cath42": "Recovery",
    "dock": "MAE",
    "gym": "Spearman",
    "tmalign": "Spearman",
    "enzyme": "Accuracy",
    "rna": "Spearman",
}

# Subplot metric labels: metric name with units in parentheses (empty if dimensionless).
TASK_METRIC_LABELS: Dict[str, str] = {
    "qm9": "MAE (Bohr³)",
    "md17": "Angular MAE (°)",
    "tmalign": "Spearman (%)",
    "dock": "MAE (pKd)",
    "rna": "Spearman (%)",
    "enzyme": "Accuracy (%)",
    "gym": "Spearman (%)",
    "cath42": "Recovery (%)",
}

# Preferred metric when multiple are logged for the same run.
METRIC_PREFERENCE: List[str] = [
    "Angular MAE",
    "MAE",
    "MSE",
    "Spearman",
    "Accuracy",
    "Recovery",
    "Top-2 Recovery",
]

# Two shades for the two frameworks, in the spirit of the exp2 color palette
FRAMEWORK_COLORS: Dict[str, str] = {
    "torch": "rgba(161, 217, 155, 1.0)",  # light green
    "tensorflow": "rgba(49, 163, 84, 1.0)",  # darker green
}

PERCENT_METRICS = {"Recovery", "Accuracy", "Spearman"}

# Y-axis limits per task. Bounds are in the metric's native units (accuracy /
# recovery as fractions, MAE as reported). ``"min"`` / ``"max"`` are taken
# from the plotted values. Percent-displayed metrics are scaled to 0-100.
TASK_YAXIS_RANGE: Dict[str, Tuple[object, object]] = {
    "qm9": (0, "max"),
    "md17": (0, 90),
    "dock": (0, "max"),
    "tmalign": (0, 1),
    "rna": ("min", 1),
    "enzyme": (0, 1),
    "gym": (0, 1),
    "cath42": (0, 1),
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


def read_metric_from_yaml(
    path: str, task_name: Optional[str] = None
) -> Tuple[Optional[str], Optional[float]]:
    """Read the preferred test metric from test_results.yaml."""
    with open(path, "r") as f:
        data = yaml.safe_load(f)

    if not isinstance(data, dict) or not data:
        return None, None

    preference: List[str] = []
    if task_name and task_name in TASK_EXPECTED_METRICS:
        preference.append(TASK_EXPECTED_METRICS[task_name])
    preference.extend(m for m in METRIC_PREFERENCE if m not in preference)

    for metric_name in preference:
        if metric_name in data:
            try:
                return metric_name, float(data[metric_name])
            except (TypeError, ValueError):
                continue

    metric_name, raw_value = next(iter(data.items()))
    try:
        return str(metric_name), float(raw_value)
    except (TypeError, ValueError):
        return None, None


def read_metric_from_run_dir(
    run_dir: str, task_name: Optional[str] = None
) -> Tuple[Optional[str], Optional[float]]:
    yaml_path = os.path.join(run_dir, "test_results.yaml")
    if os.path.isfile(yaml_path):
        return read_metric_from_yaml(yaml_path, task_name=task_name)

    csv_path = os.path.join(run_dir, "test_results.csv")
    if os.path.isfile(csv_path):
        return read_metric_from_csv(csv_path)

    return None, None


def metric_display_values(metric_name: str, values: List[float]) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if metric_name in PERCENT_METRICS:
        return arr * 100.0
    return arr


def yaxis_range(
    task: str, metric_name: str, values: np.ndarray
) -> Optional[List[float]]:
    spec = TASK_YAXIS_RANGE.get(task)
    if spec is None:
        return None
    lo, hi = spec
    scale = 100.0 if metric_name in PERCENT_METRICS else 1.0
    has_values = values.size > 0

    def resolve(bound: object, which: str) -> Optional[float]:
        if bound == "min":
            if has_values:
                return float(np.min(values))
            return -1.0 if which == "lo" else None
        if bound == "max":
            if has_values:
                return float(np.max(values))
            return None
        return float(bound) * scale

    ymin = resolve(lo, "lo")
    ymax = resolve(hi, "hi")
    if ymin is None or ymax is None or ymin >= ymax:
        return None
    if hi == "max":
        ymax = ymax * 1.05
    return [ymin, ymax]


def yaxis_tickformat(metric_name: str) -> str:
    name = metric_name.lower()
    if metric_name in PERCENT_METRICS or "angular" in name:
        return ".0f"
    return ".1f"


def metric_direction_symbol(metric_name: Optional[str]) -> str:
    """Return ↓ if lower is better, ↑ if higher is better."""
    if not metric_name:
        return "?"

    name = metric_name.lower()

    if any(
        key in name
        for key in ["mae", "mse", "rmse", "loss", "error", "nll", "l1", "l2"]
    ):
        return "↓"

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
            "spearman",
            "rho",
        ]
    ):
        return "↑"

    return "?"


def metric_label(task: Optional[str], metric_names: Dict[str, str]) -> str:
    metric_name = ""
    if task:
        metric_name = metric_names.get(task, "") or TASK_EXPECTED_METRICS.get(task, "")

    if task and task in TASK_METRIC_LABELS:
        base_label = TASK_METRIC_LABELS[task]
    elif metric_name:
        if metric_name in PERCENT_METRICS:
            base_label = f"{metric_name} (%)"
        else:
            base_label = f"{metric_name} ()"
    else:
        return "Metric"

    direction = metric_direction_symbol(metric_name)
    if direction == "?":
        return base_label
    return f"{base_label} {direction}"


def subplot_title(task: Optional[str], metric_names: Dict[str, str]) -> str:
    if task is None:
        return ""
    title = TASK_TITLES.get(task, task.upper())
    metric = metric_label(task, metric_names)
    return f"<b>{title}</b><br>{metric}"


def configure_empty_subplot(
    fig: go.Figure,
    *,
    row_idx: int,
    col_idx: int,
    tickvals: List[int],
    ticktext: List[str],
    y_range: Optional[List[float]] = None,
    metric_name: str = "",
) -> None:
    fig.update_xaxes(
        tickmode="array",
        tickvals=tickvals,
        ticktext=ticktext,
        tickangle=0,
        row=row_idx,
        col=col_idx,
    )
    yaxis_kwargs = {"tickformat": yaxis_tickformat(metric_name)}
    if y_range is not None:
        yaxis_kwargs["range"] = y_range
        yaxis_kwargs["autorange"] = False
    fig.update_yaxes(row=row_idx, col=col_idx, **yaxis_kwargs)


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

    pattern = os.path.join(RESULTS_DIR, "exp1_*")
    for run_dir in sorted(glob.glob(pattern)):
        if not os.path.isdir(run_dir):
            continue

        exp_dir = os.path.basename(run_dir)
        try:
            framework, repr_name, task_name, _seed = parse_experiment_name(exp_dir)
        except ValueError:
            # Skip folders that do not match the expected exp1 naming
            continue

        metric_name, value = read_metric_from_run_dir(run_dir, task_name=task_name)
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
        for task in DEFAULT_TASK_ORDER:
            if task not in results:
                continue
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
    # Always reserve fixed subplot slots for all planned exp1 tasks.
    ordered_tasks = DEFAULT_TASK_ORDER[:MAX_TASK_SLOTS]
    task_slots: List[Optional[str]] = list(ordered_tasks)
    while len(task_slots) < MAX_TASK_SLOTS:
        task_slots.append(None)

    subplot_titles = [
        subplot_title(task, metric_names) for task in task_slots
    ]

    fig_height = 800
    margin_top = 100
    margin_bottom = 35
    legend_y = 1.0 - (margin_top / fig_height) / 2

    fig = make_subplots(
        rows=GRID_ROWS,
        cols=GRID_COLS,
        shared_yaxes=False,
        horizontal_spacing=0.055,
        vertical_spacing=0.18,
        subplot_titles=subplot_titles,
    )

    # Configure layout (matching the style of plot_exp2.py)
    fig.update_layout(
        template="plotly_white",
        width=1600,
        height=fig_height,
        font=dict(
            family="DejaVu Sans Mono, Menlo, Monaco, Consolas, monospace",
            size=11,
        ),
        legend=dict(
            title="Framework",
            orientation="h",
            xref="container",
            xanchor="center",
            x=0.5,
            yref="container",
            yanchor="middle",
            y=legend_y,
        ),
        margin=dict(l=50, r=30, t=margin_top, b=margin_bottom),
        barmode="overlay",
    )
    fig.update_annotations(font=dict(size=12))

    # Numeric x-positions for representations so we can align bars and points
    repr_positions: Dict[str, int] = {
        repr_name: idx for idx, repr_name in enumerate(REPRS)
    }
    tickvals = [repr_positions[r] for r in REPRS]
    ticktext = [REPR_LABELS[r] for r in REPRS]
    if len(FRAMEWORKS) > 1:
        bar_width = 0.18 * 5 / 3
        group_spacing = bar_width
        framework_offsets = {
            framework: (idx - (len(FRAMEWORKS) - 1) / 2) * group_spacing
            for idx, framework in enumerate(FRAMEWORKS)
        }
    else:
        framework_offsets = {framework: 0.0 for framework in FRAMEWORKS}
        bar_width = 0.35 * 5 / 3

    # Only show each legend entry once for the whole figure
    show_legend_for_framework: Dict[str, bool] = {fw: True for fw in FRAMEWORKS}

    # Fill each task slot (some may be empty if not yet run)
    for slot_idx, task in enumerate(task_slots):
        row_idx = slot_idx // GRID_COLS + 1
        col_idx = slot_idx % GRID_COLS + 1

        task_metric = (
            metric_names.get(task, TASK_EXPECTED_METRICS.get(task, ""))
            if task is not None
            else ""
        )
        plotted: List[float] = []

        if task is None:
            configure_empty_subplot(
                fig,
                row_idx=row_idx,
                col_idx=col_idx,
                tickvals=tickvals,
                ticktext=ticktext,
            )
            continue

        if task not in results:
            configure_empty_subplot(
                fig,
                row_idx=row_idx,
                col_idx=col_idx,
                tickvals=tickvals,
                ticktext=ticktext,
                y_range=yaxis_range(task, task_metric, np.asarray([])),
                metric_name=task_metric,
            )
            continue

        for framework in FRAMEWORKS:
            for repr_name in REPRS:
                values = results[task][repr_name].get(framework, [])
                if not values:
                    continue

                arr = metric_display_values(task_metric, values)
                plotted.extend(arr.tolist())
                x_pos = repr_positions[repr_name] + framework_offsets[framework]
                mean = float(np.mean(arr))
                sem = (
                    float(np.std(arr, ddof=1) / np.sqrt(len(arr)))
                    if len(arr) > 1
                    else 0.0
                )
                plotted.append(mean + sem)

                fig.add_trace(
                    go.Bar(
                        x=[x_pos],
                        y=[mean],
                        name=FRAMEWORK_LABELS[framework],
                        marker=dict(
                            color=FRAMEWORK_COLORS[framework],
                            line=dict(color=FRAMEWORK_COLORS[framework], width=1),
                        ),
                        opacity=0.85,
                        width=bar_width,
                        error_y=dict(
                            type="data",
                            array=[sem],
                            color="black",
                            thickness=1,
                            width=3,
                        ),
                        showlegend=show_legend_for_framework[framework],
                    ),
                    row=row_idx,
                    col=col_idx,
                )
                fig.add_trace(
                    go.Scatter(
                        x=[x_pos] * len(arr),
                        y=arr.tolist(),
                        mode="markers",
                        name=FRAMEWORK_LABELS[framework],
                        marker=dict(color="black", size=3),
                        showlegend=False,
                    ),
                    row=row_idx,
                    col=col_idx,
                )

                show_legend_for_framework[framework] = False

        y_range = yaxis_range(task, task_metric, np.asarray(plotted))
        yaxis_kwargs = {"tickformat": yaxis_tickformat(task_metric)}
        if y_range is not None:
            yaxis_kwargs["range"] = y_range
            yaxis_kwargs["autorange"] = False
        fig.update_yaxes(row=row_idx, col=col_idx, **yaxis_kwargs)
        fig.update_xaxes(
            tickmode="array",
            tickvals=tickvals,
            ticktext=ticktext,
            tickangle=0,
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
