import os
import csv
import tempfile
import argparse

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOGS_DIR = os.path.join(REPO_ROOT, "logs")
RESULTS_DIR = os.path.join(REPO_ROOT, "results")
TEMP_CACHE_ROOT = os.path.join(tempfile.gettempdir(), "mambareview-cache")
MPLCONFIGDIR = os.path.join(TEMP_CACHE_ROOT, "matplotlib")
XDG_CACHE_HOME = os.path.join(TEMP_CACHE_ROOT, "xdg")

os.makedirs(MPLCONFIGDIR, exist_ok=True)
os.makedirs(XDG_CACHE_HOME, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", MPLCONFIGDIR)
os.environ.setdefault("XDG_CACHE_HOME", XDG_CACHE_HOME)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

from model_configs import MODEL_DISPLAY_NAMES


def read_metrics(csv_path):
    rows = []
    if not os.path.exists(csv_path):
        return rows

    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def resolve_default_log(*candidates):
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
    return candidates[0]


def extract_series(rows, x_key, y_key, skip_na=True):
    xs = []
    ys = []
    for row in rows:
        x_val = row.get(x_key, "")
        y_val = row.get(y_key, "")
        if skip_na and (y_val == "N/A" or y_val.strip() == ""):
            continue
        try:
            xs.append(float(x_val))
            ys.append(float(y_val))
        except ValueError:
            continue
    return xs, ys


def split_on_reset(rows, x_key):
    segments = []
    current = []
    last_x = None

    for row in rows:
        x_val = row.get(x_key, "")
        try:
            x = float(x_val)
        except ValueError:
            continue

        if last_x is not None and x < last_x and current:
            segments.append(current)
            current = []

        current.append(row)
        last_x = x

    if current:
        segments.append(current)

    return segments


def select_latest_segment(rows, x_key):
    segments = split_on_reset(rows, x_key)
    if len(segments) <= 1:
        return rows

    latest = segments[-1]
    print(
        f"Detected {len(segments)} runs in log for x_key='{x_key}'; "
        f"plotting the latest segment with {len(latest)} rows."
    )
    return latest


def style_axes(ax):
    ax.set_facecolor('white')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_color('black')
    ax.spines['left'].set_color('black')
    ax.tick_params(colors='black')
    ax.grid(True, linestyle=':', color='gray', alpha=0.5)


def infer_label(log_path):
    stem = os.path.basename(log_path).replace("_metrics.csv", "")
    if stem in MODEL_DISPLAY_NAMES:
        return MODEL_DISPLAY_NAMES[stem]
    return stem


def plot_series(series_specs, x_key, y_key, xlabel, ylabel, title, out_name, x_formatter=None, skip_na=True):
    valid = False
    plt.figure(figsize=(10, 6), facecolor='white')
    ax = plt.axes()
    style_axes(ax)

    styles = ['-', '--', '-.', ':']
    for idx, spec in enumerate(series_specs):
        rows = read_metrics(spec["path"])
        rows = select_latest_segment(rows, x_key)
        xs, ys = extract_series(rows, x_key, y_key, skip_na=skip_na)
        if not xs:
            continue
        valid = True
        plt.plot(xs, ys, color='black', linestyle=styles[idx % len(styles)], linewidth=2, label=spec["label"])

    if not valid:
        print(f"No data found for {out_name}.")
        plt.close()
        return

    plt.xlabel(xlabel, fontsize=12, color='black', fontweight='bold')
    plt.ylabel(ylabel, fontsize=12, color='black', fontweight='bold')
    plt.title(title, fontsize=14, color='black', fontweight='bold')
    if x_formatter is not None:
        ax.xaxis.set_major_formatter(FuncFormatter(x_formatter))
    plt.legend(frameon=True, facecolor='white', edgecolor='black', fontsize=11)
    plt.tight_layout()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, out_name)
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Plot saved successfully to {out_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--logs",
        nargs="+",
        default=[
            os.path.join(LOGS_DIR, "mamba1_metrics.csv"),
            resolve_default_log(
                os.path.join(LOGS_DIR, "mamba2_lr05x_warm2x_metrics.csv"),
                os.path.join(LOGS_DIR, "mamba2_metrics.csv"),
            ),
        ],
    )
    parser.add_argument("--labels", nargs="+", default=None)
    args = parser.parse_args()

    series_specs = []
    for idx, log_path in enumerate(args.logs):
        label = args.labels[idx] if args.labels and idx < len(args.labels) else infer_label(log_path)
        series_specs.append({"path": log_path, "label": label})

    def format_millions(x, pos):
        return f'{x/1e6:.1f}M'

    plot_series(
        series_specs,
        x_key="tokens_seen",
        y_key="val_ppl",
        xlabel="Tokens Seen",
        ylabel="Validation Perplexity",
        title="TinyStories Validation Perplexity",
        out_name="ppl_comparison.png",
        x_formatter=format_millions,
        skip_na=True,
    )
    plot_series(
        series_specs,
        x_key="tokens_seen",
        y_key="train_loss",
        xlabel="Tokens Seen",
        ylabel="Training Loss",
        title="TinyStories Training Loss",
        out_name="train_loss_comparison.png",
        x_formatter=format_millions,
        skip_na=False,
    )
    plot_series(
        series_specs,
        x_key="elapsed_seconds",
        y_key="val_ppl",
        xlabel="Elapsed Time (seconds)",
        ylabel="Validation Perplexity",
        title="TinyStories Validation Perplexity vs Time",
        out_name="ppl_vs_time.png",
        skip_na=True,
    )

if __name__ == "__main__":
    main()
