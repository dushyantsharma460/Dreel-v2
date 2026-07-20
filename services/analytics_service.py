"""
Analytics Service

Loads project datasets and builds dashboard summaries
and Matplotlib chart images for the /analytics page.
"""

import os

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import patheffects as pe
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import MaxNLocator

from config import DATA_FOLDER, STATIC_FOLDER

AUDIO_FEATURES_CSV = os.path.join(DATA_FOLDER, "audio_features.csv")
IMAGE_FEATURES_CSV = os.path.join(DATA_FOLDER, "image_features.csv")
CHARTS_DIR = os.path.join(STATIC_FOLDER, "reports", "charts")

CHART_BG = "#0f0c29"
CHART_PANEL = "#1a1538"
CHART_TEXT = "#e8e8f0"
CHART_MUTED = "#9aa0b8"
CHART_GRID = "#2f2a52"
ACCENT_BLUE = "#2575fc"
ACCENT_PURPLE = "#6a11cb"
ACCENT_GREEN = "#00d4aa"
ACCENT_GOLD = "#ffc107"
ACCENT_RED = "#ff4d4d"

QUALITY_COLORS = {
    "good": ACCENT_GREEN,
    "fair": ACCENT_GOLD,
    "poor": ACCENT_RED,
}

BAR_CMAP = LinearSegmentedColormap.from_list(
    "dreel_bar",
    [ACCENT_PURPLE, ACCENT_BLUE],
)
QUALITY_CMAP = LinearSegmentedColormap.from_list(
    "dreel_quality",
    [ACCENT_RED, ACCENT_GOLD, ACCENT_GREEN],
)


def _configure_matplotlib() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "axes.titlesize": 15,
        "axes.titleweight": "bold",
        "axes.labelsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "figure.facecolor": CHART_BG,
        "axes.facecolor": CHART_PANEL,
        "savefig.facecolor": CHART_BG,
        "text.color": CHART_TEXT,
    })


def _load_csv(path: str) -> pd.DataFrame:
    if not os.path.isfile(path):
        return pd.DataFrame()

    return pd.read_csv(path)


def load_audio_features() -> pd.DataFrame:
    dataframe = _load_csv(AUDIO_FEATURES_CSV)

    if dataframe.empty:
        return dataframe

    dataframe["timestamp"] = pd.to_datetime(dataframe["timestamp"], utc=True)
    return dataframe.drop_duplicates(subset=["reel_id"], keep="last")


def load_image_features() -> pd.DataFrame:
    dataframe = _load_csv(IMAGE_FEATURES_CSV)

    if dataframe.empty:
        return dataframe

    dataframe["timestamp"] = pd.to_datetime(dataframe["timestamp"], utc=True)
    return dataframe.drop_duplicates(subset=["reel_id", "filename"], keep="last")


def _new_figure(wide: bool = False):
    width = 11.5 if wide else 6.2
    height = 4.8 if wide else 5.0
    figure, axis = plt.subplots(figsize=(width, height))
    figure.patch.set_facecolor(CHART_BG)
    axis.set_facecolor(CHART_PANEL)
    return figure, axis


def _style_axes(axis, title: str, xlabel: str = "", ylabel: str = "") -> None:
    axis.set_title(title, color="#ffffff", pad=16, loc="left", fontsize=15, fontweight="bold")
    axis.set_xlabel(xlabel, color=CHART_MUTED, labelpad=10)
    axis.set_ylabel(ylabel, color=CHART_MUTED, labelpad=10)
    axis.tick_params(colors=CHART_TEXT, length=0, pad=8)
    axis.grid(True, axis="y", alpha=0.22, color=CHART_GRID, linestyle="-", linewidth=0.8)
    axis.grid(False, axis="x")
    for spine in ("top", "right"):
        axis.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        axis.spines[spine].set_color(CHART_GRID)
        axis.spines[spine].set_linewidth(1.0)


def _bar_colors(count: int, cmap=BAR_CMAP) -> list:
    if count <= 1:
        return [ACCENT_BLUE]
    return [cmap(value) for value in np.linspace(0.25, 0.95, count)]


def _annotate_bars(axis, bars, fmt="{:.1f}") -> None:
    for bar in bars:
        height = bar.get_height()
        if height <= 0:
            continue
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            height + max(height * 0.03, 0.04),
            fmt.format(height),
            ha="center",
            va="bottom",
            color="#ffffff",
            fontsize=9,
            fontweight="bold",
        )


def _save_chart(figure, filename: str) -> str:
    os.makedirs(CHARTS_DIR, exist_ok=True)
    output_path = os.path.join(CHARTS_DIR, filename)
    figure.savefig(
        output_path,
        dpi=160,
        bbox_inches="tight",
        facecolor=CHART_BG,
        edgecolor="none",
        pad_inches=0.35,
    )
    plt.close(figure)
    return f"reports/charts/{filename}"


def build_summary(audio_df: pd.DataFrame, image_df: pd.DataFrame) -> dict:
    return {
        "total_reels": int(len(audio_df)),
        "total_images": int(len(image_df)),
        "avg_tempo": round(float(audio_df["tempo_bpm"].mean()), 1) if not audio_df.empty else 0,
        "avg_quality": round(float(image_df["quality_score"].mean()), 3) if not image_df.empty else 0,
        "avg_duration": round(float(audio_df["duration_sec"].mean()), 1) if not audio_df.empty else 0,
        "avg_slide_duration": round(
            float(audio_df["avg_slide_duration"].mean()),
            2,
        ) if not audio_df.empty else 0,
        "low_quality_images": int((image_df["quality_label"] == "poor").sum()) if not image_df.empty else 0,
    }


def build_recent_reels(audio_df: pd.DataFrame, image_df: pd.DataFrame) -> list[dict]:
    if audio_df.empty:
        return []

    quality_by_reel = (
        image_df.groupby("reel_id")["quality_score"].mean().round(3)
        if not image_df.empty else pd.Series(dtype=float)
    )

    rows = []

    for _, row in audio_df.sort_values("timestamp", ascending=False).iterrows():
        reel_id = row["reel_id"]
        rows.append({
            "reel_id": reel_id,
            "short_id": reel_id[:8],
            "timestamp": row["timestamp"].strftime("%Y-%m-%d %H:%M"),
            "image_count": int(row["image_count"]),
            "tempo_bpm": round(float(row["tempo_bpm"]), 1),
            "duration_sec": round(float(row["duration_sec"]), 1),
            "avg_quality": float(quality_by_reel.get(reel_id, 0)),
        })

    return rows


def _chart_tempo(audio_df: pd.DataFrame) -> dict:
    figure, axis = _new_figure()
    values = audio_df["tempo_bpm"]
    bin_count = max(4, min(8, len(values)))
    counts, bins, patches = axis.hist(
        values,
        bins=bin_count,
        color=ACCENT_BLUE,
        edgecolor=CHART_BG,
        linewidth=1.4,
        alpha=0.92,
        rwidth=0.88,
    )

    for patch, color in zip(patches, _bar_colors(len(patches))):
        patch.set_facecolor(color)

    mean_tempo = float(values.mean())
    axis.axvline(
        mean_tempo,
        color=ACCENT_RED,
        linestyle="--",
        linewidth=2.2,
        label=f"Mean: {mean_tempo:.0f} BPM",
    )
    axis.legend(
        frameon=True,
        facecolor=CHART_PANEL,
        edgecolor=CHART_GRID,
        labelcolor=CHART_TEXT,
        loc="upper right",
    )
    _style_axes(axis, "Audio Tempo Distribution", "Tempo (BPM)", "Number of Reels")
    axis.yaxis.set_major_locator(MaxNLocator(integer=True))

    return {
        "title": "Audio Tempo Distribution",
        "image": _save_chart(figure, "tempo_distribution.png"),
        "wide": False,
    }


def _chart_timeline(audio_df: pd.DataFrame) -> dict:
    figure, axis = _new_figure(wide=True)
    timeline_df = audio_df.copy()
    timeline_df["date"] = timeline_df["timestamp"].dt.date.astype(str)
    reels_over_time = (
        timeline_df.groupby("date", as_index=False)
        .size()
        .rename(columns={"size": "reel_count"})
    )

    x_positions = np.arange(len(reels_over_time))
    bars = axis.bar(
        x_positions,
        reels_over_time["reel_count"],
        color=_bar_colors(len(reels_over_time)),
        edgecolor=CHART_BG,
        linewidth=1.2,
        width=0.62,
        zorder=3,
    )
    axis.plot(
        x_positions,
        reels_over_time["reel_count"],
        color=ACCENT_GREEN,
        marker="o",
        markersize=8,
        linewidth=2.2,
        zorder=4,
    )
    axis.set_xticks(x_positions)
    axis.set_xticklabels(reels_over_time["date"], rotation=0)
    _annotate_bars(axis, bars, fmt="{:.0f}")
    _style_axes(axis, "Reels Created Over Time", "Date", "Reels Produced")
    axis.yaxis.set_major_locator(MaxNLocator(integer=True))

    return {
        "title": "Reels Over Time",
        "image": _save_chart(figure, "reels_over_time.png"),
        "wide": True,
    }


def _chart_slide_duration(audio_df: pd.DataFrame) -> dict:
    figure, axis = _new_figure()
    slide_df = audio_df.assign(short_id=audio_df["reel_id"].str[:8]).sort_values(
        "avg_slide_duration",
        ascending=True,
    )
    y_positions = np.arange(len(slide_df))
    colors = _bar_colors(len(slide_df))

    bars = axis.barh(
        y_positions,
        slide_df["avg_slide_duration"],
        color=colors,
        edgecolor=CHART_BG,
        linewidth=1.2,
        height=0.62,
    )
    axis.set_yticks(y_positions)
    axis.set_yticklabels(slide_df["short_id"])
    for bar in bars:
        width = bar.get_width()
        axis.text(
            width + 0.08,
            bar.get_y() + bar.get_height() / 2,
            f"{width:.2f}s",
            va="center",
            ha="left",
            color="#ffffff",
            fontsize=9,
            fontweight="bold",
        )
    _style_axes(axis, "Beat-Sync Slide Duration", "Seconds per Image", "Reel ID")
    axis.grid(True, axis="x", alpha=0.22, color=CHART_GRID)
    axis.grid(False, axis="y")

    return {
        "title": "Slide Duration",
        "image": _save_chart(figure, "slide_duration.png"),
        "wide": False,
    }


def _chart_quality_labels(image_df: pd.DataFrame) -> dict:
    figure, axis = _new_figure()
    label_counts = image_df["quality_label"].value_counts()
    labels = [label.title() for label in label_counts.index]
    colors = [QUALITY_COLORS.get(label, ACCENT_BLUE) for label in label_counts.index]

    wedges, texts, autotexts = axis.pie(
        label_counts.values,
        labels=labels,
        autopct="%1.0f%%",
        startangle=110,
        colors=colors,
        pctdistance=0.78,
        explode=[0.03] * len(label_counts),
        wedgeprops={"width": 0.42, "edgecolor": CHART_BG, "linewidth": 2.5},
        textprops={"color": CHART_TEXT, "fontsize": 11, "fontweight": "bold"},
    )

    for autotext in autotexts:
        autotext.set_color("#ffffff")
        autotext.set_fontsize(10)
        autotext.set_fontweight("bold")
        autotext.set_path_effects([pe.withStroke(linewidth=2, foreground=CHART_BG)])

    axis.set_title(
        "Image Quality Breakdown",
        color="#ffffff",
        pad=16,
        loc="left",
        fontsize=15,
        fontweight="bold",
    )
    centre_circle = plt.Circle((0, 0), 0.28, fc=CHART_PANEL, ec=CHART_GRID, linewidth=1.2)
    axis.add_artist(centre_circle)
    axis.text(
        0,
        0.04,
        f"{len(image_df)}",
        ha="center",
        va="center",
        color="#ffffff",
        fontsize=22,
        fontweight="bold",
    )
    axis.text(0, -0.14, "images", ha="center", va="center", color=CHART_MUTED, fontsize=10)

    return {
        "title": "Image Quality Labels",
        "image": _save_chart(figure, "quality_labels.png"),
        "wide": False,
    }


def _chart_blur_quality(image_df: pd.DataFrame) -> dict:
    figure, axis = _new_figure()

    for label in ("good", "fair", "poor"):
        subset = image_df[image_df["quality_label"] == label]
        if subset.empty:
            continue
        axis.scatter(
            subset["blur_variance"],
            subset["quality_score"],
            label=label.title(),
            color=QUALITY_COLORS[label],
            alpha=0.9,
            s=95,
            edgecolors="#ffffff",
            linewidths=0.8,
            zorder=3,
        )

    if len(image_df) >= 3:
        coefficients = np.polyfit(image_df["blur_variance"], image_df["quality_score"], 1)
        x_line = np.linspace(image_df["blur_variance"].min(), image_df["blur_variance"].max(), 100)
        y_line = coefficients[0] * x_line + coefficients[1]
        axis.plot(x_line, y_line, color=ACCENT_GOLD, linestyle="--", linewidth=2, alpha=0.9, label="Trend")

    axis.legend(
        frameon=True,
        facecolor=CHART_PANEL,
        edgecolor=CHART_GRID,
        labelcolor=CHART_TEXT,
        loc="lower right",
    )
    _style_axes(
        axis,
        "Sharpness vs Quality Score",
        "Blur Variance (higher = sharper)",
        "Combined Quality Score",
    )

    return {
        "title": "Blur vs Quality",
        "image": _save_chart(figure, "blur_quality.png"),
        "wide": False,
    }


def _chart_reel_quality(image_df: pd.DataFrame) -> dict:
    figure, axis = _new_figure()
    reel_quality = (
        image_df.groupby("reel_id", as_index=False)["quality_score"]
        .mean()
        .assign(short_id=lambda df: df["reel_id"].str[:8])
        .sort_values("quality_score", ascending=False)
    )

    x_positions = np.arange(len(reel_quality))
    normalized = (reel_quality["quality_score"] - reel_quality["quality_score"].min())
    denominator = max(normalized.max(), 0.001)
    colors = [QUALITY_CMAP(0.35 + 0.6 * (value / denominator)) for value in normalized]

    bars = axis.bar(
        x_positions,
        reel_quality["quality_score"],
        color=colors,
        edgecolor=CHART_BG,
        linewidth=1.2,
        width=0.62,
        zorder=3,
    )
    axis.set_xticks(x_positions)
    axis.set_xticklabels(reel_quality["short_id"])
    _annotate_bars(axis, bars, fmt="{:.2f}")
    axis.set_ylim(0, min(1.05, reel_quality["quality_score"].max() + 0.12))
    _style_axes(axis, "Average Image Quality per Reel", "Reel ID", "Mean Quality Score")

    return {
        "title": "Reel Quality",
        "image": _save_chart(figure, "reel_quality.png"),
        "wide": False,
    }


def build_charts(audio_df: pd.DataFrame, image_df: pd.DataFrame) -> list[dict]:
    _configure_matplotlib()
    charts: list[dict] = []

    if not audio_df.empty:
        charts.append(_chart_tempo(audio_df))
        charts.append(_chart_timeline(audio_df))
        charts.append(_chart_slide_duration(audio_df))

    if not image_df.empty:
        charts.append(_chart_quality_labels(image_df))
        charts.append(_chart_blur_quality(image_df))
        charts.append(_chart_reel_quality(image_df))

    return charts


def get_dashboard_context() -> dict:
    audio_df = load_audio_features()
    image_df = load_image_features()
    has_data = not audio_df.empty or not image_df.empty

    return {
        "has_data": has_data,
        "summary": build_summary(audio_df, image_df),
        "charts": build_charts(audio_df, image_df),
        "recent_reels": build_recent_reels(audio_df, image_df),
    }
