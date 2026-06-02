#!/usr/bin/env python3
"""
Generate report-quality figures from the telemetry SQLite database.

This script is intentionally read-only: it does not modify telemetry tables,
dashboard behavior, scoring, or recommendation logic.
"""

import argparse
import csv
import math
import os
import sqlite3
from pathlib import Path


DEFAULT_CONTAINER_DB = Path("/workspace/data/metrics.db")
DEFAULT_HOST_DB = Path("data/metrics.db")
DEFAULT_OUTPUT_DIR = Path("reports/figures")

MAIN_EXPERIMENTS = [
    ("big5_baseline_001", "5-node baseline"),
    ("big5_node1_delay20_001", "5-node node1 delay (100ms)"),
    ("big4_recommended_without_node1_after100ms_001", "Recommended 4-node subset"),
]

OPTIONAL_EXPERIMENTS = [
    ("five_baseline_clean_001", "5-node baseline (3 epochs)"),
    ("five_node1_delay_20ms_001", "5-node node1 delay (3 epochs)"),
    ("recommended_without_node1_001", "Recommended 4-node subset (3 epochs)"),
    ("four_node_subset_baseline_001", "4-node subset baseline"),
]

HEATMAP_JOBS = {
    "fig5_rtt_matrix_baseline.png": (
        "big5_baseline_001",
        "Per-Peer RTT Matrix: 5-Node Baseline",
    ),
    "fig6_rtt_matrix_node1_delay.png": (
        "big5_node1_delay20_001",
        "Per-Peer RTT Matrix: Node1 Delay Stress Test (100ms netem)",
    ),
    "fig7_rtt_matrix_recommended_subset.png": (
        "big4_recommended_without_node1_after100ms_001",
        "Per-Peer RTT Matrix: Recommendation-Guided 4-Node Run",
    ),
}


def resolve_default_db():
    if DEFAULT_CONTAINER_DB.exists():
        return DEFAULT_CONTAINER_DB
    return DEFAULT_HOST_DB


def connect(db_path):
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def table_exists(conn, table_name):
    row = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
        (table_name,),
    ).fetchone()
    return row is not None


def metric_summaries(conn, job_ids):
    if not job_ids:
        return {}
    placeholders = ",".join("?" for _ in job_ids)
    rows = conn.execute(
        f"""
        SELECT job_id,
               COUNT(*) AS metrics_rows,
               COUNT(DISTINCT node_id) AS node_count,
               COUNT(*) * 1.0 / NULLIF(COUNT(DISTINCT node_id), 0) AS avg_rows_per_node,
               AVG(all_reduce_ms) AS avg_all_reduce_ms,
               AVG(rtt_ms) AS avg_rtt_ms,
               AVG(nic_bytes_sent + nic_bytes_recv) / 1000000.0 AS avg_total_nic_MBps
        FROM metrics
        WHERE job_id IN ({placeholders})
        GROUP BY job_id
        """,
        job_ids,
    ).fetchall()
    return {row["job_id"]: dict(row) for row in rows}


def rtt_counts(conn, job_ids):
    if not job_ids or not table_exists(conn, "rtt_metrics"):
        return {}
    placeholders = ",".join("?" for _ in job_ids)
    rows = conn.execute(
        f"""
        SELECT job_id,
               COUNT(*) AS rtt_rows,
               SUM(CASE WHEN rtt_ms IS NULL THEN 1 ELSE 0 END) AS timeout_count
        FROM rtt_metrics
        WHERE job_id IN ({placeholders})
        GROUP BY job_id
        """,
        job_ids,
    ).fetchall()
    return {row["job_id"]: dict(row) for row in rows}


def per_node_all_reduce(conn, job_ids):
    if not job_ids:
        return []
    placeholders = ",".join("?" for _ in job_ids)
    return conn.execute(
        f"""
        SELECT job_id, node_id, AVG(all_reduce_ms) AS avg_all_reduce_ms
        FROM metrics
        WHERE job_id IN ({placeholders})
        GROUP BY job_id, node_id
        ORDER BY node_id, job_id
        """,
        job_ids,
    ).fetchall()


def rtt_pair_rows(conn, job_ids):
    if not job_ids or not table_exists(conn, "rtt_metrics"):
        return []
    placeholders = ",".join("?" for _ in job_ids)
    return conn.execute(
        f"""
        SELECT job_id,
               node_id AS source_node,
               peer_node_id,
               COUNT(*) AS rows_seen,
               AVG(rtt_ms) AS avg_rtt_ms,
               MAX(rtt_ms) AS max_rtt_ms,
               SUM(CASE WHEN rtt_ms IS NULL THEN 1 ELSE 0 END) AS timeout_count
        FROM rtt_metrics
        WHERE job_id IN ({placeholders})
        GROUP BY job_id, node_id, peer_node_id
        ORDER BY job_id, node_id, peer_node_id
        """,
        job_ids,
    ).fetchall()


def job_exists(conn, job_id):
    row = conn.execute("SELECT 1 FROM metrics WHERE job_id = ? LIMIT 1", (job_id,)).fetchone()
    return row is not None


def present_experiments(conn, experiments):
    present = []
    missing = []
    for job_id, label in experiments:
        if job_exists(conn, job_id):
            present.append((job_id, label))
        else:
            missing.append(job_id)
    return present, missing


def warn_missing(missing):
    for job_id in missing:
        print(f"Warning: job {job_id!r} not found; skipping related figure data.")


def value_label(ax, bars, fmt="{:.1f}"):
    for bar in bars:
        height = bar.get_height()
        if height is None or math.isnan(height):
            continue
        ax.annotate(
            fmt.format(height),
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8,
        )


def save_bar_chart(plt, output_path, labels, values, ylabel, title, value_fmt="{:.1f}"):
    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(labels, values)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.25)
    ax.tick_params(axis="x", rotation=15)
    value_label(ax, bars, value_fmt)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def figure_1_to_3(plt, output_dir, experiments, summaries):
    labels = [label for job_id, label in experiments if job_id in summaries]
    all_reduce = [summaries[job_id]["avg_all_reduce_ms"] for job_id, _ in experiments if job_id in summaries]
    rtt = [summaries[job_id]["avg_rtt_ms"] for job_id, _ in experiments if job_id in summaries]
    nic = [summaries[job_id]["avg_total_nic_MBps"] for job_id, _ in experiments if job_id in summaries]

    generated = []
    if labels:
        path = output_dir / "fig1_avg_all_reduce_by_experiment.png"
        save_bar_chart(
            plt,
            path,
            labels,
            all_reduce,
            "Average all-reduce time (ms)",
            "Average All-Reduce Time Across Experiments",
        )
        generated.append(path)

        path = output_dir / "fig2_avg_rtt_by_experiment.png"
        save_bar_chart(
            plt,
            path,
            labels,
            rtt,
            "Average RTT (ms)",
            "Average RTT Across Experiments",
        )
        generated.append(path)

        path = output_dir / "fig3_avg_nic_total_by_experiment.png"
        save_bar_chart(
            plt,
            path,
            labels,
            nic,
            "Average total NIC throughput (MB/s)",
            "Average NIC Throughput Across Experiments",
        )
        generated.append(path)
    return generated


def figure_4(plt, output_dir, experiments, rows):
    if not rows:
        print("Warning: no per-node all-reduce data found; skipping figure 4.")
        return None

    job_to_label = dict(experiments)
    nodes = sorted({row["node_id"] for row in rows})
    jobs = [job_id for job_id, _ in experiments]
    values = {(row["job_id"], row["node_id"]): row["avg_all_reduce_ms"] for row in rows}

    fig, ax = plt.subplots(figsize=(10, 5.5))
    x_positions = list(range(len(nodes)))
    width = 0.8 / max(len(jobs), 1)

    for idx, job_id in enumerate(jobs):
        offsets = [x - 0.4 + width / 2 + idx * width for x in x_positions]
        heights = [values.get((job_id, node), 0.0) for node in nodes]
        bars = ax.bar(offsets, heights, width=width, label=job_to_label[job_id])
        for bar, node in zip(bars, nodes):
            if (job_id, node) not in values:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    1,
                    "not\neval",
                    ha="center",
                    va="bottom",
                    fontsize=7,
                    color="dimgray",
                )

    ax.set_xticks(x_positions)
    ax.set_xticklabels(nodes)
    ax.set_ylabel("Average all-reduce time (ms)")
    ax.set_title("Per-Node All-Reduce Comparison")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    path = output_dir / "fig4_per_node_all_reduce.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def rtt_matrix(conn, job_id):
    if not table_exists(conn, "rtt_metrics"):
        return [], []
    rows = conn.execute(
        """
        SELECT node_id, peer_node_id, AVG(rtt_ms) AS avg_rtt_ms
        FROM rtt_metrics
        WHERE job_id = ?
        GROUP BY node_id, peer_node_id
        ORDER BY node_id, peer_node_id
        """,
        (job_id,),
    ).fetchall()
    nodes = sorted({row["node_id"] for row in rows} | {row["peer_node_id"] for row in rows})
    return nodes, rows


def heatmap_figure(plt, output_dir, conn, filename, job_id, title):
    nodes, rows = rtt_matrix(conn, job_id)
    if not rows:
        print(f"Warning: no RTT matrix data for job {job_id!r}; skipping {filename}.")
        return None

    index = {node: i for i, node in enumerate(nodes)}
    matrix = [[math.nan for _ in nodes] for _ in nodes]
    for row in rows:
        matrix[index[row["node_id"]]][index[row["peer_node_id"]]] = (
            math.nan if row["avg_rtt_ms"] is None else row["avg_rtt_ms"]
        )

    fig, ax = plt.subplots(figsize=(7.2, 6.2))
    cmap = plt.cm.viridis.copy()
    cmap.set_bad(color="#eeeeee")
    image = ax.imshow(matrix, cmap=cmap)
    ax.set_xticks(range(len(nodes)))
    ax.set_yticks(range(len(nodes)))
    ax.set_xticklabels(nodes)
    ax.set_yticklabels(nodes)
    ax.set_xlabel("Peer node")
    ax.set_ylabel("Source node")
    ax.set_title(title)
    fig.colorbar(image, ax=ax, label="Average RTT (ms)")

    for i, _ in enumerate(nodes):
        for j, _ in enumerate(nodes):
            value = matrix[i][j]
            if math.isnan(value):
                text = ""
            else:
                text = f"{value:.1f}"
            ax.text(j, i, text, ha="center", va="center", color="white" if not math.isnan(value) else "black", fontsize=8)

    fig.tight_layout()
    path = output_dir / filename
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def figure_8(plt, output_dir, summaries):
    degraded = summaries.get("big5_node1_delay20_001")
    recommended = summaries.get("big4_recommended_without_node1_after100ms_001")
    if not degraded or not recommended:
        print("Warning: missing degraded or recommended-subset job; skipping figure 8.")
        return None, None

    degraded_ar = degraded["avg_all_reduce_ms"] or 0
    recommended_ar = recommended["avg_all_reduce_ms"] or 0
    degraded_rtt = degraded["avg_rtt_ms"] or 0
    recommended_rtt = recommended["avg_rtt_ms"] or 0

    ar_reduction = 100 * (degraded_ar - recommended_ar) / degraded_ar if degraded_ar else 0
    rtt_reduction = 100 * (degraded_rtt - recommended_rtt) / degraded_rtt if degraded_rtt else 0

    path = output_dir / "fig8_recovery_improvement_summary.png"
    save_bar_chart(
        plt,
        path,
        ["All-reduce reduction", "RTT reduction"],
        [ar_reduction, rtt_reduction],
        "Percent reduction (%)",
        "Performance Recovery After Avoiding Degraded Node",
        "{:.1f}%",
    )
    return path, {"all_reduce_reduction": ar_reduction, "rtt_reduction": rtt_reduction}


def write_experiment_summary(output_dir, experiments, summaries, rtt_summary):
    path = output_dir / "experiment_summary.csv"
    fields = [
        "job_id",
        "friendly_label",
        "node_count",
        "epochs_per_node",
        "avg_all_reduce_ms",
        "avg_rtt_ms",
        "avg_total_nic_MBps",
        "metrics_rows",
        "rtt_rows",
        "timeout_count",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for job_id, label in experiments:
            summary = summaries.get(job_id)
            if not summary:
                continue
            rtt = rtt_summary.get(job_id, {})
            writer.writerow({
                "job_id": job_id,
                "friendly_label": label,
                "node_count": summary["node_count"],
                "epochs_per_node": round(summary["avg_rows_per_node"] or 0, 2),
                "avg_all_reduce_ms": round(summary["avg_all_reduce_ms"] or 0, 4),
                "avg_rtt_ms": round(summary["avg_rtt_ms"] or 0, 4),
                "avg_total_nic_MBps": round(summary["avg_total_nic_MBps"] or 0, 4),
                "metrics_rows": summary["metrics_rows"],
                "rtt_rows": rtt.get("rtt_rows", 0) or 0,
                "timeout_count": rtt.get("timeout_count", 0) or 0,
            })
    return path


def write_rtt_pair_summary(output_dir, rows):
    path = output_dir / "rtt_pair_summary.csv"
    fields = [
        "job_id",
        "source_node",
        "peer_node",
        "rows",
        "avg_rtt_ms",
        "max_rtt_ms",
        "timeout_count",
        "status",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            avg_rtt = row["avg_rtt_ms"]
            max_rtt = row["max_rtt_ms"]
            timeout_count = row["timeout_count"] or 0
            if timeout_count:
                status = "timeout"
            elif (avg_rtt is not None and avg_rtt >= 10.0) or (max_rtt is not None and max_rtt >= 25.0):
                status = "high"
            else:
                status = "ok"
            writer.writerow({
                "job_id": row["job_id"],
                "source_node": row["source_node"],
                "peer_node": row["peer_node_id"],
                "rows": row["rows_seen"],
                "avg_rtt_ms": "" if avg_rtt is None else round(avg_rtt, 4),
                "max_rtt_ms": "" if max_rtt is None else round(max_rtt, 4),
                "timeout_count": timeout_count,
                "status": status,
            })
    return path


def main():
    parser = argparse.ArgumentParser(description="Generate report figures from telemetry SQLite data.")
    parser.add_argument("--db", default=str(resolve_default_db()), help="Path to metrics.db")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Directory for figures and CSVs")
    args = parser.parse_args()

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        raise SystemExit(
            "matplotlib is required to generate report figures. "
            "Install it locally or rebuild the Docker image with the updated Dockerfile. "
            f"Original error: {exc}"
        )

    db_path = Path(args.db)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not db_path.exists():
        raise SystemExit(f"Database not found: {db_path}")

    all_experiments = MAIN_EXPERIMENTS + OPTIONAL_EXPERIMENTS
    all_job_ids = [job_id for job_id, _ in all_experiments]
    generated_figures = []

    with connect(db_path) as conn:
        if not table_exists(conn, "metrics"):
            raise SystemExit("metrics table does not exist in the database.")

        main_present, main_missing = present_experiments(conn, MAIN_EXPERIMENTS)
        optional_present, optional_missing = present_experiments(conn, OPTIONAL_EXPERIMENTS)
        warn_missing(main_missing)
        for job_id in optional_missing:
            print(f"Note: optional job {job_id!r} not found; skipping optional summary row.")

        present_experiments_all = main_present + optional_present
        present_job_ids = [job_id for job_id, _ in present_experiments_all]
        summaries = metric_summaries(conn, all_job_ids)
        rtt_summary = rtt_counts(conn, all_job_ids)

        generated_figures.extend(figure_1_to_3(plt, output_dir, main_present, summaries))

        fig4 = figure_4(plt, output_dir, main_present, per_node_all_reduce(conn, [job_id for job_id, _ in main_present]))
        if fig4:
            generated_figures.append(fig4)

        for filename, (job_id, title) in HEATMAP_JOBS.items():
            fig = heatmap_figure(plt, output_dir, conn, filename, job_id, title)
            if fig:
                generated_figures.append(fig)

        fig8, reductions = figure_8(plt, output_dir, summaries)
        if fig8:
            generated_figures.append(fig8)

        experiment_csv = write_experiment_summary(output_dir, present_experiments_all, summaries, rtt_summary)
        rtt_csv = write_rtt_pair_summary(output_dir, rtt_pair_rows(conn, present_job_ids))

    print()
    print("Generated figures:")
    for path in generated_figures:
        print(f"- {path}")
    print("Generated CSV:")
    print(f"- {experiment_csv}")
    print(f"- {rtt_csv}")

    baseline = summaries.get("big5_baseline_001")
    degraded = summaries.get("big5_node1_delay20_001")
    recommended = summaries.get("big4_recommended_without_node1_after100ms_001")

    print()
    print("Key numeric results:")
    if baseline:
        print(f"- baseline average all-reduce: {baseline['avg_all_reduce_ms']:.2f} ms")
    else:
        print("- baseline average all-reduce: missing")
    if degraded:
        print(f"- degraded average all-reduce: {degraded['avg_all_reduce_ms']:.2f} ms")
    else:
        print("- degraded average all-reduce: missing")
    if recommended:
        print(f"- recommended subset average all-reduce: {recommended['avg_all_reduce_ms']:.2f} ms")
    else:
        print("- recommended subset average all-reduce: missing")
    if reductions:
        print(f"- all-reduce reduction: {reductions['all_reduce_reduction']:.1f}%")
        print(f"- RTT reduction: {reductions['rtt_reduction']:.1f}%")


if __name__ == "__main__":
    main()
