import os
from pathlib import Path
import duckdb
import numpy as np
from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt


YOLO_RESULTS_DIR = Path("./data")
YOLO_RESULTS_FILE = "cut_ANU-25-summer-17_20260112_*.parquet"


def main():
    print("Hello from analyze-yolo-results!")

    yolo_results_path = YOLO_RESULTS_DIR / YOLO_RESULTS_FILE
    print(f"YOLO results path: {yolo_results_path}")

    con = duckdb.connect()
    # DuckDB supports globs via parquet_scan
    df = con.execute(
        f"SELECT * FROM parquet_scan('{yolo_results_path.as_posix()}')"
    ).fetchdf()

    print(f"Loaded {len(df)} rows, {len(df.columns)} columns")
    print(df.head())

    col_map = {c.lower(): c for c in df.columns}

    # converter -> (x_min, y_min, width, height)
    def to_xywh(row):
        cx = row[col_map["x"]]
        cy = row[col_map["y"]]
        w = row[col_map["width"]]
        h = row[col_map["height"]]
        return cx - w / 2.0, cy - h / 2.0, w, h

    # compute extents to set axes limits
    boxes = [to_xywh(row) for _, row in df.iterrows()]

    fig, ax = plt.subplots(figsize=(10, 10))
    for i, (x, y, w, h) in enumerate(boxes):
        ec = "red"
        rect = Rectangle(
            (x, y), w, h, linewidth=1, edgecolor=ec, facecolor="none", alpha=0.6
        )
        ax.add_patch(rect)

    ax.set_xlim(0, 1640)  # fixed image width
    ax.set_ylim(0, 1232)  # fixed image height
    ax.set_aspect("equal")
    ax.set_title("All detected boxes")
    plt.tight_layout()
    plt.show(block=True)


if __name__ == "__main__":
    main()
