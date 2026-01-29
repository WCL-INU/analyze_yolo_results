import os
from pathlib import Path
import duckdb
import numpy as np
from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt


YOLO_RESULTS_DIR = Path("./data")
YOLO_RESULTS_FILE = "cut_ANU-25-summer-4_20260105_*.parquet"
# YOLO_RESULTS_FILE = "cut_ANU-25-summer-17_20260112_*.parquet"


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

    # frame  box_index     x     y  width  height

    # Convert to NumPy once to speed up downstream plotting
    cols = ["x", "y", "width", "height", "frame", "box_index"]
    data_arrays = {
        col: df[col].to_numpy(dtype=float, copy=False)
        for col in cols
        if col in df.columns
    }
    print(f"Converted columns to NumPy arrays: {list(data_arrays.keys())}")

    # # Plot histograms of x, y, width, height
    # fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    # for ax, col in zip(axes.ravel(), cols):
    #     if col in df.columns:
    #         ax.hist(data_arrays[col][~np.isnan(data_arrays[col])], bins=500, color="C0", edgecolor="black")
    #         ax.set_title(f"Histogram of {col}")
    #         ax.set_xlabel(col)
    #         ax.set_ylabel("Count")
    #     else:
    #         ax.text(0.5, 0.5, f"{col} not found", ha="center", va="center")
    # plt.tight_layout()
    # plt.show(block=False)

    # Scatter plot of x vs y
    if {"x", "y"} <= set(df.columns):
        xy = np.column_stack((data_arrays["x"], data_arrays["y"]))
        xy = xy[~np.isnan(xy).any(axis=1)]
        fig2, ax2 = plt.subplots(figsize=(8, 6))
        hb = ax2.hexbin(xy[:, 0], xy[:, 1], gridsize=100, cmap="inferno", mincnt=1)
        fig2.colorbar(hb, ax=ax2, label="counts")
        ax2.set_xlabel("x")
        ax2.set_ylabel("y")
        ax2.set_title("Distribution on x-y plane (hexbin)")
        ax2.grid(True, linestyle=":", linewidth=0.5)
        ax2.invert_yaxis()  # Invert y-axis if needed
        plt.tight_layout()
        plt.show(block=False)
    else:
        print("Columns x and y not found; skipping hexbin plot.")

    # # Visualize bounding boxes on a all frames at once
    # if {"x", "y", "width", "height"} <= set(df.columns):
    #     # Using PatchCollection to avoid slow DataFrame iteration
    #     bbox_data = np.column_stack(
    #         (data_arrays["x"], data_arrays["y"], data_arrays["width"], data_arrays["height"])
    #     )
    #     bbox_data = bbox_data[~np.isnan(bbox_data).any(axis=1)]
    #     fig3, ax3 = plt.subplots(figsize=(10, 8))
    #     rectangles = [
    #         Rectangle((x, y), w, h, linewidth=1, edgecolor="C1", facecolor="none", alpha=0.3)
    #         for x, y, w, h in bbox_data
    #     ]
    #     from matplotlib.collections import PatchCollection

    #     ax3.add_collection(PatchCollection(rectangles, match_original=True))
    #     ax3.set_xlabel("x")
    #     ax3.set_ylabel("y")
    #     ax3.set_title("Bounding Boxes from YOLO Detections")
    #     ax3.set_xlim(0, 1640)
    #     ax3.set_ylim(0, 1232)
    #     ax3.grid(True, linestyle=":", linewidth=0.5)
    #     ax3.invert_yaxis()  # Invert y-axis if needed
    #     plt.tight_layout()
    #     plt.show()
    # else:
    #     print("Bounding box columns missing; skipping rectangle plot.")

    # Visualize bounding boxes on a all frames at once with color by frame
    if {"x", "y", "width", "height", "frame"} <= set(df.columns):
        # Using PatchCollection to avoid slow DataFrame iteration
        bbox_data = np.column_stack(
            (
                data_arrays["x"],
                data_arrays["y"],
                data_arrays["width"],
                data_arrays["height"],
                data_arrays["frame"],
            )
        )
        bbox_data = bbox_data[~np.isnan(bbox_data).any(axis=1)]
        fig4, ax4 = plt.subplots(figsize=(10, 8))
        frames = bbox_data[:, 4]
        norm = plt.Normalize(vmin=np.min(frames), vmax=np.max(frames))
        cmap = plt.get_cmap("viridis")
        rectangles = [
            Rectangle(
                (x, y),
                w,
                h,
                linewidth=1,
                edgecolor=cmap(norm(frame)),
                facecolor="none",
                alpha=0.5,
            )
            for x, y, w, h, frame in bbox_data
        ]
        from matplotlib.collections import PatchCollection

        ax4.add_collection(PatchCollection(rectangles, match_original=True))
        ax4.set_xlabel("x")
        ax4.set_ylabel("y")
        ax4.set_title("Bounding Boxes from YOLO Detections Colored by Frame")
        ax4.set_xlim(0, 1640)
        ax4.set_ylim(0, 1232)
        ax4.grid(True, linestyle=":", linewidth=0.5)
        ax4.invert_yaxis()  # Invert y-axis if needed
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = fig4.colorbar(sm, ax=ax4)
        cbar.set_label("Frame Number")
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()
