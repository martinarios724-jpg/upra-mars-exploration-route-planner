import argparse
from pathlib import Path
import sys

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import numpy as np

OUTBOUND_ROUTES = {
    "h": ("ruta_H.csv", "Elevation only (H)"),
    "topografica": ("ruta_topografica.csv", "Terrain (H + slope + roughness)"),
    "temperatura": ("ruta_temperatura.csv", "Terrain + temperature"),
}
RETURN_FILE = "retorno_temperatura.csv"


def load_route_csv(file_path):
    if not file_path.is_file():
        raise FileNotFoundError(f"File not found: {file_path}")
    try:
        data = np.loadtxt(file_path, delimiter=",", skiprows=1, ndmin=2)
    except (OSError, ValueError) as exc:
        raise ValueError(f"Cannot read {file_path}: {exc}") from exc
    if data.ndim != 2 or data.shape[1] != 2 or len(data) < 1:
        raise ValueError(f"Invalid CSV format in {file_path}: expected latitud,longitud columns")
    if not np.isfinite(data).all():
        raise ValueError(f"Non-finite coordinates in {file_path}")
    if (np.abs(data[:, 0]) > 90).any() or (np.abs(data[:, 1]) > 180).any():
        raise ValueError(f"Coordinates out of range in {file_path}")
    return data  # Ordered [latitude, longitude] samples


def choose_route():
    print("\n=== MARS MISSION ANIMATION ===")
    print("1. Elevation only")
    print("2. Terrain")
    print("3. Terrain + temperature")
    options = {"1": "h", "2": "topografica", "3": "temperatura"}
    while True:
        response = input("Select outbound route [1/2/3]: ").strip()
        if response in options:
            return options[response]
        print("Invalid option. Enter 1, 2, or 3.")


def sampled_indices(count, max_frames):
    """Limit animation frames without changing route data or point order."""
    if count <= max_frames:
        return np.arange(count, dtype=int)
    return np.unique(np.rint(np.linspace(0, count - 1, max_frames)).astype(int))


def animate_route(outbound, inbound, route_label, interval_ms=65, max_frames=350,
                  save_path=None):
    # Outbound must end at B; the return must start at B and end at A.
    coordinate_tolerance = 1e-5  # CSVs contain matching grid-node coordinates
    if np.max(np.abs(outbound[-1] - inbound[0])) > coordinate_tolerance:
        raise ValueError("Return route does not start at B. Check that both CSVs belong to the same mission.")
    if np.max(np.abs(outbound[0] - inbound[-1])) > coordinate_tolerance:
        raise ValueError("Return route does not end at A. Check that both CSVs belong to the same mission.")

    outbound_steps = sampled_indices(len(outbound), max_frames)
    return_steps = sampled_indices(len(inbound), max_frames)
    outbound_frames = len(outbound_steps)
    total = outbound_frames + len(return_steps)

    fig, ax = plt.subplots(figsize=(11, 7))
    if hasattr(fig.canvas.manager, "set_window_title"):
        fig.canvas.manager.set_window_title("Mars mission: outbound and return")
    ax.set_title("Mars mission: exploration and return to base")
    ax.set_xlabel("Martian longitude (°)")
    ax.set_ylabel("Martian latitude (°)")
    ax.grid(True, alpha=0.25)

    # Display geographic paths without implying a terrain or temperature background.
    ax.plot(outbound[:, 1], outbound[:, 0], color="#e35d5b", alpha=0.25, lw=1.5,
            label=f"Outbound: {route_label}")
    ax.plot(inbound[:, 1], inbound[:, 0], color="#8254b8", alpha=0.25,
            lw=1.5, label="Return: terrain + temperature")
    ax.scatter([outbound[0, 1]], [outbound[0, 0]], s=90, color="limegreen",
               edgecolors="black", zorder=6, label="Base (A)")
    ax.scatter([outbound[-1, 1]], [outbound[-1, 0]], s=90, color="royalblue",
               edgecolors="black", zorder=6, label="Destination (B)")

    # The active route and position marker change color for the return.
    active_line, = ax.plot([], [], lw=3, color="#e35d5b", zorder=4)
    marker, = ax.plot([], [], marker="o", markersize=10, color="#e35d5b",
                       markeredgecolor="black", linestyle="None", zorder=7)
    status_text = ax.text(0.02, 0.02, "", transform=ax.transAxes,
                     va="bottom", fontsize=11,
                     bbox=dict(facecolor="white", alpha=0.88, edgecolor="gray"))

    lon_total = np.concatenate([outbound[:, 1], inbound[:, 1]])
    lat_total = np.concatenate([outbound[:, 0], inbound[:, 0]])
    width_span = max(float(np.ptp(lon_total)), 1.0)
    height_span = max(float(np.ptp(lat_total)), 1.0)
    ax.set_xlim(float(lon_total.min()) - width_span * 0.08,
                float(lon_total.max()) + width_span * 0.08)
    ax.set_ylim(float(lat_total.min()) - height_span * 0.08,
                float(lat_total.max()) + height_span * 0.08)
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()

    def update(frame):
        if frame < outbound_frames:
            k = int(outbound_steps[frame])
            coords = outbound[:k + 1]
            color = "#e35d5b"
            phase = "OUTBOUND: A → POI(s) → B"
            progress = (k + 1) / len(outbound) * 100
        else:
            k = int(return_steps[frame - outbound_frames])
            coords = inbound[:k + 1]
            color = "#8254b8"
            phase = "RETURN: B → A (no required POIs)"
            progress = (k + 1) / len(inbound) * 100
        active_line.set_data(coords[:, 1], coords[:, 0])
        active_line.set_color(color)
        marker.set_data([coords[-1, 1]], [coords[-1, 0]])
        marker.set_color(color)
        status_text.set_text(f"{phase}\nVisual progress: {progress:.0f}%\n"
                        f"Lat: {coords[-1, 0]:.3f}° | Lon: {coords[-1, 1]:.3f}°")
        return active_line, marker, status_text

    animation = FuncAnimation(fig, update, frames=total,
                             interval=interval_ms, repeat=False,
                             blit=False, cache_frame_data=False)
    if save_path:
        output_path = Path(save_path)
        if output_path.suffix.lower() != ".gif":
            raise ValueError("--save requires a .gif file")
        print(f"Saving GIF ({total} frames): {output_path}")
        animation.save(str(output_path), writer=PillowWriter(fps=max(1, round(1000 / interval_ms))))
        print(f"GIF saved: {output_path.resolve()}")
    else:
        print("Displaying animation. Close the window to exit.")
        plt.show()
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Animate an outbound A→POI→B route followed by return B→A")
    parser.add_argument("--folder", type=Path, default=Path("."),
                        help="Directory containing the CSV files (default: current directory)")
    parser.add_argument("--route", choices=OUTBOUND_ROUTES.keys(),
                        help="Outbound route; prompts if omitted")
    parser.add_argument("--interval", type=int, default=65,
                        help="Milliseconds between frames (default: 65)")
    parser.add_argument("--max-frames", type=int, default=350,
                        help="Maximum frames per leg; does not alter the route CSVs")
    parser.add_argument("--save", type=Path,
                        help="Export a GIF instead of opening a window")
    args = parser.parse_args()
    if args.interval < 1 or args.max_frames < 2:
        parser.error("--interval must be >= 1 and --max-frames must be >= 2")

    selection = args.route or choose_route()
    outbound_filename, route_label = OUTBOUND_ROUTES[selection]
    outbound_file = args.folder / outbound_filename
    return_path = args.folder / RETURN_FILE
    if not return_path.is_file():
        raise FileNotFoundError(
            f"Missing {return_path}. In the main planner, answer 'y' "
            "to the return-to-base prompt to generate retorno_temperatura.csv."
        )
    outbound = load_route_csv(outbound_file)
    inbound = load_route_csv(return_path)
    print(f"Outbound: {outbound_file} ({len(outbound)} nodes)")
    print(f"Return: {return_path} ({len(inbound)} nodes)")
    animate_route(outbound, inbound, route_label, args.interval, args.max_frames,
           args.save)


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        sys.exit(1)
