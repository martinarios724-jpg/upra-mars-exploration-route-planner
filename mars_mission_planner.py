import time
import heapq
from itertools import count
import numpy as np
import matplotlib.pyplot as plt
import rasterio
from pathlib import Path
from scipy.ndimage import uniform_filter, correlate

# ============================================================
# 1. CONFIGURATION
# ============================================================

ELEVATION_ARRAY_FILE = "H_completo.npy"
DEM_GEOTIFF_FILE = "SECTOR_MARTE_200M.tif"

DOWNSAMPLE_FACTOR = 32  
NO_DATA_VALUE = -32768

MARS_RADIUS_M = 3396000.0

# Set mission endpoints as (latitude, longitude) in Martian degrees.
START_LATLON = (-15, -40)
END_LATLON = (-60, -20)

# Scientific POIs: formation centers, not verified landing or sampling sites.
POI_CATALOG = {
    1: {"name": "Eberswalde Crater", "latlon": (-23.86, -33.27)},
    2: {"name": "Holden Crater", "latlon": (-26.38, -34.92)},
    3: {"name": "Hale Crater", "latlon": (-35.78, -36.50)},
    4: {"name": "Argyre Planitia", "latlon": (-49.67, -44.00)},
    5: {"name": "Galle Crater", "latlon": (-50.87, -30.90)},
    6: {"name": "Copernicus Crater", "latlon": (-48.87, -169.20)},
    7: {"name": "Reynolds Crater", "latlon": (-74.99, -157.59)},
    8: {"name": "Newton Crater", "latlon": (-40.47, -158.10)},
    9: {"name": "Kepler Crater", "latlon": (-46.76, -140.90)},
    10: {"name": "Ross Crater", "latlon": (-57.39, -107.84)},
    11: {"name": "Aniak Crater", "latlon": (-31.84, -69.56)},
    12: {"name": "Margaritifer Chaos", "latlon": (-9.30, -21.70)},
}
SELECTED_POIS = [1, 2]  # Empty list = direct A-to-B route.

# Automatic order minimizes the full-model cost; manual preserves the list order.
POI_ORDER_MODE = "automatic"

# Limit expensive pairwise A* searches during POI optimization.
MAX_OPTIMIZED_POIS = 7

# Optional direct return from destination B to base A.
def ask_return_to_base():
    """Ask whether to calculate a direct B-to-A return route."""
    while True:
        answer = input(
            "Calculate a direct return route from B to base A? [y/n]: "
        ).strip().lower()
        if answer in {"y", "yes"}:
            return "return"
        if answer in {"n", "no"}:
            return "none"
        print("Invalid answer. Enter y or n.")


RETURN_MODE = ask_return_to_base()

# Select the preprocessed surface-temperature scenario.
SEASON = "summer" # ("spring", "summer", "autumm", "winter")
TIME_OF_DAY = "day" # ("day", "night")

# Experimental terrain thresholds and weights; not EVA safety limits.
MAX_SLOPE_DEG = 20.0
MAX_ROUGHNESS_M = 500.0

SLOPE_WEIGHT = 2.0
ROUGHNESS_WEIGHT = 1.0

# Thermal penalty parameters (kelvin).
TEMPERATURE_WEIGHT = 0.2
REFERENCE_TEMPERATURE_K = 220.0  # K
TEMPERATURE_SCALE_K = 40.0  # K
TEMPERATURE_DIR = Path("temperaturas_alineadas")

ROUGHNESS_WINDOW = 5

NEIGHBOR_OFFSETS = [
    (-1, 0), (1, 0),
    (0, -1), (0, 1),
    (-1, -1), (-1, 1),
    (1, -1), (1, 1)
]


# Map English scenario options to the existing Spanish temperature filenames.
SEASON_FILE_NAMES = {"spring": "primavera", "summer": "verano", "autumn": "otono", "winter": "invierno"}
PERIOD_FILE_NAMES = {"day": "dia", "night": "noche"}

# ============================================================
# THERMAL SCENARIO
# ============================================================

season = SEASON.strip().lower()
time_of_day = TIME_OF_DAY.strip().lower()

if season not in {"spring", "summer", "autumn", "winter"}:
    raise ValueError("Invalid season")
if time_of_day not in {"day", "night"}:
    raise ValueError("Invalid time of day")
temperature_file = TEMPERATURE_DIR / f"T_{SEASON_FILE_NAMES[season]}_{PERIOD_FILE_NAMES[time_of_day]}_alineada.npy"
if not temperature_file.is_file():
    raise FileNotFoundError(f"Temperature scenario not found: {temperature_file}")


# ============================================================
# 2. LOAD DOWNSAMPLED DEM
# ============================================================

t0 = time.perf_counter()

original_elevation = np.load(ELEVATION_ARRAY_FILE, mmap_mode="r")

with rasterio.open(DEM_GEOTIFF_FILE) as src:

    if src.shape != original_elevation.shape:
        raise ValueError(
            "GeoTIFF and elevation array dimensions "
            "do not match. Check the source files."
        )

    if not src.crs.is_geographic:
        raise ValueError(
            "Expected a geographic DEM "
            "in Martian degrees."
        )

    transform = src.transform

    longitude_resolution = abs(transform.a)
    latitude_resolution = abs(transform.e)

    north_latitude = transform.f

H = np.asarray(
    original_elevation[::DOWNSAMPLE_FACTOR, ::DOWNSAMPLE_FACTOR],
    dtype=np.float64
).copy()

H[H == NO_DATA_VALUE] = np.nan

num_rows, num_cols = H.shape

# Temperature grid must match the downsampled DEM.
T = np.load(temperature_file, mmap_mode="r")
if T.shape != H.shape:
    raise ValueError(f"Shape mismatch: DEM {H.shape}, temperature {T.shape}")
if not np.isfinite(T).all():
    raise ValueError("Temperature grid contains NaN or infinite values")
print(f"Temperature scenario: {season}, {time_of_day}; file: {temperature_file.name}")

sampled_original_rows = np.arange(num_rows) * DOWNSAMPLE_FACTOR

row_latitudes = (
    north_latitude
    - (sampled_original_rows + 0.5) * latitude_resolution
)

# Convert geographic pixel spacing into approximate meters on Mars.
row_dx_m = (
    MARS_RADIUS_M
    * np.cos(np.radians(row_latitudes))
    * np.radians(longitude_resolution)
    * DOWNSAMPLE_FACTOR
)

dy_m = (
    MARS_RADIUS_M
    * np.radians(latitude_resolution)
    * DOWNSAMPLE_FACTOR
)

print("=== MAP ===")
print("Original:", original_elevation.shape)
print("Downsampled:", H.shape)
print(f"North-south grid spacing: {dy_m:.2f} m")

print(
    "Load time:",
    round(time.perf_counter() - t0, 3),
    "s"
)

# ============================================================
# 3. TERRAIN ROUGHNESS
# ============================================================

def compute_roughness(H, window=5):

    if window < 3 or window % 2 == 0:
        raise ValueError("Window size must be odd and >= 3")

    r = window // 2
    n = window**2

    valid_cells = np.isfinite(H)

    if not valid_cells.any():
        raise ValueError("No valid elevation samples found")

    reference_elevation = np.mean(H[valid_cells])

    H0 = np.where(
        valid_cells, H - reference_elevation, 0.0
    )

    offsets = np.arange(-r, r + 1, dtype=float)
    X, Y = np.meshgrid(offsets, offsets)

    sx2 = np.sum(X**2)
    sy2 = np.sum(Y**2)

    local_mean = uniform_filter(
        H0, size=window,
        mode="constant", cval=0.0
    )

    mean_square = uniform_filter(
        H0**2, size=window,
        mode="constant", cval=0.0
    )

    a = correlate(
        H0, X, mode="constant", cval=0.0
    ) / sx2

    b = correlate(
        H0, Y, mode="constant", cval=0.0
    ) / sy2

    variance = (
        mean_square - local_mean**2
        - (a*a*sx2 + b*b*sy2) / n
    )

    rho = np.sqrt(
        np.maximum(variance, 0.0)
    )

    # Reject windows with missing elevation samples.
    coverage = uniform_filter(
        valid_cells.astype(float),
        size=window,
        mode="constant",
        cval=0.0
    )

    rho[coverage < 1.0 - 1e-8] = np.nan

    rho[:r, :] = np.nan
    rho[-r:, :] = np.nan
    rho[:, :r] = np.nan
    rho[:, -r:] = np.nan

    return rho


# ============================================================
# 4. SLOPE AND ROUGHNESS
# ============================================================

t1 = time.perf_counter()

rho = compute_roughness(
    H, ROUGHNESS_WINDOW
)

height_gradient_row, height_gradient_col = np.gradient(H)

height_gradient_x = height_gradient_col / row_dx_m[:, None]
height_gradient_y = -height_gradient_row / dy_m

theta = np.degrees(
    np.arctan(
        np.hypot(height_gradient_x, height_gradient_y)
    )
)

valid_elevation = np.isfinite(H)

valid_terrain = (
    valid_elevation
    & np.isfinite(rho)
    & (rho <= MAX_ROUGHNESS_M)
)

print(
    "Terrain processing time:",
    round(time.perf_counter() - t1, 3),
    "s"
)


# ============================================================
# 5. COORDINATE CONVERSION
# ============================================================

def latlon_to_grid(latitude, longitude):
    """Map geographic coordinates to the nearest sampled DEM cell."""
    fractional_col, fractional_row = (~transform) * (longitude, latitude)

    if not (0 <= fractional_row < original_elevation.shape[0]
            and 0 <= fractional_col < original_elevation.shape[1]):
        raise ValueError(
            f"Coordinates outside the GeoTIFF: ({latitude}, {longitude})"
        )

    # Snap coordinates to the nearest sampled DEM cell.
    row = int(np.floor((fractional_row - 0.5) / DOWNSAMPLE_FACTOR + 0.5))
    col = int(np.floor((fractional_col - 0.5) / DOWNSAMPLE_FACTOR + 0.5))
    row = int(np.clip(row, 0, num_rows - 1))
    col = int(np.clip(col, 0, num_cols - 1))
    return row, col


def grid_to_latlon(point):
    row, col = point
    lon, lat = rasterio.transform.xy(
        transform, row * DOWNSAMPLE_FACTOR, col * DOWNSAMPLE_FACTOR, offset="center"
    )
    return float(lat), float(lon)


for name, (lat, lon) in [("start", START_LATLON), ("destination", END_LATLON)]:
    if not (-90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0):
        raise ValueError(f"Invalid coordinates for {name}: {(lat, lon)}")

if POI_ORDER_MODE not in {"automatic", "manual"}:
    raise ValueError("POI_ORDER_MODE must be automatic or manual")
if not isinstance(SELECTED_POIS, (list, tuple)):
    raise ValueError("SELECTED_POIS must be a list, for example [1, 3]")
if len(set(SELECTED_POIS)) != len(SELECTED_POIS):
    raise ValueError("Do not select the same POI more than once")
if any(type(k) is not int or k not in POI_CATALOG for k in SELECTED_POIS):
    raise ValueError(f"Invalid POI. Available IDs: {sorted(POI_CATALOG)}")

waypoints = []
for poi_id in SELECTED_POIS:
    info = POI_CATALOG[poi_id]
    point = latlon_to_grid(*info["latlon"])
    waypoints.append(point)
    print(f"POI {poi_id}: {info['name']}, "
          f"requested={info['latlon']}, mapped={grid_to_latlon(point)}")

start = latlon_to_grid(*START_LATLON)
goal = latlon_to_grid(*END_LATLON)

print("\n=== ROUTE COORDINATES ===")
print("Requested start (lat, lon):", START_LATLON)
print("Requested destination (lat, lon):", END_LATLON)
print("A* start (lat, lon):", grid_to_latlon(start))
print("A* destination (lat, lon):", grid_to_latlon(goal))
print("Start grid cell:", start)
print("Destination grid cell:", goal)

# ============================================================
# 6. EDGE DISTANCE AND COSTS
# ============================================================

def edge_geometry(current, neighbor):

    i, j = current
    ni, nj = neighbor

    local_dx = 0.5 * (
        row_dx_m[i] + row_dx_m[ni]
    )

    dxy = np.hypot(
        (nj - j) * local_dx,
        (ni - i) * dy_m
    )

    dh = H[ni, nj] - H[i, j]

    d3 = np.hypot(dxy, dh)

    slope_deg = np.degrees(
        np.arctan2(abs(dh), dxy)
    )

    return dxy, d3, slope_deg


def elevation_cost(current, neighbor):

    _, d3, _ = edge_geometry(
        current, neighbor
    )

    return d3


def temperature_penalty(temperature):
    # Penalize temperatures below the experimental reference.
    return (max(0.0, REFERENCE_TEMPERATURE_K - temperature) / TEMPERATURE_SCALE_K)**2


def terrain_cost(current, neighbor):

    _, d3, slope_deg = edge_geometry(
        current, neighbor
    )

    if slope_deg > MAX_SLOPE_DEG:
        return np.inf

    S = (
        np.tan(np.radians(slope_deg))
        / np.tan(np.radians(MAX_SLOPE_DEG))
    )**2

    i, j = current
    ni, nj = neighbor

    local_roughness = 0.5 * (
        rho[i, j] + rho[ni, nj]
    )

    R = (
        local_roughness / MAX_ROUGHNESS_M
    )**2

    return d3 * (
        1
        + SLOPE_WEIGHT * S
        + ROUGHNESS_WEIGHT * R
    )


def terrain_temperature_cost(current, neighbor):
    # Preserve topographic constraints and add the thermal penalty.
    base_cost = terrain_cost(current, neighbor)
    if not np.isfinite(base_cost):
        return np.inf

    i, j = current
    ni, nj = neighbor
    local_temperature = 0.5 * (float(T[i, j]) + float(T[ni, nj]))
    Q = temperature_penalty(local_temperature)
    _, d3, _ = edge_geometry(current, neighbor)

    return base_cost + d3 * TEMPERATURE_WEIGHT * Q


# ============================================================
# 7. A* HEURISTIC
# ============================================================

dx_min = float(np.min(row_dx_m))

def heuristic(a, b, use_heuristic=True):
    if not use_heuristic:
        return 0.0

    return np.hypot(
        (a[1] - b[1]) * dx_min,
        (a[0] - b[0]) * dy_m
    )


# ============================================================
# 8. A* SEARCH
# ============================================================

def astar(start, goal, mask, cost_function):

    for point in (start, goal):

        i, j = point

        if not (
            0 <= i < num_rows
            and 0 <= j < num_cols
        ):
            raise ValueError("Point outside map")

        if not mask[i, j]:
            return None, np.inf, 0

    g = np.full(H.shape, np.inf)
    g[start] = 0.0

    closed = np.zeros(H.shape, dtype=bool)
    parents = {}

    counter = count()

    open_heap = [
        (
            heuristic(start, goal),
            next(counter),
            start
        )
    ]

    explored = 0

    while open_heap:

        _, _, current = heapq.heappop(open_heap)

        if closed[current]:
            continue

        if current == goal:

            route = [current]

            while current != start:
                current = parents[current]
                route.append(current)

            return route[::-1], g[goal], explored

        closed[current] = True
        explored += 1

        i, j = current

        for di, dj in NEIGHBOR_OFFSETS:

            ni, nj = i + di, j + dj

            if not (
                0 <= ni < num_rows
                and 0 <= nj < num_cols
            ):
                continue

            if not mask[ni, nj]:
                continue

            if closed[ni, nj]:
                continue

            if di != 0 and dj != 0:
                if not (
                    mask[i + di, j]
                    and mask[i, j + dj]
                ):
                    continue

            neighbor = (ni, nj)

            c = cost_function(current, neighbor)

            if not np.isfinite(c):
                continue

            tentative_g = g[current] + c

            if tentative_g < g[neighbor]:

                g[neighbor] = tentative_g
                parents[neighbor] = current

                heapq.heappush(
                    open_heap,
                    (
                        tentative_g + heuristic(
                            neighbor, goal, True
                        ),
                        next(counter),
                        neighbor
                    )
                )

    return None, np.inf, explored


# ============================================================
# 9. ROUTING WITH OPTIONAL WAYPOINTS
# ============================================================

def astar_with_waypoints(origin, waypoints, destination, mask, cost_function):
    """Join A* legs through optional POIs without duplicating endpoints."""
    targets = [origin] + list(waypoints) + [destination]
    full_route = []
    total_cost = 0.0
    total_explored = 0

    for k, (a, b) in enumerate(zip(targets[:-1], targets[1:]), 1):
        if a == b:
            segment, segment_cost, explored = [a], 0.0, 0
        else:
            segment, segment_cost, explored = astar(a, b, mask, cost_function)
        total_explored += explored
        if segment is None:
            print(f"  No route for segment {k}/{len(targets)-1}: "
                  f"{grid_to_latlon(a)} -> {grid_to_latlon(b)}")
            return None, np.inf, total_explored
        if full_route:
            full_route.extend(segment[1:])
        else:
            full_route.extend(segment)
        total_cost += segment_cost
        print(f"  Segment {k}/{len(targets)-1}")

    return full_route, total_cost, total_explored

def optimize_poi_order(start, waypoints, goal):
    """Find the minimum-cost POI order using pairwise A* and Held-Karp DP."""
    n = len(waypoints)
    if n <= 1:
        return list(range(n)), 0.0
    if n > MAX_OPTIMIZED_POIS:
        raise ValueError(
            f"You selected {n} POIs; exact optimization limited to "
            f"{MAX_OPTIMIZED_POIS} to limit runtime. "
            "Select fewer POIs or use POI_ORDER_MODE='manual'."
        )
    points = [start] + list(waypoints) + [goal]
    pair_costs = np.full((n + 2, n + 2), np.inf, dtype=float)
    np.fill_diagonal(pair_costs, 0.0)
    for i in range(n + 2):
        for j in range(i + 1, n + 2):
            if i == 0 and j == n + 1:
                continue
            if points[i] == points[j]:
                cost = 0.0
            else:
                _, cost, _ = astar(
                    points[i], points[j], valid_terrain, terrain_temperature_cost
                )
            pair_costs[i, j] = pair_costs[j, i] = cost
            print(f"  POI pair cost {i}/{j}: {cost:.1f}")

    # Held-Karp dynamic programming over selected POIs.
    dp = {}
    for j in range(n):
        if np.isfinite(pair_costs[0, j + 1]):
            dp[(1 << j, j)] = (pair_costs[0, j + 1], (j,))
    for mask in range(1, 1 << n):
        for j in range(n):
            state = dp.get((mask, j))
            if state is None:
                continue
            current_cost, order = state
            for k in range(n):
                if mask & (1 << k) or not np.isfinite(pair_costs[j + 1, k + 1]):
                    continue
                new_state = (mask | (1 << k), k)
                candidate = current_cost + pair_costs[j + 1, k + 1]
                previous = dp.get(new_state)
                if previous is None or candidate < previous[0]:
                    dp[new_state] = (candidate, order + (k,))
    full_mask = (1 << n) - 1
    best = None
    for j in range(n):
        state = dp.get((full_mask, j))
        if state is None or not np.isfinite(pair_costs[j + 1, n + 1]):
            continue
        total = state[0] + pair_costs[j + 1, n + 1]
        if best is None or total < best[0]:
            best = (total, state[1])
    if best is None:
        raise RuntimeError("No feasible POI order connects A and B under the full cost model")
    return list(best[1]), best[0]


print("\n=== SCIENTIFIC WAYPOINTS ===")
if not SELECTED_POIS:
    print("No POIs selected: direct A -> B route")
elif POI_ORDER_MODE == "automatic":
    print("Optimizing POI order using terrain and temperature costs...")
    order_indices, optimal_cost = optimize_poi_order(start, waypoints, goal)
    SELECTED_POIS = [SELECTED_POIS[i] for i in order_indices]
    waypoints = [waypoints[i] for i in order_indices]
    print(f"Minimum mission cost (full model): {optimal_cost:.2f}")
else:
    print("Using the requested manual order")
if SELECTED_POIS:
    print("Final POI order:", " -> ".join(
        f"{k}: {POI_CATALOG[k]['name']}" for k in SELECTED_POIS
    ))

print("\n=== A* H ===")
t2 = time.perf_counter()
elevation_route, elevation_total_cost, elevation_explored = astar_with_waypoints(
    start, waypoints, goal, valid_elevation, elevation_cost
)
elevation_elapsed = time.perf_counter() - t2
print("Time:", round(elevation_elapsed, 3), "s")
print("Explored nodes:", elevation_explored)

print("\n=== A* H + THETA + RHO ===")
t3 = time.perf_counter()
terrain_route, terrain_total_cost, terrain_explored = astar_with_waypoints(
    start, waypoints, goal, valid_terrain, terrain_cost
)
terrain_elapsed = time.perf_counter() - t3
print("Time:", round(terrain_elapsed, 3), "s")
print("Explored nodes:", terrain_explored)

print("\n=== A* H + THETA + RHO + TEMPERATURE ===")
t4 = time.perf_counter()
thermal_route, thermal_total_cost, thermal_explored = astar_with_waypoints(
    start, waypoints, goal, valid_terrain, terrain_temperature_cost
)
thermal_elapsed = time.perf_counter() - t4
print("Time:", round(thermal_elapsed, 3), "s")
print("Explored nodes:", thermal_explored)
print("Thermal A* time:", round(thermal_elapsed, 3), "s")

# ============================================================
# 9B. DIRECT RETURN TO BASE: B -> A
# ============================================================

# Return route uses the full topographic and thermal cost model.
return_route = None
return_cost = np.inf

if RETURN_MODE == "return":
    print("\n=== RETURN TO BASE: B -> A (TERRAIN + TEMPERATURE) ===")
    print("Departure (B):", grid_to_latlon(goal))
    print("Base (A):", grid_to_latlon(start))
    print("Return route skips all POI waypoints.")
    return_start_time = time.perf_counter()
    return_route, return_cost, return_explored = astar(
        goal, start, valid_terrain, terrain_temperature_cost
    )
    if return_route is None:
        print("No return route found under the current constraints.")
    else:
        print(f"Costo={return_cost:.2f}, nodos={len(return_route)}, "
              f"explored={return_explored}, "
              f"time={time.perf_counter()-return_start_time:.2f}s")


# ============================================================
# 10. ROUTE METRICS
# ============================================================

def measure_route(route):

    if route is None:
        return None

    distance = 0.0
    ascent = 0.0
    descent = 0.0
    max_slope = 0.0

    for a, b in zip(route[:-1], route[1:]):

        _, d3, slope_deg = edge_geometry(a, b)

        dh = H[b] - H[a]

        distance += d3
        ascent += max(dh, 0.0)
        descent += max(-dh, 0.0)

        max_slope = max(
            max_slope, slope_deg
        )

    return {
        "distance_km": distance / 1000,
        "ascent_m": ascent,
        "descent_m": descent,
        "max_slope_deg": max_slope
    }


elevation_metrics = measure_route(elevation_route)
terrain_metrics = measure_route(terrain_route)
thermal_metrics = measure_route(thermal_route)

if RETURN_MODE == "return":
    print("\n=== RETURN ROUTE METRICS ===")
    m = measure_route(return_route)
    if m is None:
        print("No return route found with terrain and temperature")
    else:
        print(f"Distance={m['distance_km']:.2f} km, "
              f"ascent={m['ascent_m']:.2f} m, "
              f"descent={m['descent_m']:.2f} m, "
              f"maximum slope={m['max_slope_deg']:.2f}°, "
              f"cost={return_cost:.2f}")

def mean_route_temperature(route):
    if route is None:
        return None
    points = np.asarray(route, dtype=int)
    return float(np.mean(T[points[:, 0], points[:, 1]]))

print("\n=== SURFACE TEMPERATURE ===")
print("Scenario:", season, time_of_day)
print("Elevation route mean (K):", mean_route_temperature(elevation_route))
print("Terrain route mean (K):", mean_route_temperature(terrain_route))
print("Thermal route mean (K):", mean_route_temperature(thermal_route))

print("\n=== ROUTE COMPARISON ===")

for name, metrics in [
    ("H", elevation_metrics),
    ("H + theta + rho", terrain_metrics),
    ("H + theta + rho + temperature", thermal_metrics)
]:
    if metrics is None:
        print(name, ": no route")
    else:
        print(name)
        print(
            "  Distance:",
            round(metrics["distance_km"], 2),
            "km"
        )
        print(
            "  Ascent:",
            round(metrics["ascent_m"], 2),
            "m"
        )
        print(
            "  Maximum slope:",
            round(metrics["max_slope_deg"], 2),
            "°"
        )

print("\nElevation A* time:", round(elevation_elapsed, 3), "s")
print("Terrain A* time:", round(terrain_elapsed, 3), "s")


# Export geographic route coordinates for the interface.
def export_route_csv(route, file_path):
    if route is None:
        return
    coords = np.array([grid_to_latlon(p) for p in route])
    np.savetxt(file_path, coords, delimiter=",", header="latitud,longitud",
               comments="", fmt="%.8f")
    print("Route exported:", file_path)

export_route_csv(elevation_route, "ruta_H.csv")
export_route_csv(terrain_route, "ruta_topografica.csv")
export_route_csv(thermal_route, "ruta_temperatura.csv")
if RETURN_MODE == "return":
    export_route_csv(return_route, "retorno_temperatura.csv")


# ============================================================
# 11. ROUTE PLOTTING
# ============================================================

grid_lon_centers = transform.c + (np.arange(num_cols) * DOWNSAMPLE_FACTOR + 0.5) * transform.a
grid_lat_centers = transform.f + (np.arange(num_rows) * DOWNSAMPLE_FACTOR + 0.5) * transform.e

longitude_step = DOWNSAMPLE_FACTOR * transform.a
latitude_step = abs(DOWNSAMPLE_FACTOR * transform.e)
geographic_extent = (
    grid_lon_centers[0] - longitude_step / 2,
    grid_lon_centers[-1] + longitude_step / 2,
    grid_lat_centers[-1] - latitude_step / 2,
    grid_lat_centers[0] + latitude_step / 2,
)


def route_coordinates(route):
    """Return geographic coordinates for a route."""
    r = np.asarray(route, dtype=int)
    return grid_lon_centers[r[:, 1]], grid_lat_centers[r[:, 0]]


def plot_routes(ax):
    if elevation_route is not None:
        lon, lat = route_coordinates(elevation_route)
        ax.plot(lon, lat, color="red", linewidth=2, label="A*: H")

    if terrain_route is not None:
        lon, lat = route_coordinates(terrain_route)
        ax.plot(lon, lat, color="cyan", linewidth=2,
                linestyle="--", label="A*: H + θ + ρ")

    if thermal_route is not None:
        lon, lat = route_coordinates(thermal_route)
        ax.plot(lon, lat, color="magenta", linewidth=2,
                linestyle=":", label="A*: H + θ + ρ + T")

    if RETURN_MODE == "return" and return_route is not None:
        return_lon, return_lat = route_coordinates(return_route)
        ax.plot(return_lon, return_lat, color="purple", linewidth=3,
                linestyle="-.", alpha=0.9,
                label="Return B → A: H + θ + ρ + T", zorder=5)

    for poi_id, point in zip(SELECTED_POIS, waypoints):
        poi_lon, poi_lat = grid_lon_centers[point[1]], grid_lat_centers[point[0]]
        ax.scatter(poi_lon, poi_lat, s=110, marker="*", color="yellow",
                   edgecolors="black", linewidths=0.8, zorder=7)
        ax.annotate(f"POI {poi_id}", (poi_lon, poi_lat),
                    xytext=(7, 7), textcoords="offset points",
                    fontsize=9, fontweight="bold", color="white",
                    bbox=dict(facecolor="black", alpha=0.6, pad=1.5), zorder=8)

    ax.scatter(grid_lon_centers[start[1]], grid_lat_centers[start[0]],
               c="lime", s=60, edgecolors="black", zorder=5,
               label="Base / Start")
    ax.scatter(grid_lon_centers[goal[1]], grid_lat_centers[goal[0]],
               c="blue", s=60, edgecolors="black", zorder=5,
               label="Destination")


# ============================================================
# 12. FOUR MAPS
# ============================================================

def save_map(data, title, cmap, colorbar_label, output_filename):
    """Plot route overlays and save a georeferenced map image."""
    fig, ax = plt.subplots(figsize=(14, 8))

    image = ax.imshow(
        data,
        cmap=cmap,
        origin="upper",
        interpolation="nearest",
        extent=geographic_extent,
        aspect="auto"
    )

    plot_routes(ax)
    ax.set_title(title)
    ax.set_xlabel("Longitude (°)")
    ax.set_ylabel("Latitude (°)")
    ax.legend(loc="best")

    fig.colorbar(
        image,
        ax=ax,
        orientation="horizontal",
        pad=0.10,
        fraction=0.045,
        aspect=40,
        label=colorbar_label
    )

    fig.tight_layout()
    fig.savefig(output_filename, dpi=200, bbox_inches="tight")
    print("Saved image:", output_filename)
    return fig


elevation_figure = save_map(
    H,
    "Elevation",
    "terrain",
    "Elevation (m)",
    "elevation_map.png"
)

slope_figure = save_map(
    theta,
    "Slope",
    "inferno",
    "Slope (°)",
    "slope_map.png"
)

roughness_figure = save_map(
    rho,
    "Roughness",
    "magma",
    "Roughness (m)",
    "roughness_map.png"
)

# Keep kelvin for routing; convert to Celsius only for display.
temperature_figure = save_map(
    T - 273.15,
    f"Temperature: {season}, {time_of_day}",
    "coolwarm",
    "Temperature (°C)",
    "temperature_map.png"
)

plt.show()
