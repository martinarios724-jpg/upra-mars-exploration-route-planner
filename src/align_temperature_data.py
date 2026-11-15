from pathlib import Path
import numpy as np
import rasterio
from scipy.interpolate import RegularGridInterpolator

# Input data and output grid must use the same sampling factor as the route planner.
BASE_DIR = Path(__file__).resolve().parent
DEM_FILE = BASE_DIR / "SECTOR_MARTE_200M.tif"
INPUT_DIR = BASE_DIR / "temperaturas"
OUTPUT_DIR = BASE_DIR / "temperaturas_alineadas"
DOWNSAMPLE_FACTOR = 32

SCENARIOS = [
    f"T_{season}_{period}"
    for season in ("otono", "invierno", "primavera", "verano")
    for period in ("dia", "noche")
]

def get_dem_coordinates():
    """Return the geographic centers of the DEM pixels sampled by the planner."""
    with rasterio.open(DEM_FILE) as source:
        if source.crs is None or not source.crs.is_geographic:
            raise ValueError("The GeoTIFF must use a geographic Martian CRS.")
        transform = source.transform
        if abs(transform.b) > 1e-12 or abs(transform.d) > 1e-12:
            raise ValueError("The GeoTIFF must not be rotated.")
        if transform.a <= 0 or transform.e >= 0:
            raise ValueError("A north-up raster is required.")
        rows = np.arange(0, source.height, DOWNSAMPLE_FACTOR)
        columns = np.arange(0, source.width, DOWNSAMPLE_FACTOR)
        latitudes = transform.f + (rows + 0.5) * transform.e
        longitudes = transform.c + (columns + 0.5) * transform.a
        return latitudes, longitudes


def sort_grid_axis(coordinates, values, axis, axis_name):
    """Sort an interpolation axis and reorder its corresponding temperature data."""
    coordinates = np.asarray(coordinates, dtype=np.float64)
    if not np.all(np.isfinite(coordinates)):
        raise ValueError(f"{axis_name}: non-finite coordinates.")
    indices = np.argsort(coordinates)
    coordinates = coordinates[indices]
    values = np.take(values, indices, axis=axis)
    if not np.all(np.diff(coordinates) > 0):
        raise ValueError(f"{axis_name}: duplicate or non-increasing coordinates.")
    return coordinates, values


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    dem_latitudes, dem_longitudes = get_dem_coordinates()
    output_shape = (len(dem_latitudes), len(dem_longitudes))
    print("Downsampled DEM shape:", output_shape)
    print(
        f"DEM coverage: latitude {dem_latitudes.min():.4f} to "
        f"{dem_latitudes.max():.4f}; longitude {dem_longitudes.min():.4f} "
        f"to {dem_longitudes.max():.4f}"
    )

    input_files = [INPUT_DIR / f"{scenario}.npz" for scenario in SCENARIOS]
    missing_files = [str(path) for path in input_files if not path.is_file()]
    if missing_files:
        raise FileNotFoundError("Missing temperature files:\n" + "\n".join(missing_files))

    longitude_grid, latitude_grid = np.meshgrid(dem_longitudes, dem_latitudes)
    query_points = np.column_stack((latitude_grid.ravel(), longitude_grid.ravel()))

    for input_file in input_files:
        with np.load(input_file) as data:
            temperature = np.asarray(data["T"], dtype=np.float64)
            latitudes = np.asarray(data["latitudes"], dtype=np.float64)
            longitudes = np.asarray(data["longitudes"], dtype=np.float64)

        expected_shape = (len(latitudes), len(longitudes))
        if temperature.shape != expected_shape:
            raise ValueError(
                f"{input_file.name}: temperature shape {temperature.shape}; "
                f"expected {expected_shape}."
            )
        if not np.all(np.isfinite(temperature)):
            raise ValueError(f"{input_file.name}: non-finite temperatures.")

        latitudes, temperature = sort_grid_axis(latitudes, temperature, 0, "latitudes")
        longitudes, temperature = sort_grid_axis(longitudes, temperature, 1, "longitudes")

        # Reject scenarios that would require extrapolation beyond climate data coverage.
        if (
            dem_latitudes.min() < latitudes.min()
            or dem_latitudes.max() > latitudes.max()
            or dem_longitudes.min() < longitudes.min()
            or dem_longitudes.max() > longitudes.max()
        ):
            raise ValueError(
                f"{input_file.name}: temperature grid does not cover the DEM. "
                f"Temperature latitude=[{latitudes.min()}, {latitudes.max()}], "
                f"longitude=[{longitudes.min()}, {longitudes.max()}]."
            )

        interpolator = RegularGridInterpolator(
            (latitudes, longitudes),
            temperature,
            method="linear",
            bounds_error=False,
            fill_value=np.nan,
        )
        aligned_temperature = interpolator(query_points).reshape(output_shape).astype(np.float32)
        invalid_count = int(np.count_nonzero(~np.isfinite(aligned_temperature)))
        if invalid_count:
            raise ValueError(f"{input_file.name}: {invalid_count} cells have no temperature.")

        output_file = OUTPUT_DIR / f"{input_file.stem}_alineada.npy"
        np.save(output_file, aligned_temperature)
        print(
            f"{output_file.name}: {aligned_temperature.shape}; "
            f"min={aligned_temperature.min():.2f} K; "
            f"max={aligned_temperature.max():.2f} K; invalid={invalid_count}"
        )

    print("\nDone: eight scenarios aligned using DOWNSAMPLE_FACTOR=32.")


if __name__ == "__main__":
    main()
