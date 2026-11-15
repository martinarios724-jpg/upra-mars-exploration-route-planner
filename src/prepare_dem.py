"""Convert a Mars DEM GeoTIFF into a memory-mapped NumPy elevation array."""

import time
from pathlib import Path
import numpy as np
import rasterio
from numpy.lib.format import open_memmap
from rasterio.windows import Window

# Input and output files are located in the repository root.
BASE_DIR = Path(__file__).resolve().parent.parent
DEM_FILE = BASE_DIR / "SECTOR_MARTE_200M.tif"
ELEVATION_FILE = BASE_DIR / "H_completo.npy"
METADATA_FILE = BASE_DIR / "metadata_H.npz"
BLOCK_SIZE = 512


def main():
    start_time = time.perf_counter()

    if not DEM_FILE.is_file():
        raise FileNotFoundError(f"DEM file not found: {DEM_FILE}")

    with rasterio.open(DEM_FILE) as source:
        rows, columns = source.height, source.width

        print("=== ORIGINAL DEM ===")
        print("Dimensions:", source.shape)
        print("Resolution:", source.res)
        print("CRS:", source.crs)
        print("Data type:", source.dtypes[0])
        print("NoData:", source.nodata)

        if source.crs is None:
            raise ValueError("The DEM has no coordinate reference system (CRS).")

        # Create an on-disk NumPy array without loading the full DEM into memory.
        elevation = open_memmap(
            ELEVATION_FILE,
            mode="w+",
            dtype=np.dtype(source.dtypes[0]),
            shape=(rows, columns),
        )

        # Copy elevation values in blocks to limit memory usage.
        for row in range(0, rows, BLOCK_SIZE):
            block_height = min(BLOCK_SIZE, rows - row)

            for column in range(0, columns, BLOCK_SIZE):
                block_width = min(BLOCK_SIZE, columns - column)
                window = Window(
                    col_off=column,
                    row_off=row,
                    width=block_width,
                    height=block_height,
                )
                block = source.read(1, window=window)
                elevation[
                    row : row + block_height,
                    column : column + block_width,
                ] = block

            print(f"Processed {row + block_height} of {rows} rows")

        elevation.flush()
        del elevation

        # Preserve georeferencing information for subsequent processing.
        np.savez(
            METADATA_FILE,
            transform=np.array(tuple(source.transform)),
            crs=source.crs.to_wkt(),
            nodata=source.nodata,
        )

    print("\n=== OUTPUT VERIFICATION ===")
    elevation = np.load(ELEVATION_FILE, mmap_mode="r")
    print("Dimensions:", elevation.shape)
    print("Data type:", elevation.dtype)
    print("Total time:", round(time.perf_counter() - start_time, 2), "seconds")
    print("\nGenerated files:")
    print(f" - {ELEVATION_FILE.name}")
    print(f" - {METADATA_FILE.name}")


if __name__ == "__main__":
    main()
