from pathlib import Path
import numpy as np
import pandas as pd

# This script lives in src/; input and output data live in the project root.
BASE_DIR = Path(__file__).resolve().parent.parent
EXCEL_FILE = BASE_DIR / "Temps Marte y Sect1 REAL .xlsx"
OUTPUT_DIR = BASE_DIR / "temperaturas"

# Global Mars temperature sheets: four seasons, day and night.
SCENARIOS = {
    "otono_noche": "Temp Marte Ls 0° 0h",
    "otono_dia": "Temp Marte Ls 0° 12h",
    "invierno_noche": "Temp Marte Ls 90° 0h",
    "invierno_dia": "Temp Marte Ls 90° 12h",
    "primavera_noche": "Temp Marte Ls 180° 0h",
    "primavera_dia": "Temp Marte Ls 180° 12h",
    "verano_noche": "Temp Marte Ls 270° 0h",
    "verano_dia": "Temp Marte Ls 270° 12h",
}


def parse_coordinate(value):
    """Convert Excel coordinate labels such as '-90°' to floats."""
    return float(str(value).replace("°", "").strip())


def main():
    if not EXCEL_FILE.is_file():
        raise FileNotFoundError(f"Excel workbook not found: {EXCEL_FILE}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for scenario_name, sheet_name in SCENARIOS.items():
        # B7:AX71: header row contains latitudes; data rows contain longitudes.
        frame = pd.read_excel(
            EXCEL_FILE,
            sheet_name=sheet_name,
            usecols="B:AX",
            skiprows=6,
            nrows=64,
        )

        latitudes = np.array(
            [parse_coordinate(value) for value in frame.columns[1:]],
            dtype=np.float64,
        )
        longitudes = frame.iloc[:, 0].map(parse_coordinate).to_numpy(dtype=np.float64)

        values = frame.iloc[:, 1:].apply(
            pd.to_numeric, errors="coerce"
        ).to_numpy(dtype=np.float64)

        # Interpolation expects T[latitude_index, longitude_index].
        temperature = values.T

        if temperature.shape != (len(latitudes), len(longitudes)):
            raise ValueError(f"Unexpected temperature shape in {sheet_name}: {temperature.shape}")
        if not np.all(np.diff(latitudes) > 0):
            raise ValueError(f"Latitude coordinates must increase in {sheet_name}")
        if not np.all(np.diff(longitudes) > 0):
            raise ValueError(f"Longitude coordinates must increase in {sheet_name}")
        if not np.isfinite(temperature).all():
            raise ValueError(f"Invalid temperature values in {sheet_name}")

        output_file = OUTPUT_DIR / f"T_{scenario_name}.npz"
        np.savez_compressed(
            output_file,
            T=temperature,
            latitudes=latitudes,
            longitudes=longitudes,
        )
        print(
            f"{scenario_name}: shape={temperature.shape}, "
            f"range={temperature.min():.1f} to {temperature.max():.1f} K, "
            f"saved={output_file}"
        )

    print("\nConversion complete: eight temperature scenarios exported.")


if __name__ == "__main__":
    main()
