from pathlib import Path
import numpy as np
import rasterio
from scipy.interpolate import RegularGridInterpolator

# Configuración: ejecutar desde la carpeta DATA.
BASE = Path(__file__).resolve().parent
ARCHIVO_DEM = BASE / "SECTOR_MARTE_200M.tif"
ENTRADA = BASE / "temperaturas"
SALIDA = BASE / "temperaturas_alineadas"
FACTOR = 32

ESCENARIOS = [
    f"T_{estacion}_{periodo}"
    for estacion in ("otono", "invierno", "primavera", "verano")
    for periodo in ("dia", "noche")
]


def coordenadas_dem():
    with rasterio.open(ARCHIVO_DEM) as src:
        if src.crs is None or not src.crs.is_geographic:
            raise ValueError("El GeoTIFF debe tener un CRS geográfico marciano.")
        tr = src.transform
        if abs(tr.b) > 1e-12 or abs(tr.d) > 1e-12:
            raise ValueError("Este programa requiere un GeoTIFF sin rotación.")
        if tr.a <= 0 or tr.e >= 0:
            raise ValueError("Se espera un ráster orientado al norte.")
        filas = np.arange(0, src.height, FACTOR)
        columnas = np.arange(0, src.width, FACTOR)
        # Centros de los mismos píxeles muestreados por H_original[::FACTOR, ::FACTOR].
        lat = tr.f + (filas + 0.5) * tr.e
        lon = tr.c + (columnas + 0.5) * tr.a
        return lat, lon


def preparar_eje(eje, matriz, axis, nombre):
    eje = np.asarray(eje, dtype=np.float64)
    if not np.all(np.isfinite(eje)):
        raise ValueError(f"{nombre}: coordenadas no finitas.")
    indices = np.argsort(eje)
    eje = eje[indices]
    matriz = np.take(matriz, indices, axis=axis)
    if not np.all(np.diff(eje) > 0):
        raise ValueError(f"{nombre}: coordenadas duplicadas o no crecientes.")
    return eje, matriz


def main():
    SALIDA.mkdir(exist_ok=True)
    lat_dem, lon_dem = coordenadas_dem()
    forma = (len(lat_dem), len(lon_dem))
    print("DEM reducido:", forma)
    print(f"Cobertura DEM: lat {lat_dem.min():.4f} a {lat_dem.max():.4f}; "
          f"lon {lon_dem.min():.4f} a {lon_dem.max():.4f}")

    archivos = [ENTRADA / f"{nombre}.npz" for nombre in ESCENARIOS]
    faltantes = [str(p) for p in archivos if not p.is_file()]
    if faltantes:
        raise FileNotFoundError("Faltan archivos de temperatura:\n" + "\n".join(faltantes))

    # La malla de consulta se genera una vez para los ocho escenarios.
    lon_malla, lat_malla = np.meshgrid(lon_dem, lat_dem)
    consultas = np.column_stack((lat_malla.ravel(), lon_malla.ravel()))

    for archivo in archivos:
        with np.load(archivo) as datos:
            T = np.asarray(datos["T"], dtype=np.float64)
            lat = np.asarray(datos["latitudes"], dtype=np.float64)
            lon = np.asarray(datos["longitudes"], dtype=np.float64)

        if T.shape != (len(lat), len(lon)):
            raise ValueError(f"{archivo.name}: T tiene forma {T.shape}, "
                             f"pero se esperaban {(len(lat), len(lon))}.")
        if not np.all(np.isfinite(T)):
            raise ValueError(f"{archivo.name}: contiene temperaturas no finitas.")

        lat, T = preparar_eje(lat, T, 0, "latitudes")
        lon, T = preparar_eje(lon, T, 1, "longitudes")

        # Evitar extrapolación fuera de la cuadrícula climática.
        if (lat_dem.min() < lat.min() or lat_dem.max() > lat.max()
                or lon_dem.min() < lon.min() or lon_dem.max() > lon.max()):
            raise ValueError(
                f"{archivo.name}: el mapa térmico no cubre todo el DEM. "
                f"Lat T=[{lat.min()}, {lat.max()}], "
                f"Lon T=[{lon.min()}, {lon.max()}]."
            )

        interpolador = RegularGridInterpolator(
            (lat, lon), T, method="linear",
            bounds_error=False, fill_value=np.nan
        )
        alineada = interpolador(consultas).reshape(forma).astype(np.float32)
        invalidos = np.count_nonzero(~np.isfinite(alineada))
        if invalidos:
            raise ValueError(f"{archivo.name}: {invalidos} celdas sin temperatura.")

        destino = SALIDA / f"{archivo.stem}_alineada.npy"
        np.save(destino, alineada)
        print(f"{destino.name}: {alineada.shape}; "
              f"mín={alineada.min():.2f} K; máx={alineada.max():.2f} K; "
              f"NaN={invalidos}")

    print("\nListo: ocho escenarios alineados con FACTOR=32.")


if __name__ == "__main__":
    main()
