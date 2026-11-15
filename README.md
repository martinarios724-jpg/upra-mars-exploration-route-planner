# upra-mars-exploration-route-planner
Mars exploration route planning using A*, scientific terrain data, and a wearable Raspberry Pi prototype.

# Mars Exploration Route Planner

**NASA Space Apps Challenge 2026**

**Challenge:** Interplanetary Survival Guide: Martian Map  
**Team:** UPRA SEA RACCOON TEAM

**Team Members:**

Kevin Eduardo Reyes Luna.
Cesar Raúl González González.
Josue David Rodriguez Cortes.
Martín Alejandro Ríos Juárez.
Kevin Othoniel Araiza Rodriguez.
Carlos Omar Medina Rodriguez.

## Project Purpose

This project demonstrates how scientific data, computational route planning, and wearable hardware concepts can be combined to support future planetary exploration planning.

It is intended as a research and educational prototype, not an operational navigation system.

## Overview

Mars Exploration Route Planner is a prototype decision-support system designed to assist with regional exploration planning on Mars.

The project combines scientific terrain and temperature datasets with the A* pathfinding algorithm to calculate and compare routes between selected locations.

Users can select scientific points of interest (POIs), optimize their visiting order, and calculate a direct return route to the starting base.

The project also includes a wearable Raspberry Pi prototype.

## The Problem

Planning exploration routes on Mars involves more than finding the shortest distance between two locations.

Terrain elevation, slope, surface roughness, and temperature can affect route selection. Scientific exploration may also require visiting multiple locations before reaching a final destination.

Our project explores how geographic datasets and pathfinding algorithms can support these planning decisions.

## Our Solution

The system provides the following capabilities:

- **A* Route Planning:** Calculates routes using three different cost models.
- **Terrain Analysis:** Derives slope and roughness from elevation data.
- **Temperature Integration:** Incorporates seasonal and day/night surface temperature scenarios.
- **Scientific Points of Interest:** Supports selecting multiple documented Martian geological locations.
- **POI Order Optimization:** Finds a minimum-cost visiting order using the terrain and temperature cost model.
- **Return-to-Base Planning:** Calculates a direct route from destination B back to base A.
- **Route Visualization:** Generates geographic maps and animated route visualizations.
- **CSV Export:** Exports calculated paths as geographic coordinates.

## Scientific Data

### Elevation

The project uses a combined HRSC/MOLA digital elevation model (DEM) of Mars.

The original elevation dataset is processed into a NumPy array and downsampled by a factor of 32 for regional path planning.

### Terrain Slope

Slope is calculated from elevation differences and geographic distances between neighboring grid cells.

### Surface Roughness

Surface roughness is estimated using the root-mean-square residual between elevation samples and a locally fitted plane.

### Temperature

Surface temperature scenarios are obtained from the Mars Climate Database (MCD).

Eight scenarios are considered, representing four seasons and two local times (day and night).

Temperature values are interpolated onto the reduced DEM grid.

### Scientific Points of Interest

The project includes a catalog of Martian geological features, including craters and other formations.

POI coordinates are based on planetary nomenclature references and are used as geographic targets, not as validated landing or sampling locations.

## Route Planning Algorithm

The system uses the A* search algorithm:

**f(n) = g(n) + h(n)**

Where:

- **g(n):** Accumulated route cost from the starting point.
- **h(n):** Estimated remaining cost to the destination.
- **f(n):** Priority used to explore candidate nodes.

Three cost models are implemented:

| Model | Considered Factors |
|---|---|
| Distance-based | Three-dimensional distance |
| Terrain-aware | Distance, slope, and roughness |
| Terrain and temperature | Distance, slope, roughness, and surface temperature |

The terrain-aware models apply experimental slope and roughness restrictions.

The temperature-aware model adds a penalty for temperatures below a configurable reference value.

These parameters are experimental and have not been validated as astronaut safety thresholds.

## Data Preparation

The planner requires a preprocessed Martian DEM and aligned temperature arrays.

The preprocessing workflow consists of:

1. Preparing the Mars elevation DEM as `H_completo.npy`.
2. Converting temperature Excel worksheets into `.npz` files.
3. Interpolating temperature scenarios onto the reduced DEM grid.
4. Running the mission planner.

The scientific datasets are not included directly in this repository due to their size.

**Dataset download and preparation instructions:** [Add verified data links and instructions]

## Usage

Run the mission planner from the repository root:

```bash
python src/mars_mission_planner.py
```

Configure the starting location, destination, selected POIs, and temperature scenario in the Python configuration section.

The program generates geographic route files and four map visualizations:

- Elevation map
- Slope map
- Roughness map
- Temperature map

To animate the calculated routes:

```bash
python src/animate_mars_routes.py --folder .
```

A return-route CSV must be generated beforehand to display the full outbound-and-return mission animation.

## Hardware Prototype

TODO

## Limitations

This project is an experimental regional planning prototype.

- The reduced elevation grid has approximately 6.4 km north–south spacing between sampled cells.
- Small rocks, local obstacles, and pedestrian-scale hazards cannot be reliably detected.
- Temperature scenarios are climatological estimates rather than live weather measurements.
- Route cost weights and terrain restrictions are experimental.
- Geological feature centers are not validated access or sampling sites.
- The system does not guarantee astronaut safety or physical traversability.

## Technologies

- Python
- NumPy
- SciPy
- Matplotlib
- Rasterio
- Pandas
- OpenPyXL
- Pillow
- Raspberry Pi 4 Model B

## Data Sources and References

### 1. Mars Elevation Data — HRSC/MOLA Blended DEM

**Source:** U.S. Geological Survey (USGS) Astrogeology Science Center  
**Dataset:** Mars MGS MOLA – MEX HRSC Blended DEM Global 200m, Version 2

**Link:** https://astrogeology.usgs.gov/search/map/mars_mgs_mola_mex_hrsc_blended_dem_global_200m

This digital elevation model combines data from NASA's Mars Orbiter Laser Altimeter (MOLA), aboard Mars Global Surveyor, and the High Resolution Stereo Camera (HRSC), aboard ESA's Mars Express spacecraft.

The dataset provides the elevation information used to calculate terrain slope, surface roughness, and three-dimensional route distances.

For computational efficiency, our project uses a cropped region of the DEM and downsamples it by a factor of 32.

### 2. Mars Climate Database — Surface Temperature

**Source:** Mars Climate Database (MCD)

**Link:** https://www-mars.lmd.jussieu.fr/mcd_python/

Surface temperature data were obtained through the Mars Climate Database web interface.

Our project processes eight temperature scenarios representing four Martian seasons and two local-time settings.

These datasets are converted into NumPy arrays and spatially interpolated onto the reduced elevation grid.

The temperature layer is used to introduce an experimental thermal penalty into the A* route cost model.

### 3. USGS Interactive Maps of Mars

**Source:** USGS Astrogeology Science Center

**Link:** https://astrogeology-usgs.hub.arcgis.com/pages/interactive-maps-mars

The USGS interactive Mars maps provide a geographic reference for exploring Martian surface features and contextualizing the study region.

### 4. Scientific Points of Interest

**Reference:** USGS Gazetteer of Planetary Nomenclature

**Link:** https://planetarynames.wr.usgs.gov/

The project uses named Martian geological formations as scientific points of interest (POIs).

Their catalog coordinates are treated as geographic reference points rather than validated landing sites or safe sampling locations.

### Data Limitations

The elevation and temperature datasets have different original spatial resolutions. Temperature interpolation does not create additional scientific detail.

The reduced elevation grid is suitable for experimental regional route planning, not pedestrian-scale navigation.

Route restrictions and cost weights are experimental and have not been validated as astronaut safety criteria.


## Artificial Intelligence Disclosure

ChatGPT was used to assist with Python code development and debugging.

The project team is responsible for reviewing, testing, and validating the resulting software and scientific interpretations.
