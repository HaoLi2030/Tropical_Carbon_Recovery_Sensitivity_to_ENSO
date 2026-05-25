# Tropical_Carbon_Recovery_Sensitivity_to_ENSO
The project investigates the sensitivity of tropical carbon recovery to ENSO over seven decades using atmospheric CO2 observations and CMIP6 model evaluation.

## Research Overview
Tropical forests are a critical carbon sink that buffers atmospheric CO2 accumulation, yet whether this capacity is approaching tipping points remains debated. This study tests tropical carbon resilience by identifying 14 carbon recovery episodes---transitions from carbon loss (El Niño) to sustained carbon gain (La Niña)---in the atmospheric CO2 record. We find that carbon recovery sensitivity to ENSO has remained stable over seven decades, but its episode-to-episode variability is strongly constrained by antecedent El Niño intensity. Stronger preceding El Niño events trigger greater water stress and warmer temperatures, which impair subsequent ecosystem recovery. Most CMIP6 models fail to reproduce this antecedent constraint, revealing systematic deficiencies in simulating ecosystem responses to drought legacy effects.

## Data Sources

The analysis integrates multiple publicly available datasets:

### Observational Data
- **Atmospheric CO2**: NOAA Global Monitoring Laboratory (https://gml.noaa.gov/ccgg/trends/global.html)
- **Sea Surface Temperature**: ERA5 reanalysis (https://cds.climate.copernicus.eu/cdsapp#!/dataset/reanalysis-era5-single-levels-monthly-means)
- **Terrestrial Water Storage**: GRACE-REC reconstruction (https://doi.org/10.6084/m9.figshare.7670849)
- **Air Temperature**: CRU TS4.03 (https://www.uea.ac.uk/groups-and-centres/climatic-research-unit)

### Model Output
- **CMIP6 ESM Outputs**: Earth System Grid Federation (https://esgf-node.llnl.gov/search/cmip6)
  - 18 models evaluated from the historical experiment (1959--2014)
  - Variables: net biome production (NBP), sea surface temperature (SST)
