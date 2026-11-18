# Data sources

The notebook can download four official UCI archives automatically (`ALLOW_DOWNLOAD=True`). Raw datasets are not included in this repository or the supplied final output ZIP.

| Dataset | Role | Expected file | Official record |
| --- | --- | --- | --- |
| Air Quality | Core: hourly CO(GT), mg/m³ | `AirQualityUCI.csv` | https://archive.ics.uci.edu/dataset/360/air+quality |
| Appliances Energy Prediction | Core: ten-minute appliance use, Wh | `energydata_complete.csv` | https://archive.ics.uci.edu/dataset/374/appliances+energy+prediction |
| Beijing Multi-Site Air-Quality Data | External CO forecasting and twelve-station transfer | `PRSA_Data_*.csv` | https://archive.ics.uci.edu/dataset/501/beijing+multi+site+air+quality+data |
| Individual Household Electric Power Consumption | External: global active power, kW; aggregated to ten minutes | `household_power_consumption.txt` | https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption |

## Download URLs used by the source

- https://archive.ics.uci.edu/static/public/360/air+quality.zip
- https://archive.ics.uci.edu/static/public/374/appliances+energy+prediction.zip
- https://archive.ics.uci.edu/static/public/501/beijing+multi+site+air+quality+data.zip
- https://archive.ics.uci.edu/static/public/235/individual+household+electric+power+consumption.zip

These links were extracted from the supplied code; downloads were not repeated during repository preparation. Retain provider attribution and check each official record's current license before redistributing raw data.

## Offline inputs

Download and extract the official archives. Set `DATA_PATHS` and `EXTERNAL_DATA_PATHS` in the notebook, or `data_paths` and `external_data_paths` in a local configuration. Set `BEIJING_MULTISITE_ROOT` / `beijing_multisite_root` to the directory containing all twelve Beijing CSV files. Set `ALLOW_DOWNLOAD=False` / `allow_download=false` after configuring the local inputs.

Do not substitute a differently processed Kaggle copy without recording the change. The source decodes Air Quality sentinel values, retains natural missingness, builds predictors using available history, and aggregates Household Power to ten-minute means. Targets are not imputed. See `src/primna_esd_data.py`, `src/primna_esd_external.py`, and `src/primna_esd_multisite.py` for the exact processing and historical splits.
