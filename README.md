# GML_means

<strong>GML_means</strong> generates annual mean mole fractions for NOAA Global Monitoring Laboratory (GML) halocarbon and related trace gas measurements.

The main program, <strong>gml_annualmeans.py</strong>, loads data from the NOAA/GML website (formerly the FTP site), calculates semi-hemispheric annual means, and writes global annual mean products. The file <strong>gml_config.yaml</strong> controls the list of gases, source selection, metadata headers, and background site choices used in those calculations.

## What this repository does

This repository:

- loads NOAA/GML data for the gases listed in <strong>gml_config.yaml</strong>
- calculates annual means for four semi-hemispheric regions
- HN corresponds to latitudes greater than or equal to 30N
- LN corresponds to latitudes from 0 to less than 30N
- LS corresponds to latitudes south of the equator to greater than -30
- HS corresponds to latitudes less than or equal to -30
- calculates the global mean as the average of the four semi-hemispheric means
- writes per-gas annual mean CSV files
- writes two combined files — <strong>GML_jan_annual_means.csv</strong> and <strong>GML_jul_annual_means.csv</strong> — containing global means for all configured gases
- creates figures for each gas in <strong>gml_annual_means/figures</strong>

## Data sources and processing

The code uses two NOAA/GML source groupings defined in <strong>gml_config.yaml</strong>:

- <strong>combined</strong>: used for gases with combined ECD and MSD products
- <strong>msd</strong>: used for gases loaded from MSD-based products

For combined products, the code reshapes the downloaded data to match the site-based structure used in the rest of the processing. Annual means are computed from monthly values and only retained when all 12 months are present in a yearly window.

Two annual-mean windows are generated for each gas. The file suffix refers to the <em>center month</em> of the averaging window:

- <strong>jan</strong>: July 1 of year Y−1 to June 30 of year Y, labeled with year Y (January is the center month)
- <strong>jul</strong>: January 1 to December 31 of year Y, labeled with year Y (July is the center month)

The combined files <strong>GML_jan_annual_means.csv</strong> and <strong>GML_jul_annual_means.csv</strong> contain the global means for all configured gases in the respective windows.

## Repository layout

Key files and directories:

- <strong>gml_annualmeans.py</strong>: main processing script
- <strong>gml_config.yaml</strong>: gas list, source mapping, background sites, and output header templates
- <strong>gml_annual_means/data_files</strong>: per-gas yearly output CSV files
- <strong>gml_annual_means/figures</strong>: generated figures for each gas
- <strong>gml_annual_means/GML_jan_annual_means.csv</strong>: combined global means, Jul–Jun windows
- <strong>gml_annual_means/GML_jul_annual_means.csv</strong>: combined global means, Jan–Dec windows

The loading of GML data is handled by the code in the <strong>NOAA_halocarbons_loader</strong> repository:

<https://github.com/duttong/NOAA_halocarbons_loader>

## NOAA_halocarbons_loader dependency

This repository expects a local checkout of <strong>NOAA_halocarbons_loader</strong> in the top level of this project directory.

Example setup:

```bash
cd GML_means
git clone https://github.com/duttong/NOAA_halocarbons_loader.git
```

After cloning, the directory structure should look like:

```text
GML_means/
├── gml_annualmeans.py
├── gml_config.yaml
└── NOAA_halocarbons_loader/
```

Once that repository is present, run:

```bash
./gml_annualmeans.py              # full run: every gas listed in gml_config.yaml
./gml_annualmeans.py HFC236fa     # single-gas shortcut: skips the config list
```

The script can be invoked from any directory; all output paths are resolved relative to this repository, not the current working directory.

The single-gas shortcut writes the per-gas CSVs and figure for just that gas. It does <em>not</em> touch <strong>GML_jan_annual_means.csv</strong> or <strong>GML_jul_annual_means.csv</strong>, since those files are snapshots of the full config list — partially overwriting them from a single-gas call would corrupt that semantic. Use the full run to regenerate them.

## Configuration

The main configuration file is <strong>gml_config.yaml</strong>. It defines:

- the gases processed by the script
- which gases use <strong>combined</strong> versus <strong>msd</strong> loading
- the default background sites used in the semi-hemispheric means
- gas-specific background site overrides
- header text written into the output CSV products

If you want to add or remove gases, or change the site selection used in the averaging, update <strong>gml_config.yaml</strong>.

## Output products

Running <strong>python gml_annualmeans.py</strong> writes results into <strong>gml_annual_means</strong>:

- <strong>data_files/{gas}_jan_yearly.csv</strong>
- <strong>data_files/{gas}_jul_yearly.csv</strong>
- <strong>figures/{gas}_annual_means.png</strong>
- <strong>GML_jan_annual_means.csv</strong>
- <strong>GML_jul_annual_means.csv</strong>

The per-gas CSV files contain:

- year
- HN
- LN
- LS
- HS
- Global

## Notes

- The script excludes the current calendar year by default because the data may not yet be fully quality controlled.
- Figures show background-site observations together with the annual mean time series.
- The software license information is provided in <strong>LICENSE.md</strong>.
