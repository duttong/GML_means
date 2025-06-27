# GML_means

The main program, <strong>gml_annualmeans.py</strong>, loads data from the NOAA/GML website (formerly the FTP site) and calculates annual global means. The file <strong>gml_config.yaml</strong> controls the list of gases and data sources for the yearly mean calculations. 

The loading of GML data is handled by the code in the <strong>NOAA_halocarbons_loader</strong> repository.

Results are stored in the <strong>gml_annual_means</strong> directory.
