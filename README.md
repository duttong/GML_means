# GML_means

The main program, <h3>gml_annualmeans.py</h3>, loads data from the NOAA/GML website (formerly the FTP site) and calculates annual global means. The file <h3>gml_config.yaml</h3> controls the list of gases and data sources for the yearly mean calculations. 

The loading of GML data is handled by the code in the <h3>NOAA_halocarbons_loader</h3> repository.

Results are stored in the <h3>gml_annual_means</h3> directory.
