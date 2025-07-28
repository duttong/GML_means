#! /usr/bin/env python

import pandas as pd
import numpy as np
import yaml
import sys
from pathlib import Path
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D

here = Path.cwd()
loader_path = (here / "NOAA_halocarbons_loader").resolve()
sys.path.insert(0, str(loader_path.parent))  # Add parent directory of oa2026

#sys.path.append('NOAA_halocarbons_loader')
import NOAA_halocarbons_loader.halocarbons_loader as halocarbons_loader


class GMLAnnualMeans:
    def __init__(self):
        config_path = Path(__file__).parent / "gml_config.yaml"
        self.config_path = Path(config_path)
        self._load_config()
        self.gml = halocarbons_loader.HATS_Loader()

    def _load_config(self):
        """ parses the config yaml file """
        with open(self.config_path) as f:
            self.config = yaml.safe_load(f)
        self.header_tpl = self.config["annual_means_file_header"]
        self.gml_header_tpl = self.config["GML_means_file_header"]
        self.gases = self.config["gases"]
        self.sources = self.config["data_source"]
        self.combo_gases = self.sources.get("combined", [])
        self.msd_gases = self.sources.get("msd", [])
        self.default_bk_sites = self.config["background_sites"]
        self.bk_overrides = self.config.get("gas_background_overrides", {})

    def refactor_combo_df(self, df):
        """ The combined datafiles are structured differently than flask data files.
            This method refactors the combinded file to be similar to flask files.
        """
        df2 = df.copy()
        df2.index.name = 'date'
        df2 = df2.reset_index()

        to_drop = ['NH', 'NH_sd', 'SH', 'SH_sd', 'Global', 'Global_sd', 'Programs']
        df2 = df2.drop(columns=to_drop)

        mf_cols = [c for c in df2.columns if not c.endswith('_sd') and c != 'date']
        sd_cols = [f"{c}_sd" for c in mf_cols]

        mf_long = df2.melt(id_vars='date', value_vars=mf_cols, var_name='site', value_name='mf')
        sd_long = df2.melt(id_vars='date', value_vars=sd_cols, var_name='site', value_name='sd')
        sd_long['site'] = sd_long['site'].str[:-3]

        long_df = pd.merge(mf_long, sd_long, on=['date', 'site'])
        long_df = long_df.sort_values(['date', 'site']).reset_index(drop=True)
        long_df = self.gml.add_location(long_df)
        long_df.reset_index(inplace=True)

        return long_df

    def semi_hemispheric_means(self, df, gas, sites_included=None, phi=30.0):
        """
        Compute cosine‐latitude weighted monthly means for four semi‐hemispheres
        and the global mean (average of the four).

        Semi‐hemispheres:
            HN: lat ≥  phi
            LN: 0 ≤ lat < phi
            LS: −phi < lat < 0
            HS: lat ≤ −phi

        Parameters
        ----------
        df : pandas.DataFrame
            Must contain columns ['site','date','lat','mf'].
        gas: str
            Used for gas and site specific weights. Namely for HFCs PSA is deweighted (Steve's method)
        sites_included : list of str, optional
            If given, only rows where df['site'] is in this list are used.
        phi : float, optional
            Latitude cutoff (in degrees) between “low” and “high” zones.

        Returns
        -------
        semi_means : pandas.DataFrame
            Index = date  
            Columns = ['HN','LN','LS','HS','Global']  
            Values = weighted mean of mf for each region, and Global = mean of the four.
        """
        df = df.copy()
        df.reset_index(inplace=True)

        if sites_included is not None:
            df = df[df['site'].isin(sites_included)]

        # Adjust weights for specific sites
        df.loc[df['site'] == 'spo', 'lat'] = -65.0
        # Deweight psa for HFCs and CH2Cl2
        if gas in ['CH2Cl2', 'HFC32', 'HFC125', 'HFC134a', 'HFC143a', 'HFC152a', 
                    'HFC227ea', 'HFC365mfc', 'HFC236fa']:
            df.loc[df['site'] == 'psa', 'lat'] = -80.0

        # Define regions based on latitude
        def _bucket(lat):
            if lat >= phi:
                return 'HN'
            elif lat >= 0:
                return 'LN'
            elif lat <= -phi:
                return 'HS'
            else:
                return 'LS'

        df['region'] = df['lat'].apply(_bucket)

        # Precompute fixed weights for all sites
        df['fixed_w'] = np.cos(np.deg2rad(df['lat']))
        
        # drop rows with missing mf so the weighted mean calculated
        df2 = df.dropna(subset=['mf']).copy()
        
        # compute weighted mean per (date,region)
        w = (df2['mf'] * df2['fixed_w']).rename('mf_sum')
        df2 = df2.assign(mf_sum=w)

        wm = ( df2
            .groupby(['date','region'])[['mf', 'mf_sum', 'fixed_w']]
            .apply(lambda g: g['mf_sum'].sum() / g['fixed_w'].sum())
            .unstack('region')
        )
        
        # now wm is a DataFrame with columns ['HN','LN','LS','HS']
        # Calculate Global mean using only valid values from all four regions
        wm['Global'] = wm[['HN', 'LN', 'LS', 'HS']].mean(axis=1, skipna=False)

        return wm

    def annual_means(self, gas, end_year=None, save_file=False):
        """
            Calculate annual means from NOAA/GML data available through GML's website (and legacy FTP site).
            The annual means are computed for two different periods:
            - January to January (starting from January of the first year to January of the next year)
            - July to July (starting from July of the first year to July of the next year)
            The results are saved as CSV files in the 'gml_annual_means' directory.
            Parameters
            ----------
            gas : str
                The name of the gas for which to calculate annual means (e.g., 'CH3br').
            end_year : int, optional
                The last year to include in the annual means. If None, defaults to the previous year.
            Returns
            -------
            df : pandas.DataFrame or None
                DataFrame containing the annual means for the specified gas, or None if no data is found.
            Notes
            -----
            - The function uses the NOAA halocarbons loader to fetch data.
            - The annual means are calculated as semi-hemispheric means, which are then averaged globally.
            - The results are saved in the 'gml_annual_means' directory with the gas name as part of the filename.
            - The index of the resulting DataFrame is the year, formatted as YYYY.
            - The function prints a message indicating whether the annual means were successfully calculated and saved.
        """
        
        program = 'combined' if gas in self.combo_gases else 'msd'
        df = self.gml.loader(gas, program=program, gapfill=True)
        if df.empty:
            print(f"No data found for {gas}.")
            return None

        if program == 'combined':
            # The gap between otto and fe3 in the fECD data is not accounted for
            # in the Igor Pro combined data code. Interpolate here.
            df_numeric = df.apply(pd.to_numeric, errors='coerce')
            df = df_numeric.interpolate(method='time', limit_area='inside')
            df = self.refactor_combo_df(df)
        else:
            df.reset_index(inplace=True)

        # background sites used for a given gas
        bksites = self.bk_overrides.get(gas, self.default_bk_sites)
        means = self.semi_hemispheric_means(df, gas, sites_included=bksites)

        # exclude the current year (data is not fully QCed)
        if end_year is None:
            end_year = pd.Timestamp.now().year - 1
        else:
            end_year = int(end_year)
        means = means.loc[means.index.year <= end_year]
        
        def strick_annual_mean(df, month):
            """ Compute annual mean centered on month.
                This function is used to compute the annual mean for January (JAN) and July (JUL).
                It returns data for the full year, but only if all 12 months are present.
            """ 
            res = means.resample(f'YS-{month}', label='left', closed='left')
            # compute the annual mean and the count of valid months
            annual_mean  = res.mean()
            month_counts = res.count()
            complete_mask = month_counts.eq(12).all(axis=1)
            annual_strict = annual_mean.where(complete_mask)
            annual_strict.index = annual_strict.index + pd.DateOffset(months=6)  # shift index to center on the month

            col_order = ['HN', 'LN', 'LS', 'HS', 'Global']
            return annual_strict[col_order]

        jan_mean = strick_annual_mean(means, 'JUL')
        jul_mean = strick_annual_mean(means, 'JAN')
        
        if save_file:
            save_dir = Path("gml_annual_means/data_files")
            save_dir.mkdir(exist_ok=True)
            self.annual_means_figure(gas, df, jan_mean)

            for period, df_yearly in [("jan", jan_mean), ("jul", jul_mean)]:
                out_file = save_dir / f"{gas}_{period}_yearly.csv"
                hdr = self.get_file_header(gas)
                with open(out_file, "w") as fh:
                    fh.write(hdr)
                    df_yearly.to_csv(
                        fh,
                        float_format="%.3f",
                        date_format="%Y",
                        index_label="year",
                        na_rep='NaN'
                    )
            print(f"Annual means for {gas} calculated and saved.")

        # return raw and annual means centered on July
        return df, jul_mean

    def annual_means_figure(self, gas, raw, annual):
        fig, ax = plt.subplots()
        fig.suptitle(f'NOAA/GML {gas} annual means', fontsize=16)
        
        today = pd.Timestamp.today().normalize()

        bksites = self.bk_overrides.get(gas, self.default_bk_sites)
        for site in raw['site'].unique():
            if site not in bksites:
                continue
            sd = raw[raw['site'] == site]
            sd = sd[sd['date'] <= today]    # trim off forcast data
            ax.plot(sd['date'], sd['mf'], color='0.7', alpha=0.7, label='_nolegend_')

        bg_proxy = Line2D([0], [0], color='0.7', alpha=0.7, linewidth=1)

        x = annual.index + pd.DateOffset(months=6)
        mean_handles = []
        for col in annual.columns:
            if col == 'Global':
                h, = ax.plot(x, annual[col], color='k', marker='o', linestyle='-', linewidth=2, markersize=4, label=col)
            else:
                h, = ax.plot(x, annual[col], marker='', linestyle='-', linewidth=1, label=col)
            mean_handles.append(h)

        ax.legend([bg_proxy] + mean_handles, ['background sites'] + list(annual.columns), ncol=1, loc='best')
        ax.set_xlabel('Year')
        ax.set_ylabel(f'{gas} mole fraction (ppt)')

        save_dir = Path("gml_annual_means/figures")
        save_dir.mkdir(exist_ok=True)

        out_png = save_dir / f"{gas}_annual_means.png"
        print(f"Saving annual means figure for {gas} to {out_png}")
        fig.savefig(out_png, dpi=300, bbox_inches='tight')
        plt.close(fig)

    def get_file_header(self, gas):
        now = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
        return self.header_tpl.format(gas=gas, generated_on=now)

    def main(self):
        dfs = []
        for gas in self.gases:
            raw, jul = self.annual_means(gas, save_file=True)
            if jul is not None and "Global" in jul.columns:
                s = jul["Global"].copy()
                s.name = gas
                dfs.append(s)

        if not dfs:
            print("No data found for any gases.")
            return

        all_jul = pd.concat(dfs, axis=1)

        out_file = "gml_annual_means/GML_annual_means.csv"
        now = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
        hdr = self.gml_header_tpl.format(generated_on=now)
        with open(out_file, "w") as fh:
            fh.write(hdr)
            all_jul.to_csv(
                fh,
                float_format="%.3f",
                na_rep="NaN",
                index_label="year"
            )

        print(f"All annual means saved to '{out_file}'")


if __name__ == '__main__':
    gml_annual_means = GMLAnnualMeans()
    gml_annual_means.main()