#! /usr/bin/env python

import pandas as pd
import numpy as np
import yaml            # pip install pyyaml
import sys
from pathlib import Path
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D


sys.path.append('NOAA_halocarbons_loader')
import NOAA_halocarbons_loader.halocarbons_loader as halocarbons_loader

# ——— load config at module import ———
_CFG_PATH = Path(__file__).parent / "gml_config.yaml"
with open(_CFG_PATH) as f:
    _CFG = yaml.safe_load(f)

# extract bits
_HEADER_TPL      = _CFG["annual_means_file_header"]
_GML_HEADER_TPL  = _CFG["GML_means_file_header"]
GASES            = _CFG["gases"]
SOURCES          = _CFG["data_source"]
COMBO_GASES      = SOURCES.get("combined", [])
MSD_GASES        = SOURCES.get("msd", [])
DEFAULT_BK_SITES = _CFG["background_sites"]
BK_OVERRIDES     = _CFG.get("gas_background_overrides", {})


gml = halocarbons_loader.HATS_Loader()

def refactor_combo_df(df):
    """
    Load and refactor a combined GML data file into long format.
    
    Parameters
    ----------
    file_path : str or Path
        Path to the combined GML data CSV file.
        
    Returns
    -------
    pandas.DataFrame
        Refactored DataFrame in long format.
    """
    
    df2 = df.copy()
    df2.index.name = 'date'
    df2 = df2.reset_index()

    # Drop the unwanted columns:
    to_drop = ['NH','NH_sd','SH','SH_sd','Global','Global_sd','Programs']
    df2 = df2.drop(columns=to_drop)

    # Identify which columns are “mf” vs “sd”:
    mf_cols = [c for c in df2.columns if not c.endswith('_sd') and c != 'date']
    sd_cols = [f"{c}_sd" for c in mf_cols]

    # Melt the mf columns:
    mf_long = df2.melt(
        id_vars='date',
        value_vars=mf_cols,
        var_name='site',
        value_name='mf'
    )

    # Melt the sd columns, then strip the “_sd” suffix to get the same site names:
    sd_long = df2.melt(
        id_vars='date',
        value_vars=sd_cols,
        var_name='site',
        value_name='sd'
    )
    sd_long['site'] = sd_long['site'].str[:-3]

    # Merge them back together:
    long_df = pd.merge(mf_long, sd_long, on=['date','site'])

    long_df = long_df.sort_values(['date','site']).reset_index(drop=True)
    long_df = gml.add_location(long_df)
    long_df.reset_index(inplace=True)
    
    return long_df

def semi_hemispheric_means(df, sites_included=None, phi=30.0):
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

    # bump 'spo' to lat = –65°
    df.loc[df['site']=='spo', 'lat'] = -65.0

    # bucket into semi-hemispheres
    def _bucket(lat):
        if   lat >=  phi: return 'HN'
        elif lat >=   0 : return 'LN'
        elif lat <=  -phi: return 'HS'
        else:              return 'LS'
    df['region'] = df['lat'].apply(_bucket)

    # compute cosine-latitude weights
    df['w'] = np.cos(np.deg2rad(df['lat']))
    
    # validate data and drop rows with NaN in 'mf'
    # (this is necessary for the groupby weighted mean to work correctly)
    df_valid = df.dropna(subset=['mf'])

    # group by region & date, take weighted mean
    weighted = (
        df_valid
        .groupby(['region','date'])[['mf','w']]
        .apply(lambda g: np.average(g['mf'], weights=g['w']))
        .reset_index(name='mf_semi_mean')
    )

    # pivot to wide form
    semi_means = weighted.pivot(index='date', columns='region', values='mf_semi_mean')

    # compute global mean = average of the four semi-hemispheric columns
    # (all four semi-hemispheric means must be present)
    for col in ['HN','LN','LS','HS']:
        if col not in semi_means.columns:
            semi_means[col] = np.nan
    semi_means['Global'] = semi_means[['HN','LN','LS','HS']].mean(axis=1, skipna=False)

    return semi_means

def annual_means(gas, end_year=None, save_file=False):
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
    Examples
    --------
    >>> annual_means('CH3br')
    >>> annual_means('MC', end_year=2020)
    """
    
    program = 'combined' if gas in COMBO_GASES else 'msd'
    
    df = gml.loader(gas, program=program, gapfill=True)
    if df.empty:
        print(f"No data found for {gas}.")
        return None
    
    if program == 'combined':
        df = refactor_combo_df(df)
    else:
        df.reset_index(inplace=True)
    
    # background sites
    bksites = BK_OVERRIDES.get(gas, DEFAULT_BK_SITES)

    means = semi_hemispheric_means(df, sites_included=bksites)

    # drop everything after end_year
    if end_year is None:
        end_year = pd.Timestamp.now().year - 1
    else:
        end_year = int(end_year)
    means = means.loc[means.index.year <= end_year]

    # JAN‐to‐JAN
    jan_mean  = means.resample('YS-JAN', label='left', closed='left').mean()
    jan_count = means.resample('YS-JAN', label='left', closed='left').count()
    # keep only those years where *all* regions have 12 months
    valid_jan = jan_count.min(axis=1) >= 12
    jan_yearly = jan_mean.loc[valid_jan]

    # JUL‐to‐JUL
    jul_mean  = means.resample('YS-JUL', label='left', closed='left').mean()
    jul_count = means.resample('YS-JUL', label='left', closed='left').count()
    valid_jul = jul_count.min(axis=1) >= 12
    jul_yearly = jul_mean.loc[valid_jul]

    # reorder both DataFrames
    col_order = ['HN','LN','LS','HS','Global']
    jan_yearly = jan_yearly[col_order]
    jul_yearly = jul_yearly[col_order]

    # save out
    if save_file:
        save_dir = Path("gml_annual_means/data_files")
        save_dir.mkdir(exist_ok=True)
        annual_means_figure(gas, df, jan_yearly)
        
        for period, df_yearly in [("jan", jan_yearly), ("jul", jul_yearly)]:
            out_file = save_dir / f"{gas}_{period}_yearly.csv"
            hdr = get_file_header(gas)
            with open(out_file, "w") as fh:
                fh.write(hdr)
                df_yearly.to_csv(
                    fh,
                    float_format="%.3f",
                    date_format="%Y",
                    index_label="year"
                )
        print(f"Annual means for {gas} calculated and saved.")
        
    return df, jan_yearly

def annual_means_figure(gas, raw, annual):
    fig, ax = plt.subplots()
    fig.suptitle(f'NOAA/GML {gas} annual means', fontsize=16)

    # draw all background sites (no legend entries)
    bksites = BK_OVERRIDES.get(gas, DEFAULT_BK_SITES)
    for site in raw['site'].unique():
        if site not in bksites:
            continue
        sd = raw[raw['site'] == site]
        ax.plot(sd['date'], sd['mf'],
                color='0.7', alpha=0.7, label='_nolegend_')

    # proxy for background-sites legend entry
    bg_proxy = Line2D([0], [0], color='0.7', alpha=0.7, linewidth=1)

    # plot each annual column, shifting forward 6 months
    x = annual.index + pd.DateOffset(months=6)
    mean_handles = []
    for col in annual.columns:
        if col == 'Global':
            # force Global to be black
            h, = ax.plot(x, annual[col],
                        color='k', marker='o', linestyle='-', linewidth=2, markersize=4,
                        label=col)
        else:
            # let matplotlib cycle default colors for the others
            h, = ax.plot(x, annual[col],
                        marker='', linestyle='-', linewidth=1, label=col)
        mean_handles.append(h)

    # assemble legend: one entry for bg sites + one per annual series
    ax.legend(
        [bg_proxy] + mean_handles,
        ['background sites'] + list(annual.columns),
        ncol=1,
        loc='best'
    )

    ax.set_xlabel('Year')
    ax.set_ylabel(f'{gas} mole fraction (ppt)')
    
    save_dir = Path("gml_annual_means/figures")
    save_dir.mkdir(exist_ok=True)

    out_png = save_dir / f"{gas}_annual_means.png"
    print(f"Saving annual means figure for {gas} to {out_png}")
    fig.savefig(out_png, dpi=300, bbox_inches='tight')
    plt.close(fig)
    

def get_file_header(gas: str) -> str:
    now = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
    return _HEADER_TPL.format(gas=gas, generated_on=now)


def main():
    dfs = []
    for gas in GASES:
        raw, jan = annual_means(gas, save_file=True)
        if jan is not None and "Global" in jan.columns:
            # extract only the Global column, rename it to the gas
            s = jan["Global"].copy()
            s.name = gas
            dfs.append(s)

    if not dfs:
        print("No data found for any gases.")
        return

    # now concat a bunch of single‐column Series → one col per gas
    all_jan = pd.concat(dfs, axis=1)

    # shift the index to mid‐year floats, and name it "year"
    mid_years = all_jan.index.year + 0.5
    all_jan.index = mid_years
    all_jan.index.name = "year"
    all_jan.index = all_jan.index.map(lambda y: f"{y:.1f}")

    out_file = "gml_annual_means/GML_annual_means.csv"
    now = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
    hdr = _GML_HEADER_TPL.format(generated_on=now)
    with open(out_file, "w") as fh:
        fh.write(hdr)
        # write the entire DataFrame — header will be: year,<gas1>,<gas2>,...
        all_jan.to_csv(
            fh,
            float_format="%.3f",
            na_rep="NaN",
            index_label="year"
        )

    print(f"All annual means saved to '{out_file}'")
    
if __name__ == '__main__':
    main()