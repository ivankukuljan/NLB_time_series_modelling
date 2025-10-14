import pandas as pd
import numpy as np
from datetime import datetime
import re
import os
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, date
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from itertools import cycle
from collections import defaultdict
from typing import Dict, Tuple, Iterable, Optional
from sklearn.metrics import mean_squared_error, r2_score


eurostat_datasets = {'Unemployment Rate' :
 {'dataset_code' : 'une_rt_q',
 'frequency' : 'q',
 'url' : 'https://ec.europa.eu/eurostat/api/dissemination/sdmx/3.0/data/dataflow/ESTAT/une_rt_q/1.0/*.*.*.*.*.*?c[freq]=Q&c[s_adj]=SA&c[age]=Y20-64&c[unit]=PC_ACT&c[sex]=T&c[geo]=SI&c[TIME_PERIOD]=ge:2006-Q1+le:2023-Q4&compress=false&format=csvdata&formatVersion=2.0&lang=en&labels=name',
 'unit_column_name' : 'Unit of measure',
 'fill_missing' : False},
                     
 'Gross domestic product (GDP) (2015 = 100)' :
 {'dataset_code' : 'nama_10_gdp',
 'frequency' : 'a',
 'url' : 'https://ec.europa.eu/eurostat/api/dissemination/sdmx/3.0/data/dataflow/ESTAT/nama_10_gdp/1.0/*.*.*.*?c[freq]=A&c[unit]=CLV_I15&c[na_item]=B1GQ&c[geo]=SI&c[TIME_PERIOD]=2023,2022,2021,2020,2019,2018,2017,2016,2015,2014,2013,2012,2011,2010,2009,2008,2007,2006&compress=false&format=csvdata&formatVersion=2.0&lang=en&labels=name',
 'unit_column_name' : 'Unit of measure',
 'fill_missing' : True},

 'Gross domestic product (GDP) (percentage change on previous)' :    
 {'dataset_code' : 'nama_10_gdp',
 'frequency' : 'a',
 'url' : 'https://ec.europa.eu/eurostat/api/dissemination/sdmx/3.0/data/dataflow/ESTAT/nama_10_gdp/1.0/*.*.*.*?c[freq]=A&c[unit]=CLV_PCH_PRE&c[na_item]=B1GQ&c[geo]=SI&c[TIME_PERIOD]=2023,2022,2021,2020,2019,2018,2017,2016,2015,2014,2013,2012,2011,2010,2009,2008,2007,2006&compress=false&format=csvdata&formatVersion=2.0&lang=en&labels=name',
 'unit_column_name' : 'Unit of measure',
 'fill_missing' : True},

 'Money Market Interest Rate' : 
 {'dataset_code' : 'irt_st_m',
 'frequency' : 'm',
 'url' : 'https://ec.europa.eu/eurostat/api/dissemination/sdmx/3.0/data/dataflow/ESTAT/irt_st_m/1.0/*.*.*?c[freq]=M&c[int_rt]=IRT_M12&c[geo]=EA&c[TIME_PERIOD]=2023-12,2023-11,2023-10,2023-09,2023-08,2023-07,2023-06,2023-05,2023-04,2023-03,2023-02,2023-01,2022-12,2022-11,2022-10,2022-09,2022-08,2022-07,2022-06,2022-05,2022-04,2022-03,2022-02,2022-01,2021-12,2021-11,2021-10,2021-09,2021-08,2021-07,2021-06,2021-05,2021-04,2021-03,2021-02,2021-01,2020-12,2020-11,2020-10,2020-09,2020-08,2020-07,2020-06,2020-05,2020-04,2020-03,2020-02,2020-01,2019-12,2019-11,2019-10,2019-09,2019-08,2019-07,2019-06,2019-05,2019-04,2019-03,2019-02,2019-01,2018-12,2018-11,2018-10,2018-09,2018-08,2018-07,2018-06,2018-05,2018-04,2018-03,2018-02,2018-01,2017-12,2017-11,2017-10,2017-09,2017-08,2017-07,2017-06,2017-05,2017-04,2017-03,2017-02,2017-01,2016-12,2016-11,2016-10,2016-09,2016-08,2016-07,2016-06,2016-05,2016-04,2016-03,2016-02,2016-01,2015-12,2015-11,2015-10,2015-09,2015-08,2015-07,2015-06,2015-05,2015-04,2015-03,2015-02,2015-01,2014-12,2014-11,2014-10,2014-09,2014-08,2014-07,2014-06,2014-05,2014-04,2014-03,2014-02,2014-01,2013-12,2013-11,2013-10,2013-09,2013-08,2013-07,2013-06,2013-05,2013-04,2013-03,2013-02,2013-01,2012-12,2012-11,2012-10,2012-09,2012-08,2012-07,2012-06,2012-05,2012-04,2012-03,2012-02,2012-01,2011-12,2011-11,2011-10,2011-09,2011-08,2011-07,2011-06,2011-05,2011-04,2011-03,2011-02,2011-01,2010-12,2010-11,2010-10,2010-09,2010-08,2010-07,2010-06,2010-05,2010-04,2010-03,2010-02,2010-01,2009-12,2009-11,2009-10,2009-09,2009-08,2009-07,2009-06,2009-05,2009-04,2009-03,2009-02,2009-01,2008-12,2008-11,2008-10,2008-09,2008-08,2008-07,2008-06,2008-05,2008-04,2008-03,2008-02,2008-01,2007-12,2007-11,2007-10,2007-09,2007-08,2007-07,2007-06,2007-05,2007-04,2007-03,2007-02,2007-01,2006-12,2006-11,2006-10,2006-09,2006-08,2006-07,2006-06,2006-05,2006-04,2006-03,2006-02,2006-01&compress=false&format=csvdata&formatVersion=2.0&lang=en&labels=name',
 'unit_column_name' : 'Interest rate',
 'resample_method' : 'last',
 'fill_missing' : True},

 'House price index (2015 = 100)' : 
 {'dataset_code' : 'prc_hpi_q',
 'frequency' : 'q',
 'url' : 'https://ec.europa.eu/eurostat/api/dissemination/sdmx/3.0/data/dataflow/ESTAT/prc_hpi_q/1.0/*.*.*.*?c[freq]=Q&c[purchase]=TOTAL&c[unit]=I15_Q&c[geo]=SI&c[TIME_PERIOD]=2023-Q4,2023-Q3,2023-Q2,2023-Q1,2022-Q4,2022-Q3,2022-Q2,2022-Q1,2021-Q4,2021-Q3,2021-Q2,2021-Q1,2020-Q4,2020-Q3,2020-Q2,2020-Q1,2019-Q4,2019-Q3,2019-Q2,2019-Q1,2018-Q4,2018-Q3,2018-Q2,2018-Q1,2017-Q4,2017-Q3,2017-Q2,2017-Q1,2016-Q4,2016-Q3,2016-Q2,2016-Q1,2015-Q4,2015-Q3,2015-Q2,2015-Q1,2014-Q4,2014-Q3,2014-Q2,2014-Q1,2013-Q4,2013-Q3,2013-Q2,2013-Q1,2012-Q4,2012-Q3,2012-Q2,2012-Q1,2011-Q4,2011-Q3,2011-Q2,2011-Q1,2010-Q4,2010-Q3,2010-Q2,2010-Q1,2009-Q4,2009-Q3,2009-Q2,2009-Q1,2008-Q4,2008-Q3,2008-Q2,2008-Q1,2007-Q4,2007-Q3,2007-Q2,2007-Q1,2006-Q4,2006-Q3,2006-Q2,2006-Q1&compress=false&format=csvdata&formatVersion=2.0&lang=en&labels=name',
 'unit_column_name' : 'Unit of measure',
 'fill_missing' : True},

 'Harmonised Index of Consumer Prices (annual rate of change)':
 {'dataset_code' : 'prc_hicp_manr',
 'frequency' : 'm',
 'url' : 'https://ec.europa.eu/eurostat/api/dissemination/sdmx/3.0/data/dataflow/ESTAT/prc_hicp_manr/1.0/*.*.*.*?c[freq]=M&c[unit]=RCH_A&c[coicop]=CP00&c[geo]=SI&c[TIME_PERIOD]=2023-12,2023-11,2023-10,2023-09,2023-08,2023-07,2023-06,2023-05,2023-04,2023-03,2023-02,2023-01,2022-12,2022-11,2022-10,2022-09,2022-08,2022-07,2022-06,2022-05,2022-04,2022-03,2022-02,2022-01,2021-12,2021-11,2021-10,2021-09,2021-08,2021-07,2021-06,2021-05,2021-04,2021-03,2021-02,2021-01,2020-12,2020-11,2020-10,2020-09,2020-08,2020-07,2020-06,2020-05,2020-04,2020-03,2020-02,2020-01,2019-12,2019-11,2019-10,2019-09,2019-08,2019-07,2019-06,2019-05,2019-04,2019-03,2019-02,2019-01,2018-12,2018-11,2018-10,2018-09,2018-08,2018-07,2018-06,2018-05,2018-04,2018-03,2018-02,2018-01,2017-12,2017-11,2017-10,2017-09,2017-08,2017-07,2017-06,2017-05,2017-04,2017-03,2017-02,2017-01,2016-12,2016-11,2016-10,2016-09,2016-08,2016-07,2016-06,2016-05,2016-04,2016-03,2016-02,2016-01,2015-12,2015-11,2015-10,2015-09,2015-08,2015-07,2015-06,2015-05,2015-04,2015-03,2015-02,2015-01,2014-12,2014-11,2014-10,2014-09,2014-08,2014-07,2014-06,2014-05,2014-04,2014-03,2014-02,2014-01,2013-12,2013-11,2013-10,2013-09,2013-08,2013-07,2013-06,2013-05,2013-04,2013-03,2013-02,2013-01,2012-12,2012-11,2012-10,2012-09,2012-08,2012-07,2012-06,2012-05,2012-04,2012-03,2012-02,2012-01,2011-12,2011-11,2011-10,2011-09,2011-08,2011-07,2011-06,2011-05,2011-04,2011-03,2011-02,2011-01,2010-12,2010-11,2010-10,2010-09,2010-08,2010-07,2010-06,2010-05,2010-04,2010-03,2010-02,2010-01,2009-12,2009-11,2009-10,2009-09,2009-08,2009-07,2009-06,2009-05,2009-04,2009-03,2009-02,2009-01,2008-12,2008-11,2008-10,2008-09,2008-08,2008-07,2008-06,2008-05,2008-04,2008-03,2008-02,2008-01,2007-12,2007-11,2007-10,2007-09,2007-08,2007-07,2007-06,2007-05,2007-04,2007-03,2007-02,2007-01,2006-12,2006-11,2006-10,2006-09,2006-08,2006-07,2006-06,2006-05,2006-04,2006-03,2006-02,2006-01&compress=false&format=csvdata&formatVersion=2.0&lang=en&labels=name',
 'unit_column_name' : 'Unit of measure',
 'resample_method' : 'mean',
 'fill_missing' : True},

 'Consumer confidence indicator':
 {'dataset_code' : 'ei_bsco_m',
 'frequency' : 'm',
 'url' : 'https://ec.europa.eu/eurostat/api/dissemination/sdmx/3.0/data/dataflow/ESTAT/ei_bsco_m/1.0/*.*.*.*.*?c[freq]=M&c[indic]=BS-CSMCI&c[s_adj]=SA&c[unit]=BAL&c[geo]=SI&c[TIME_PERIOD]=2023-12,2023-11,2023-10,2023-09,2023-08,2023-07,2023-06,2023-05,2023-04,2023-03,2023-02,2023-01,2022-12,2022-11,2022-10,2022-09,2022-08,2022-07,2022-06,2022-05,2022-04,2022-03,2022-02,2022-01,2021-12,2021-11,2021-10,2021-09,2021-08,2021-07,2021-06,2021-05,2021-04,2021-03,2021-02,2021-01,2020-12,2020-11,2020-10,2020-09,2020-08,2020-07,2020-06,2020-05,2020-04,2020-03,2020-02,2020-01,2019-12,2019-11,2019-10,2019-09,2019-08,2019-07,2019-06,2019-05,2019-04,2019-03,2019-02,2019-01,2018-12,2018-11,2018-10,2018-09,2018-08,2018-07,2018-06,2018-05,2018-04,2018-03,2018-02,2018-01,2017-12,2017-11,2017-10,2017-09,2017-08,2017-07,2017-06,2017-05,2017-04,2017-03,2017-02,2017-01,2016-12,2016-11,2016-10,2016-09,2016-08,2016-07,2016-06,2016-05,2016-04,2016-03,2016-02,2016-01,2015-12,2015-11,2015-10,2015-09,2015-08,2015-07,2015-06,2015-05,2015-04,2015-03,2015-02,2015-01,2014-12,2014-11,2014-10,2014-09,2014-08,2014-07,2014-06,2014-05,2014-04,2014-03,2014-02,2014-01,2013-12,2013-11,2013-10,2013-09,2013-08,2013-07,2013-06,2013-05,2013-04,2013-03,2013-02,2013-01,2012-12,2012-11,2012-10,2012-09,2012-08,2012-07,2012-06,2012-05,2012-04,2012-03,2012-02,2012-01,2011-12,2011-11,2011-10,2011-09,2011-08,2011-07,2011-06,2011-05,2011-04,2011-03,2011-02,2011-01,2010-12,2010-11,2010-10,2010-09,2010-08,2010-07,2010-06,2010-05,2010-04,2010-03,2010-02,2010-01,2009-12,2009-11,2009-10,2009-09,2009-08,2009-07,2009-06,2009-05,2009-04,2009-03,2009-02,2009-01,2008-12,2008-11,2008-10,2008-09,2008-08,2008-07,2008-06,2008-05,2008-04,2008-03,2008-02,2008-01,2007-12,2007-11,2007-10,2007-09,2007-08,2007-07,2007-06,2007-05,2007-04,2007-03,2007-02,2007-01,2006-12,2006-11,2006-10,2006-09,2006-08,2006-07,2006-06,2006-05,2006-04,2006-03,2006-02,2006-01&compress=false&format=csvdata&formatVersion=2.0&lang=en&labels=name',
 'unit_column_name' : 'Unit of measure',
 'resample_method' : 'mean',
 'fill_missing' : True},

 'Private sectror credit flow':
 {'dataset_code' : 'tipspc10',
 'frequency' : 'a',
 'url' : 'https://ec.europa.eu/eurostat/api/dissemination/sdmx/3.0/data/dataflow/ESTAT/tipspc10/1.0/*.*.*.*.*.*.*?c[freq]=A&c[co_nco]=CO&c[sector]=S11_S14_S15&c[finpos]=LIAB&c[na_item]=F3_F4&c[unit]=PC_GDP&c[geo]=SI&c[TIME_PERIOD]=2023,2022,2021,2020,2019,2018,2017,2016,2015,2014,2013,2012,2011,2010,2009,2008,2007,2006&compress=false&format=csvdata&formatVersion=2.0&lang=en&labels=name',
 'unit_column_name' : 'Unit of measure',
 'fill_missing' : True}
                    }



def fill_missing_values(df: pd.DataFrame, fill_missing: bool = True) -> pd.DataFrame:
    """
    Fill missing values in a DataFrame using the nearest available value.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame (any columns).
    fill_missing : bool, default True
        If True, fill missing values using nearest available data (forward then backward fill).

    Returns
    -------
    pd.DataFrame
        DataFrame with missing values filled.
    """
    df = df.copy()  # Work on a copy to avoid modifying input data in-place

    if not fill_missing:
        # If filling missing values is not requested, return the DataFrame unchanged
        return df

    # Forward fill: Fill missing values with the previous non-missing value in each column.
    # Back fill: For any leading or still-missing values, fill with next available value.
    df = df.ffill().bfill()

    return df  # Return DataFrame with missing values filled



def resample_monthly_to_quarterly(
    df: pd.DataFrame,
    resampling_perscription: str = 'mean',
    fill_missing: str = 'linear_interpolate'
) -> pd.DataFrame:
    """
    Resample monthly data to quarterly. Preserves non-numeric columns with 'first'.
    Returns a DataFrame with a 'TIME_PERIOD' column formatted 'YYYY-Q#'.
    """
    df = df.copy()  # Work on a copy to avoid modifying the original DataFrame

    # Ensure the DataFrame is indexed by datetime, using 'TIME_PERIOD' if present,
    # and align to the corresponding month-end date
    if 'TIME_PERIOD' in df.columns:
        # Convert TIME_PERIOD column to datetime, assuming '%Y-%m' string format
        df['TIME_PERIOD'] = pd.to_datetime(df['TIME_PERIOD'], format='%Y-%m', errors='coerce')
        df = df.set_index('TIME_PERIOD')
    else:
        # Otherwise, try to convert the current index to datetime
        df.index = pd.to_datetime(df.index, errors='coerce')

    # Normalize index to month-end and sort index
    df.index = df.index.to_period('M').to_timestamp('M')
    df = df.sort_index()

    # Build a complete monthly DatetimeIndex from the min to max index, and reindex the DataFrame to fill gaps
    if len(df.index) > 0 and isinstance(df.index, pd.DatetimeIndex):
        full_months = pd.date_range(df.index.min(), df.index.max(), freq='M')
        df = df.reindex(full_months)

    # Fill missing values according to the fill_missing_values function
    df = fill_missing_values(df, fill_missing)

    # Identify numeric columns and non-numeric columns for disjoint aggregation policies
    num_cols = df.select_dtypes(include='number').columns.tolist()
    # Non-numeric columns = all columns that are not in num_cols
    non_num_cols = [c for c in df.columns if c not in num_cols]

    # Decide how to aggregate numeric columns: average vs last value
    if resampling_perscription == 'mean':
        num_agg = 'mean'
    elif resampling_perscription == 'last':
        num_agg = 'last'
    else:
        raise ValueError("resampling_perscription must be 'mean' or 'last'")

    # Build aggregation dictionary: numeric columns get num_agg, non-numeric get 'first'
    agg_dict = {}
    if num_cols:
        agg_dict.update({c: num_agg for c in num_cols})
    if non_num_cols:
        # For categorical/descriptive columns (unit, country, etc.), keep the first value in each quarter
        agg_dict.update({c: 'first' for c in non_num_cols})

    # Do the resampling to quarterly frequency (Q-DEC means year ends in December; i.e., Q1=Jan-Mar,...)
    df_q = df.resample('Q-DEC').agg(agg_dict)

    # Create new TIME_PERIOD column in 'YYYY-Q#' format (with dash for clarity, e.g. '2010-Q1')
    q_labels = df_q.index.to_period('Q-DEC').astype(str).str.replace('Q', '-Q')
    df_q = df_q.reset_index(drop=True)  # Remove the old DatetimeIndex (now implicit in TIME_PERIOD col)
    df_q.insert(0, 'TIME_PERIOD', q_labels)  # Insert TIME_PERIOD as first column

    return df_q  # Return the resampled DataFrame




def retrieve_eurostat_dataset(dataset_dict):
    """
    Loads, checks, cleans, rescales, and adds useful fields for a Eurostat dataset.

    Parameters
    ----------
    dataset_dict : dict
        Must contain at least:
            - 'url': csv file location
            - 'fill_missing': value or None to use for missing fills
            - 'resample_method': 'mean' or 'last' if resampling is required

    Returns
    -------
    dataset_dict : dict
        Updated in place to contain:
            - 'times': list of period labels (str or yy-mm/yy-q)
            - 'values': list of values, as originally present
            - 'scaled_values': normalized values (zero mean/unit variance)
            - 'scaler_mean', 'scaler_scale': mean/scale used to standardize
        Or None upon error.
    """
    url = dataset_dict['url']  # Path to the raw Eurostat CSV data
    time_column_name = 'TIME_PERIOD'        # Name of the time column
    value_column_name = 'OBS_VALUE'         # Name of the value column
    country_column_name = 'Geopolitical entity (reporting)'  # Name of the country col

    # Load the CSV file into a DataFrame
    df = pd.read_csv(url)

    # Ensure the dataset only covers a single country (before resampling)
    if country_column_name in df.columns:
        countries = pd.unique(df[country_column_name].dropna())
        if len(countries) > 1:
            print('Error: more than one country in the dataset')
            return None
        if len(countries) == 0:
            print('Error: no country in the dataset')
            return None

    # Fill missing values if a fill_missing policy is specified
    if dataset_dict['fill_missing'] is not None:
        df = fill_missing_values(df, dataset_dict['fill_missing'])

    # Resample to quarterly, monthly, or other target frequency if needed
    if dataset_dict.get('resample_method'):
        df = resample_monthly_to_quarterly(
            df,
            resampling_perscription=dataset_dict['resample_method'],
            fill_missing=dataset_dict['fill_missing']
        )

    # Extract "times" and "values" to lists from the (possibly resampled) DataFrame
    times = df[time_column_name].tolist()
    values = df[value_column_name].tolist()

    # Convert original values to floating point numpy array for scaling
    vals = pd.to_numeric(df[value_column_name], errors='coerce').to_numpy()

    # Standardize the series (zero mean, unit variance)
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    scaled = scaler.fit_transform(vals.reshape(-1, 1)).ravel()

    # Attach processed arrays and stats to the dictionary (side-effect!)
    dataset_dict['times'] = times
    dataset_dict['values'] = values
    dataset_dict['scaled_values'] = scaled.tolist()
    dataset_dict['scaler_mean'] = float(scaler.mean_[0])
    dataset_dict['scaler_scale'] = float(scaler.scale_[0])

    return dataset_dict



def compute_time_distance(nlb_time, eurostat_time):
    """
    Compute the signed distance between an NLB date and a Eurostat period, in Eurostat units.

    NLB time accepted formats/types:
      - 'MM/DD/YYYY'  (e.g., '03/31/2008')
      - 'YYYY-MM-DD'  (e.g., '2008-03-31')
      - 'YYYY-MM-DD HH:MM:SS' (e.g., '2008-03-31 00:00:00')
      - datetime.date / datetime.datetime
      - pandas.Timestamp (duck-typed via .to_pydatetime(), if present)

    Eurostat formats (auto-detected):
      - Annual:    'YYYY' or 4-digit int (e.g., '2009' or 2009)
      - Quarterly: 'YYYY-Q#' or 'YYYYQ#'  (e.g., '2009-Q1', '2009Q3')
      - Monthly:   'YYYY-MM' or 'YYYY-M' (e.g., '2009-01', '2009-9')

    Returns:
      Signed integer distance in Eurostat units:
        * years if annual
        * quarters if quarterly
        * months if monthly
      Negative => Eurostat period is before NLB date
      Zero     => same period
      Positive => Eurostat period is after NLB date
    """

    # --- Parse NLB time robustly ---
    nlb_dt = parse_nlb_time(nlb_time)  # Convert NLB input into a datetime object
    nlb_year = nlb_dt.year              # Extract the year from NLB date
    nlb_month = nlb_dt.month            # Extract the month from NLB date
    nlb_quarter = (nlb_month - 1) // 3 + 1   # Compute the quarter (Q1=Jan–Mar, etc.)

    # --- Detect & parse Eurostat time ---
    if isinstance(eurostat_time, int):
        # If Eurostat time is an integer, treat as yearly frequency
        euro_unit = "year"
        euro_year = eurostat_time
    else:
        # Otherwise, parse the Eurostat time string
        et = str(eurostat_time).strip()

        # Try to match yearly, quarterly, and monthly formats with regex
        m_year = re.fullmatch(r"(\d{4})", et)
        m_quarter = re.fullmatch(r"(\d{4})-?Q([1-4])", et, flags=re.IGNORECASE)
        m_month = re.fullmatch(r"(\d{4})-(\d{1,2})", et)

        if m_quarter:
            # Format: quarterly (extract year and quarter)
            euro_unit = "quarter"
            euro_year = int(m_quarter.group(1))
            euro_q = int(m_quarter.group(2))
        elif m_month:
            # Format: monthly (extract year and month)
            euro_unit = "month"
            euro_year = int(m_month.group(1))
            euro_m = int(m_month.group(2))
            if not (1 <= euro_m <= 12):
                # Guard against invalid months
                raise ValueError(f"Invalid Eurostat month in '{eurostat_time}'.")
        elif m_year:
            # Format: annual (extract year)
            euro_unit = "year"
            euro_year = int(m_year.group(1))
        else:
            # Unrecognized format; raise helpful message
            raise ValueError(
                f"Unrecognized Eurostat time '{eurostat_time}'. "
                "Use 'YYYY', 'YYYY-Q#' (or 'YYYYQ#'), or 'YYYY-MM'."
            )

    # --- Compute distance in Eurostat units ---

    # Annual: return the difference in years
    if euro_unit == "year":
        return euro_year - nlb_year

    # Quarterly: map to integer index (year*4 + quarter) and subtract
    if euro_unit == "quarter":
        nlb_q_index = nlb_year * 4 + (nlb_quarter - 1)
        euro_q_index = euro_year * 4 + (euro_q - 1)
        return euro_q_index - nlb_q_index

    # Monthly: map to integer index (year*12 + month) and subtract
    if euro_unit == "month":
        nlb_m_index = nlb_year * 12 + (nlb_month - 1)
        euro_m_index = euro_year * 12 + (euro_m - 1)
        return euro_m_index - nlb_m_index

    # Unknown case: indicate programming error
    raise RuntimeError("Unexpected Eurostat unit detection state.")



def parse_nlb_time(nlb_time):
    """
    Convert NLB time input into a datetime.datetime object.

    Supports:
        - datetime.datetime (returns as is)
        - datetime.date (converted to datetime with 00:00:00 time)
        - Pandas Timestamps (duck-typed via .to_pydatetime)
        - Strings in common formats: 'MM/DD/YYYY', 'YYYY-MM-DD', 'YYYY-MM-DD HH:MM:SS'

    Args:
        nlb_time: Input time. May be a datetime, date, pandas.Timestamp, or a string.

    Returns:
        datetime.datetime: corresponding datetime representation.

    Raises:
        ValueError: If format is not recognized.
    """

    # If already a datetime.datetime, return it directly
    if isinstance(nlb_time, datetime):
        return nlb_time

    # If it's a datetime.date (but not datetime.datetime), convert to datetime (midnight)
    if isinstance(nlb_time, date):
        return datetime(nlb_time.year, nlb_time.month, nlb_time.day)

    # If it looks like a pandas.Timestamp (duck-typed), convert to Python datetime
    if hasattr(nlb_time, "to_pydatetime"):
        return nlb_time.to_pydatetime()

    # Assume string input - try parsing with common formats
    s = str(nlb_time).strip()
    for fmt in ("%m/%d/%Y", "%Y-%m-%d", "%Y-%m-%d %H:%M:%S"):
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            pass  # Try next format if parsing fails

    # If none of the above worked, raise an error with guidance
    raise ValueError(
        f"Invalid NLB time '{nlb_time}'. "
        "Expected one of: 'MM/DD/YYYY', 'YYYY-MM-DD', 'YYYY-MM-DD HH:MM:SS', "
        "a datetime/date, or a pandas Timestamp."
    )



def get_eurostat_time_unit(eurostat_time):
    """
    Detect the time unit of a Eurostat time string.

    Eurostat formats:
      - Annual:   '2008' or 2008
      - Quarterly:'2008-Q1', '2008Q2'
      - Monthly:  '2008-01', '2008-1'

    Returns:
      'y' - annual, 'q' - quarterly, 'm' - monthly
    Raises:
      ValueError if the format is not recognized.
    """
    # If input is an integer, treat as annual ('y')
    if isinstance(eurostat_time, int):
        return "y"

    # Convert input to string and remove extra whitespace
    et = str(eurostat_time).strip()

    # Check if input matches a four-digit year: annual
    if re.fullmatch(r"\d{4}", et):
        return "y"
    # Check if input matches quarterly patterns, e.g., 2008-Q1 or 2008Q2
    elif re.fullmatch(r"\d{4}-?Q[1-4]", et, flags=re.IGNORECASE):
        return "q"
    # Check if input matches monthly patterns, e.g., 2008-01 or 2008-1
    elif re.fullmatch(r"\d{4}-\d{1,2}", et):
        return "m"
    else:
        # If none of the above patterns matched, raise an error
        raise ValueError(f"Unrecognized Eurostat time format: '{eurostat_time}'")


def compose_modeling_data(
        eurostat_datasets,
        scale=True,
        impute=True,
        nlb_window=2,
        months_window=9,
        quarters_window=3,
        annual_window=1,
        nlb_npl_data_path='/home/ivan/IskanjeDela/Banking/NLB/datav3/npl_bank.xlsx',
        data_output_folder='/home/ivan/IskanjeDela/Banking/NLB/NLB_assignement_Kukuljan/TimeSeriesModelling/data'):
    """
    Compose a modeling DataFrame combining NPL data with Eurostat datasets, with options for scaling,
    imputation, lagged features, and data output caching.
    """

    # Ensure the output folder exists
    if not os.path.exists(data_output_folder):
        os.makedirs(data_output_folder)

    # Select output file name depending on scale/impute settings
    if scale:
        if impute:
            data_output_name = (
                f'data_nlb_window_{nlb_window}_months_window_{months_window}'
                f'_quarters_window_{quarters_window}_annual_window_{annual_window}_scaled_imputed.csv'
            )
        else:
            data_output_name = (
                f'data_nlb_window_{nlb_window}_months_window_{months_window}'
                f'_quarters_window_{quarters_window}_annual_window_{annual_window}_scaled.csv'
            )
    else:
        if impute:
            data_output_name = (
                f'data_nlb_window_{nlb_window}_months_window_{months_window}'
                f'_quarters_window_{quarters_window}_annual_window_{annual_window}_imputed.csv'
            )
        else:
            data_output_name = (
                f'data_nlb_window_{nlb_window}_months_window_{months_window}'
                f'_quarters_window_{quarters_window}_annual_window_{annual_window}.csv'
            )

    data_output_path = os.path.join(data_output_folder, data_output_name)

    # If the data is already saved, just load and return it
    if os.path.exists(data_output_path):
        return pd.read_csv(data_output_path, index_col='date', parse_dates=['date'])
    else:
        # --- Read and initialize NLB NPL data ---
        nlb_npl_data = pd.read_excel(nlb_npl_data_path)
        nlb_dates = nlb_npl_data['DATE'].to_list()          # the index dates for our dataset
        nlb_npl_values = nlb_npl_data['NPL'].to_list()      # the corresponding NPL (target) values

        # Initialize DataFrame with index as nlb_dates
        data = pd.DataFrame(index=nlb_dates)
        data['npl'] = nlb_npl_values

        # Optionally construct lagged NPL features (by quarter, for nlb_window lag length)
        if nlb_window > 0:
            for i in range(1, nlb_window + 1):
                data[f'npl-{i}q'] = data['npl'].shift(i)

        # Forward/backward-fill NPL lags if imputation is requested
        if impute:
            lag_cols = [f'npl-{i}q' for i in range(1, nlb_window + 1)]
            data[lag_cols] = data[lag_cols].ffill().bfill()

        # --- Add Eurostat datasets as covariates (with appropriate history windows) ---
        dataset_names = eurostat_datasets.keys()
        for dataset_name in dataset_names:
            dataset_data = eurostat_datasets[dataset_name]
            dataset_times = dataset_data['times']

            # Choose scaled or raw Eurostat values
            if scale:
                dataset_values = dataset_data['scaled_values']
            else:
                dataset_values = dataset_data['values']

            # Determine the time unit for the current dataset (yearly, quarterly, or monthly)
            times_unit = get_eurostat_time_unit(dataset_times[0])

            # Select the rolling window based on the time unit
            if times_unit == 'y':
                time_window = annual_window
            elif times_unit == 'q':
                time_window = quarters_window
            elif times_unit == 'm':
                time_window = months_window
            else:
                raise ValueError(f"Unrecognized Eurostat time unit: '{times_unit}'")

            # Create lag columns for this dataset, for window size (0 latest to N lags back)
            for i in range(time_window + 1):
                colname = f'{dataset_name}-{i}{times_unit}'
                data[colname] = np.nan  # initialize with NaN

                # For each NLB date, attempt to match with a corresponding Eurostat date at lag -i
                for nlb_date in nlb_dates:
                    for dataset_time in dataset_times:
                        # compute_time_distance returns negative for the past
                        if compute_time_distance(nlb_date, dataset_time) == -i:
                            # Uncomment next line for trace debugging
                            # print(f'time distance: {i}, nlb_date: {nlb_date}, dataset_time: {dataset_time}')
                            data.loc[nlb_date, colname] = dataset_values[dataset_times.index(dataset_time)]
                            break

            # --- Imputation logic for missing Eurostat data ---
            if impute:
                if dataset_name == 'Unemployment Rate':
                    # For 'Unemployment Rate', do special linear extrapolation (backwards)
                    for i in range(time_window + 1):
                        col = f'{dataset_name}-{i}{times_unit}'
                        # Avoid going out of bounds with j+1 and j+2 (iterate until 2 from bottom)
                        for j in range(len(data) - 2):
                            # If current is nan and next is not nan, try to extrapolate
                            if pd.isna(data.iloc[j][col]) and not pd.isna(data.iloc[j + 1][col]):
                                start_value = data.iloc[j + 1][col]
                                next_value = data.iloc[j + 2][col]

                                # If next value is nan too, can't do linear extrapolation
                                if pd.isna(next_value):
                                    continue

                                # The difference (slope) for backwards linear fill
                                delta = start_value - next_value

                                # Fill all consecutive nans back as a line
                                for k in range(j, -1, -1):
                                    if pd.isna(data.iloc[k][col]):
                                        # The further back, the further from start_value by slope times (j+1-k)
                                        data.iloc[k, data.columns.get_loc(col)] = start_value + delta * (j + 1 - k)
                                    else:
                                        # Stop when hit a non-nan value further back
                                        break
                else:
                    # For all other datasets, fill forward then backward using nearest values
                    for i in range(time_window + 1):
                        col = f'{dataset_name}-{i}{times_unit}'
                        if col in data.columns:
                            data[col] = data[col].ffill().bfill()

        # Uncomment next line to save data to disk if needed
        # data.to_csv(data_output_path, index_label='date')
        return data




def extract_dataset_name_delay_and_unit(colname):
    """
    Parse a column name of the form:
        "<dataset name>-<delay><unit>"
    where:
        - <dataset name> can contain hyphens, spaces, parentheses, etc.
        - <delay> is an integer (e.g., 0,1,2,...)
        - <unit> is a single letter (e.g., q=quarter, y=year, m=month)
    
    Examples:
      "npl-1q"                                        -> ("npl", 1, "q")
      "Unemployment Rate-0q"                          -> ("Unemployment Rate", 0, "q")
      "Gross domestic product (2015 = 100)-1y"        -> ("Gross domestic product (2015 = 100)", 1, "y")
      "Consumer confidence - general economic next 12 months-7m"
                                                      -> ("Consumer confidence - general economic next 12 months", 7, "m")
    
    Returns:
        (dataset_name: str, delay: int, unit: str)
    Raises:
        ValueError if the name does not match the expected pattern.
    """
    s = str(colname).strip()
    if "-" not in s:
        raise ValueError(f"Column name missing '-<delay><unit>' suffix: {colname!r}")

    dataset_name, suffix = s.rsplit("-", 1)  # split on the LAST hyphen
    suffix = suffix.strip()

    m = re.fullmatch(r"(\d+)\s*([A-Za-z])", suffix)
    if not m:
        raise ValueError(
            f"Suffix must be '<delay><unit>' like '3q' or '12m'; got {suffix!r} in {colname!r}"
        )

    delay = int(m.group(1))
    unit = m.group(2).lower()  # normalize (e.g., 'Q' -> 'q')

    return dataset_name.strip(), delay, unit



def train_test_split(data, test_1_size=6, test_2_size=6, random_state=None):
    """
    Splits the data into three sets: training, test_1, and test_2.

    - test_2: The last `test_2_size` rows of the data (chronologically, for time series).
    - test_1: Random `test_1_size` rows from the remaining data (not including test_2).
    - train: The remaining data after removing both test splits.

    Parameters
    ----------
    data : pandas.DataFrame
        The complete dataset to split. Should have a time-based index if time-split is desired.
    test_1_size : int, optional
        Number of rows to sample randomly for the first test set.
    test_2_size : int, optional
        Number of final rows for the second test set (default 6).
    random_state : int, optional
        Random seed for reproducibility of the random sample.

    Returns
    -------
    data_train : pandas.DataFrame
        The training data (with neither test_1 nor test_2 rows).
    data_test_1 : pandas.DataFrame
        Random sample from the remaining data (after taking off test_2).
    data_test_2 : pandas.DataFrame
        The last `test_2_size` rows of the input data.
    """

    # Make a copy of the input data to avoid mutating the original dataframe
    data_copy = data.copy()
    
    # The last `test_2_size` entries are reserved for the second test split (chronological holdout)
    data_test_2 = data_copy.tail(test_2_size)
    
    # Remove test_2 entries from the data copy to create a pool for training and test_1 split
    data_remaining = data_copy.iloc[:-test_2_size]
    
    # Randomly sample `test_1_size` rows from the remaining data to create the first test split
    data_test_1 = data_remaining.sample(n=test_1_size, random_state=random_state)
    
    # The rows not in the first or second test splits form the training data
    data_train = data_remaining.drop(data_test_1.index)
    
    # Return the three splits in order: training, test_1, test_2
    return data_train, data_test_1, data_test_2



def fit_linear_model_with_time_delay(data):
    """
    Fit a linear regression model to the data, using all columns except 'npl' as predictors
    and 'npl' as the target variable.
    
    Parameters
    ----------
    data : pandas.DataFrame
        DataFrame containing the feature columns and an 'npl' column as target.

    Returns
    -------
    model : LinearRegression
        Fitted linear regression model.
    """
    y = data['npl']                    # Target variable
    x = data.drop(columns=['npl'])     # Predictor variables
    model = LinearRegression()         # Instantiate linear regression
    model.fit(x, y)                    # Fit model to data
    return model                       # Return trained model

def evaluate_linear_model(model, test_data):
    """
    Evaluate a fitted linear regression model on test data.
    Predicts the 'npl' value, clips predictions at zero, and computes evaluation metrics.

    Parameters
    ----------
    model : LinearRegression
        Fitted sklearn linear regression model.
    test_data : pandas.DataFrame
        DataFrame containing the features (same as train) and 'npl' column for ground truth.

    Returns
    -------
    y_pred : numpy.ndarray
        Model predictions for 'npl', clipped at zero.
    mse : float
        Mean squared error of predictions vs ground truth.
    r2 : float
        R^2 score of predictions vs ground truth.
    """
    y_true = test_data['npl']                  # Ground truth
    x_test = test_data.drop(columns=['npl'])   # Test predictors

    # Predict on test data
    y_pred = model.predict(x_test)

    y_pred = np.maximum(y_pred, 0)             # Clip predictions at zero (no negative NPL values)
    
    # Compute evaluation metrics
    mse = mean_squared_error(y_true, y_pred)   # Mean squared error
    r2 = r2_score(y_true, y_pred)              # R^2 score
        
    return y_pred, mse, r2                     # Return predictions and metrics



def plot_nlb_and_eurostat_data(data, eurostat_datasets, output_path='/home/ivan/IskanjeDela/Banking/NLB/NLB_assignement_Kukuljan/TimeSeriesModelling/data/data_plots.png'):
    """
    Plots NLB NPL data and each Eurostat dataset as subplots, arranges them in a grid,
    and saves the figure to the specified output_path.
    
    Parameters:
        data: pd.DataFrame containing at least the column 'npl' and a date/time index.
        eurostat_datasets: dict of Eurostat dataset dicts, each with keys 'times' and 'values'.
        output_path: file path where the PNG image is saved.
    """
    num_plots = 1 + len(eurostat_datasets)
    cols = 5
    rows = (num_plots + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 6, rows * 4), squeeze=False)

    # First subplot: NLB NPL
    ax = axes[0, 0]
    # Ensure the index is parsed as datetime, if not already
    try:
        idx = pd.to_datetime(data.index.to_list())
    except Exception:
        idx = data.index.to_list()
    ax.plot(idx, data['npl'])
    ax.set_title('NLB NPL')
    ax.set_xlabel('Date')
    ax.set_ylabel('NPL')
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

    # The rest: Eurostat datasets
    plot_i = 1
    for key in eurostat_datasets:
        r = plot_i // cols
        c = plot_i % cols
        ax = axes[r, c]
        dataset = eurostat_datasets[key]
        times = dataset['times']
        values = dataset['values']

        freq = get_eurostat_time_unit(times[0])

        if freq == 'a':   # annual
            x_vals = pd.to_datetime(pd.Series(times).astype(str), format='%Y')
        elif freq == 'q': # quarterly like '2019-Q4'
            x_vals = pd.PeriodIndex(times, freq='Q').to_timestamp('Q')
        elif freq == 'm': # monthly like '2019-12'
            x_vals = pd.to_datetime(times, format='%Y-%m')
        else:
            x_vals = times  # fallback

        ax.plot(x_vals, values)
        ax.set_title(key)
        ax.set_xlabel('Time')
        ax.set_ylabel('Value')
        if isinstance(x_vals, (pd.Series, pd.DatetimeIndex, list)) and hasattr(x_vals, 'dtype') and str(x_vals.dtype).startswith('datetime'):
            ax.xaxis.set_major_locator(mdates.YearLocator())
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        plot_i += 1

    # Hide any unused subplots
    for n in range(num_plots, rows * cols):
        r = n // cols
        c = n % cols
        fig.delaxes(axes[r, c])

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.show()



# Plots the NPL values (actual and predicted) for training and two test splits
def plot_npl_predictions(data_train, data_test_1, data_test_2, model):
    """
    Plot NPL values for training and test data, including model predictions.

    Parameters
    ----------
    data_train : pd.DataFrame
        The training set (expects 'npl' column and datetime index).
    data_test_1 : pd.DataFrame
        First test split (expects 'npl' column and datetime index).
    data_test_2 : pd.DataFrame
        Second test split (expects 'npl' column and datetime index).
    model : fitted sklearn model
        Trained model on data_train; must be compatible with evaluate_linear_model.

    Returns
    -------
    None
        Displays and saves the resulting plots.
    """

    # Get model predictions and performance for the two test splits
    y_pred_test_1, mse_test_1, r2_test_1 = evaluate_linear_model(model, data_test_1)
    y_pred_test_2, mse_test_2, r2_test_2 = evaluate_linear_model(model, data_test_2)

    # Create figure with two stacked subplots (taller for training/test_1, shorter for test_2)
    fig, axes = plt.subplots(
        2, 1, figsize=(12, 8), sharex=False, gridspec_kw={'height_ratios': [2, 1]}
    )

    # --- Top subplot: Training set and first test split ---
    ax = axes[0]
    # Plot observed training NPL values
    ax.plot(
        data_train.index,
        data_train['npl'],
        color='black',
        label='train data',
        linewidth=2
    )
    # Plot observed first test split (ground truth)
    ax.scatter(
        data_test_1.index,
        data_test_1['npl'],
        color='green',
        label='test_1 ground truth',
        zorder=3
    )
    # Plot predicted values for first test split
    ax.scatter(
        data_test_1.index,
        y_pred_test_1,
        color='violet',
        label='test_1 predicted',
        marker='x',
        s=70,
        zorder=4
    )
    ax.set_title(
        f"Training and Test 1 Data with Predictions (MSE: {mse_test_1:.4f}, R²: {r2_test_1:.4f})"
    )
    ax.set_ylabel("NPL Value")
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

    # --- Bottom subplot: Second test split only ---
    ax2 = axes[1]
    # Plot observed second test split (ground truth)
    ax2.scatter(
        data_test_2.index,
        data_test_2['npl'],
        color='green',
        label='test_2 ground truth',
        zorder=3
    )
    # Plot predicted values for second test split
    ax2.scatter(
        data_test_2.index,
        y_pred_test_2,
        color='violet',
        label='test_2 predicted',
        marker='x',
        s=70,
        zorder=4
    )
    ax2.set_title(
        f"Test 2 Data with Predictions (MSE: {mse_test_2:.4f}, R²: {r2_test_2:.4f})"
    )
    ax2.set_ylabel("NPL Value")
    ax2.legend()
    ax2.grid(True, linestyle='--', alpha=0.5)
    ax2.xaxis.set_major_locator(mdates.YearLocator())
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax2.set_xlabel("Date")

    # Finalize layout and save/show the plot
    plt.tight_layout()
    # Save to file, ensure tight bounding box for legend etc.
    plt.savefig(
        '/home/ivan/IskanjeDela/Banking/NLB/NLB_assignement_Kukuljan/TimeSeriesModelling/data/npl_predictions.png',
        dpi=300,
        bbox_inches="tight",   # includes outside artists
        pad_inches=0.1
    )
    plt.show()

def plot_abs_coeff_series_by_dataset(
    model: LinearRegression,
    feature_names: Iterable[str],
    title: Optional[str] = "Absolute model coefficients by dataset/unit (log scale)",
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """
    Group linear model coefficients into time-delay series per (dataset name, unit),
    then plot the absolute values on a single log-y plot. Series are ordered by
    descending max |coef| so the most influential appears first.
    """
    # Check that the model is already fit and has .coef_
    if not hasattr(model, "coef_"):
        raise ValueError("Model must be fit and have a 'coef_' attribute.")

    # Convert coefficients and feature names to arrays/lists
    coefs = np.asarray(model.coef_, dtype=float)
    names = list(feature_names)

    # Check that we have as many names as coefficients
    if coefs.shape[0] != len(names):
        raise ValueError(
            f"Length mismatch: {coefs.shape[0]} coefficients vs {len(names)} feature names."
        )

    # This dictionary will group coefficients by (dataset_name, unit) and within that by delay
    grouped: Dict[Tuple[str, str], Dict[int, float]] = defaultdict(dict)

    # Iterate over features and coefficients, parsing the pattern of <dataset name>-<delay><unit>
    for name, coef in zip(names, coefs):
        try:
            dataset, delay, unit = extract_dataset_name_delay_and_unit(name)
        except ValueError:
            # Skip feature names that don't match the pattern
            continue
        # Store absolute value of coeff by (dataset, unit) and delay
        grouped[(dataset, unit)][delay] = float(abs(coef))

    # Raise if nothing parsable
    if not grouped:
        raise ValueError("No parsable feature names found to plot.")

    # For each series (dataset, unit), collect sorted delays and respective abs coefficients
    series = []
    for key, mapping in grouped.items():
        delays = sorted(mapping.keys())  # sorted list of lag values
        ys = [mapping[d] for d in delays]  # absolute coefficient magnitudes
        max_abs = max(ys) if ys else 0.0   # for ordering by importance
        series.append({"key": key, "x": delays, "y": ys, "max_abs": max_abs})

    # Order the series by their max (descending) so most important are plotted first
    series.sort(key=lambda s: s["max_abs"], reverse=True)

    # Create new axes if none are provided
    if ax is None:
        fig, ax = plt.subplots(figsize=(11, 6))

    # Prepare cycles of styles: line, color, marker.
    # Use matplotlib's prop cycle for visually distinct colors
    linestyles = ["-", "-.", ":"]  # 3 line styles
    prop_cycler = plt.rcParams["axes.prop_cycle"]
    colors = [d.get("color") for d in prop_cycler]
    markers = ["o", "s", "^"]  # 3 marker styles

    # Combine colors, linestyles, and markers for possible style permutations
    styles = [(c, linestyles[i], markers[i]) for c in colors for i in range(len(linestyles))]

    # Plot each series using a unique combination of style/color/marker
    for i, s in enumerate(series):
        color, ls, marker = styles[i % len(styles)]
        dataset, unit = s["key"]
        label = f"{dataset} ({unit})"
        ax.plot(
            s["x"],
            s["y"],
            label=label,
            linestyle=ls,
            color=color,
            marker=marker,
            linewidth=2,
            markersize=6,
            alpha=0.9,
        )

    # Log scale for Y axis (absolute magnitude of coefficients)
    ax.set_yscale("log")
    # X label indicates the lag/delay units
    ax.set_xlabel("Delay [in time units]")
    # Y label is log(absolute coefficient)
    ax.set_ylabel("|Coefficient| (log scale)")

    # Set plot title if specified
    if title:
        ax.set_title(title)

    # Enable grid for easier reading
    ax.grid(True, which="both", linewidth=0.5, alpha=0.5)

    # Add a legend outside right of plot for visibility
    ax.legend(
        loc="center left",           # location just outside plot
        bbox_to_anchor=(1.02, 0.5),  # x=1.02 breaks out of axes
        fontsize=8,
        ncol=1,
        frameon=False
    )

    # Save figure to disk with tight bounding box to include legend
    plt.savefig(
        '/home/ivan/IskanjeDela/Banking/NLB/NLB_assignement_Kukuljan/TimeSeriesModelling/data/model_coefficients.png',
        dpi=300,
        bbox_inches="tight",
        pad_inches=0.1
    )
    # Show the plot
    plt.show()
    # Return the axes for further use/modification if needed
    return ax




                


nlb_window = 2
months_window = 6
quarters_window = 2
annual_window = 1

print('Computing at:')
print(f'nlb_window: {nlb_window}')
# print(f'months_window: {months_window}')
print(f'quarters_window: {quarters_window}')
print(f'annual_window: {annual_window}')
print('')
print('Importing Eurostat files.')

for key in eurostat_datasets:
    dataset_dict = eurostat_datasets[key]
    dataset_dict = retrieve_eurostat_dataset(dataset_dict)
    eurostat_datasets[key] = dataset_dict
 

# print(eurostat_datasets.keys())

print('Composing model data')
# data = compose_modeling_data(eurostat_datasets, scale=True, impute=True, nlb_window = 2, months_window=6, quarters_window=2, annual_window=1)
data = compose_modeling_data(eurostat_datasets, scale=True, impute=True, nlb_window = nlb_window, months_window=months_window, quarters_window=quarters_window, annual_window=annual_window)

plot_nlb_and_eurostat_data(data, eurostat_datasets, output_path='/home/ivan/IskanjeDela/Banking/NLB/NLB_assignement_Kukuljan/TimeSeriesModelling/data/data_plots.png')

# print(data.columns)

print('Computing the train - test split')
data_train, data_test_1, data_test_2 = train_test_split(data, test_1_size=6, test_2_size=6, random_state=42)

# print('data_train dates:')
# print(data_train.index)
# print('data_test_1 dates:')
# print(data_test_1.index)
# print('data_test_2 dates:')
# print(data_test_2.index)

print('Fitting the model')
model = fit_linear_model_with_time_delay(data_train)
# print(model.coef_)
# print(model.intercept_)

print('Evaluating the model')
print('')
print(f'nlb_window: {nlb_window}')
# print(f'months_window: {months_window}')
print(f'quarters_window: {quarters_window}')
print(f'annual_window: {annual_window}')
print('')
print('Importing Eurostat files.')
print('')
print('Results')
y_pred_test_1, mse_test_1, r2_test_1 = evaluate_linear_model(model, data_test_1)
print(f'MSE test_1: {mse_test_1:.4f}')
print(f'R² test_1: {r2_test_1:.4f}')
y_pred_test_2, mse_test_2, r2_test_2 = evaluate_linear_model(model, data_test_2)
print(f'MSE test_2: {mse_test_2:.4f}')
print(f'R² test_2: {r2_test_2:.4f}')

plot_npl_predictions(data_train, data_test_1, data_test_2, model)

# print(len(data.columns))
# print(len(model.coef_))


plot_abs_coeff_series_by_dataset(model, feature_names=data.drop(columns=['npl']).columns)





