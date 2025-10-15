#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
from datetime import datetime
import re
import os
import matplotlib.dates as mdates
from datetime import  date


def get_basedir():
    try:
        # Works when running a script
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        # Fallback for notebooks or interactive mode
        BASE_DIR = os.getcwd()
    
    return BASE_DIR

def load_nlb_npl_data(nlb_npl_data_path=os.path.join(get_basedir(), 'npl_bank.xlsx')):
    nlb_npl_data = pd.read_excel(nlb_npl_data_path)
    nlb_dates = nlb_npl_data['DATE'].to_list()          # the index dates for our dataset
    nlb_npl_values = nlb_npl_data['NPL'].to_list()      # the corresponding NPL (target) values

    return nlb_dates, nlb_npl_values
    


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


def year_locator_for_span(dates, max_ticks=8):
    if not len(dates):
        return mdates.YearLocator(), mdates.DateFormatter('%Y')
    dmin, dmax = pd.to_datetime(dates[0]), pd.to_datetime(dates[-1])
    span_years = max(1, int(round((dmax - dmin).days / 365.25)))
    interval = max(1, int(round(span_years / max(1, max_ticks))))
    return mdates.YearLocator(interval), mdates.DateFormatter('%Y')
