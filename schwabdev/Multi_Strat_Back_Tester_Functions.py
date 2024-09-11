
import numpy as np
import random
import time
import pandas as pd
import matplotlib.pyplot as plt
import mplfinance as mpf
import pandas_ta as ta
from scipy import stats
import re

# Data Retrieval, Hopper Filtering, and Coefficient of Variance of Price Analysis

def get_all_tickers(project_id, dataset, table, start_date, end_date):
    """
    Fetches all unique ticker symbols and their corresponding min and max date_time,
    first and last close prices, mean and standard deviation of price and volume
    from the specified dataset in Google BigQuery.
    
    Parameters:
    - project_id (str): The Google Cloud project ID.
    - dataset (str): The dataset name in BigQuery.
    - table (str): The table name in BigQuery
    - start_date (str): The start date for filtering the data.
    - end_date (str): The end date for filtering the data.
    
    Returns:
    - pd.DataFrame: A DataFrame containing all unique ticker symbols and their min and max date_time,
                    min and max close prices, first and last close prices, mean and standard deviation of price and volume.
    """
    # SQL query to fetch all unique tickers with min and max date_time, first and last close prices, mean and standard deviation of price and volume
    sql = f"""
    WITH first_last_prices AS (
        SELECT 
            ticker,
            FIRST_VALUE(close) OVER (PARTITION BY ticker ORDER BY date_time) AS first_close_price,
            LAST_VALUE(close) OVER (PARTITION BY ticker ORDER BY date_time 
                ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING) AS last_close_price,
            ROW_NUMBER() OVER (PARTITION BY ticker ORDER BY date_time) AS rn
        FROM `{project_id}.{dataset}.{table}`
        WHERE date_time BETWEEN '{start_date}' AND '{end_date}'
    )
    SELECT 
        ticker, 
        MIN(date_time) AS min_date_time, 
        MAX(date_time) AS max_date_time,
        MIN(close) AS min_close_price, 
        MAX(close) AS max_close_price,
        MAX(first_close_price) AS first_close_price,
        MAX(last_close_price) AS last_close_price,
        AVG(close) AS mean_close_price,
        STDDEV(close) AS stddev_close_price,
        MIN(volume) AS min_volume, 
        MAX(volume) AS max_volume,        
        AVG(volume) AS mean_volume,
        STDDEV(volume) AS stddev_volume
    FROM `{project_id}.{dataset}.{table}`
    LEFT JOIN first_last_prices USING (ticker)
    WHERE date_time BETWEEN '{start_date}' AND '{end_date}'
    GROUP BY ticker
    """
    
    # Execute the query and fetch the data
    df = pd.read_gbq(query=sql, project_id=project_id)
    
    # Sort the DataFrame by ticker
    df = df.sort_values(by='ticker')
    
    return df

def cv_filtering(cv_filter_df, min_mean_close_price=3, max_mean_close_price=100, min_last_close_price=3, max_last_close_price=100, min_cv_close_price=0.2, min_mean_volume=1000000, max_mean_volume=100000000):
    """
    Filters and processes the DataFrame based on specific criteria.

    Parameters:
    - cv_filter_df (pd.DataFrame): DataFrame containing the raw data to be filtered and processed.
    - min_mean_close_price (float): Minimum mean close price for filtering.
    - max_mean_close_price (float): Maximum mean close price for filtering.
    - min_last_close_price (float): Minimum last close price for filtering.
    - max_last_close_price (float): Maximum last close price for filtering.
    - min_cv_close_price (float): Minimum coefficient of variation for close price.
    - min_mean_volume (int): Minimum mean volume for filtering.
    - max_mean_volume (int): Maximum mean volume for filtering.

    Returns:
    - pd.DataFrame: Filtered and processed DataFrame.
    """
    # Dropping rows with NaN values
    cv_filter_df.dropna(inplace=True)

    # Calculating Coefficient of Variation for close price and volume
    cv_filter_df['cv_close_price'] = cv_filter_df['stddev_close_price'] / cv_filter_df['mean_close_price']
    cv_filter_df['cv_volume'] = cv_filter_df['stddev_volume'] / cv_filter_df['mean_volume']

    # Filtering rows based on mean_close_price and last_close_price
    cv_filter_df = cv_filter_df[
        (cv_filter_df['mean_close_price'].between(min_mean_close_price, max_mean_close_price)) &
        (cv_filter_df['last_close_price'].between(min_last_close_price, max_last_close_price)) &
        (cv_filter_df['cv_close_price'] > min_cv_close_price) &
        (cv_filter_df['mean_volume'].between(min_mean_volume, max_mean_volume))
    ]

    # Filter rows with min_date_time and max_date_time values that disagree with the mode of those columns
    # mode_min_date_time = cv_filter_df['min_date_time'].mode()[0]
    mode_max_date_time = cv_filter_df['max_date_time'].mode()[0]
    cv_filter_df = cv_filter_df[
        # (cv_filter_df['min_date_time'] == mode_min_date_time) &
        (cv_filter_df['max_date_time'] == mode_max_date_time)
    ]

    # Sorting
    cv_filter_df = cv_filter_df.sort_values(by='cv_close_price', ascending=False)

    # Reset index
    cv_filter_df.reset_index(drop=True, inplace=True)

    return cv_filter_df

def plot_dual_distributions(df, x_var1, x_var2, y_var='Frequency'):
    """
    Plots the distributions of two specified columns on the same histogram with different colors.

    Parameters:
    - df (pd.DataFrame): DataFrame containing the columns to be plotted.
    - x_var1 (str): The name of the first column to be plotted.
    - x_var2 (str): The name of the second column to be plotted.
    - y_var (str): The label for the y-axis. Default is 'Frequency'.
    """
    plt.figure(figsize=(20, 12))

    # Plot histogram for the first variable
    plt.hist(df[x_var1], bins=100, alpha=0.5, label=x_var1, color='blue')

    # Plot histogram for the second variable
    plt.hist(df[x_var2], bins=100, alpha=0.5, label=x_var2, color='red')

    # Add title and labels
    plt.title(f'Distribution of {x_var1} and {x_var2}')
    plt.xlabel(f'{x_var1} and {x_var2}')
    plt.ylabel(y_var)

    # Add legend
    plt.legend()

    # Show plot
    plt.show()

def plot_scatter_with_size(df, x_var, y_var, z_var):
    """
    Plots a scatterplot with one x variable, one y variable, and one z variable expressed by the size of the data point mark.

    Parameters:
    - df (pd.DataFrame): DataFrame containing the columns to be plotted.
    - x_var (str): The name of the column to be used as the x variable.
    - y_var (str): The name of the column to be used as the y variable.
    - z_var (str): The name of the column to be used as the z variable (size of the data point mark).
    """
    plt.figure(figsize=(20, 12))

    # Apply square root transformation to the z variable to handle skewness
    transformed_z_var = np.sqrt(df[z_var])

    # Scale the transformed z variable for better size visibility
    size = 1000 * (transformed_z_var - transformed_z_var.min()) / (transformed_z_var.max() - transformed_z_var.min())

    # Plot scatterplot
    plt.scatter(df[x_var], df[y_var], s=size, alpha=0.5)

    # Add title and labels
    plt.title(f'Scatterplot of {x_var} vs {y_var} with {z_var} as size')
    plt.xlabel(x_var)
    plt.ylabel(y_var)

    # Show plot
    plt.show()

def get_data(ticker, project_id, dataset, table, start_date, end_date):
    """
    Fetches stock data for a given ticker and date range, then prepares the DataFrame.
    
    Parameters:
    - ticker (str): The ticker symbol in all caps of the stock to fetch.
    - start_date (str): The start date for the data in 'YYYY-MM-DD HH:MM:SS' format.
    - end_date (str): The end date for the data in 'YYYY-MM-DD HH:MM:SS' format.
    - project_id (str): The Google Cloud project ID. eg. 'stock-chipper-87578'
    - dataset (str) The asset class and time frame. eg. 'stocks_candle_5min'
    
    Returns:
    - pd.DataFrame: A DataFrame with the stock data, indexed by date_time.
    """
    # SQL query to fetch the data
    sql = f"""

    SELECT *
    FROM `{project_id}.{dataset}.{table}`

    WHERE 
        ticker = "{ticker}" AND
        date_time BETWEEN '{start_date}' AND '{end_date}';
    """
    
    # Execute the query and fetch the data
    df = pd.read_gbq(query=sql, project_id=project_id)
    
    # Convert the date_time column to datetime, sort, and set as index
    df['date_time'] = pd.to_datetime(df['date_time'])
    df = df.sort_values(by='date_time').reset_index(drop=True)
    df.set_index('date_time', inplace=True)
    
    return df

def calculate_monthly_stats(df, period='M'):
    """
    Calculate the mean, standard deviation, and coefficient of variation (CV) of the 'close' and 'volume' columns from month to month.

    Parameters:
    - df (pd.DataFrame): DataFrame containing the 'close' and 'volume' columns and 'date_time' index.

    Returns:
    - pd.DataFrame: DataFrame with columns 'date', 'mean_close', 'stddev_close', 'cv_close', 'mean_volume', 'stddev_volume', and 'cv_volume'.
    """
    # Ensure the index is a datetime index
    if not pd.api.types.is_datetime64_any_dtype(df.index):
        df.index = pd.to_datetime(df.index)

    # Resample the data by month and calculate the mean and standard deviation
    monthly_close_stats = df['close'].resample(period).agg(['mean', 'std'])
    monthly_volume_stats = df['volume'].resample(period).agg(['mean', 'std'])

    # Rename the columns
    monthly_close_stats.columns = ['mean_close_price', 'stddev_close_price']
    monthly_volume_stats.columns = ['mean_volume', 'stddev_volume']

    # Calculate the coefficient of variation (CV)
    monthly_close_stats['cv_close_price'] = (monthly_close_stats['stddev_close_price'] / monthly_close_stats['mean_close_price'])
    monthly_volume_stats['cv_volume'] = (monthly_volume_stats['stddev_volume'] / monthly_volume_stats['mean_volume'])

    # Add a 'date' column representing the month
    monthly_close_stats['date'] = monthly_close_stats.index
    monthly_volume_stats['date'] = monthly_volume_stats.index

    # Combine the close and volume stats into a single DataFrame
    monthly_stats = pd.merge(monthly_close_stats, monthly_volume_stats, on='date')

    # Reset the index to make 'date' a column
    monthly_stats.reset_index(drop=True, inplace=True)

    return monthly_stats

def plot_time_series(df, column_ot, x_axis=20, y_axis=10):
    """
    Plot the specified column over time, ensuring the y-axis starts at 0.

    Parameters:
    - df (pd.DataFrame): DataFrame containing the 'date' and the column to be plotted.
    - column_ot (str): The name of the column to be plotted over time.
    - x_axis (int): The width of the plot.
    - y_axis (int): The height of the plot.
    """
    plt.figure(figsize=(x_axis, y_axis))
    plt.plot(df['date'], df[column_ot], marker='o', linestyle='-')
    plt.title(f'{column_ot} Over Time')
    plt.xlabel('Date')
    plt.ylabel(f'{column_ot}')
    plt.ylim(bottom=0)  # Ensure the y-axis starts at 0
    plt.grid(True)
    plt.show()

def plot_time_series_multi(monthly_df_dict, column_ot, x_axis=20, y_axis=10):
    """
    Plot the coefficient of variation (cv_close) over time for all DataFrames in the dictionary.

    Parameters:
    - monthly_cv_df_dict (dict): Dictionary of DataFrames containing the 'date' and 'cv_close' columns.
    """
    plt.figure(figsize=(x_axis, y_axis))
    
    for ticker, df in monthly_df_dict.items():
        plt.plot(df['date'], df[column_ot], marker='o', linestyle='-', label=ticker)

    plt.title(f'{column_ot} Over Time for Multiple Tickers')
    plt.xlabel('Date')
    plt.ylabel(f'{column_ot}')
    plt.legend()
    plt.grid(True)
    plt.show()

def calculate_profits(df, cash_bet=1000):
    """
    Calculate the starting cash, percentage change, and profit for each period.

    Parameters:
    df (pd.DataFrame): DataFrame containing the 'close' price column.
    cash_bet (int or float): The equity allocation amount per trade.

    Returns:
    pd.DataFrame: DataFrame with added 'starting_cash', 'percent_change', and cash_vertical columns.
    """
    df['starting_cash'] = cash_bet # Setting the equity allocation amount per trade
    df['percent_change'] = df['close'].pct_change() # Calculating percentage change from one price to the next
    df['pct_change_plus_1'] = df['percent_change'] + 1
    df['cash_after_trade'] = df['starting_cash'] * df['pct_change_plus_1'] # Calculating gain/loss from period to period
    df['cash_vertical'] = df['cash_after_trade'] - 1000
    return df

def extract_date_time_components(df):
    """
    Extracts year, month, day, and time of day from the datetime index of a DataFrame 
    and creates new columns with those values.

    Parameters:
    - df (pd.DataFrame): The DataFrame containing the datetime index.

    Returns:
    - pd.DataFrame: The DataFrame with added columns for year, month, day, and time of day.
    """
    # Ensure the index is a datetime index
    if not pd.api.types.is_datetime64_any_dtype(df.index):
        raise ValueError("The DataFrame index must be a datetime index.")
    
    # Extract year, month, day, and time of day
    df['year'] = df.index.year
    df['month'] = df.index.month
    df['day'] = df.index.day
    df['time_of_day'] = df.index.time

    return df

# Indicator Calculations

def calculate_smas(df, price_col, start=10, end=300, step=10):
    """
    Calculate SMAs for the specified periods and add them to the DataFrame.

    Parameters:
    df (pd.DataFrame): DataFrame containing the 'close' price column.
    price_col (string): The OHLC component used to calculate the SMA.
    start (int): Starting period for SMA calculation.
    end (int): Ending period for SMA calculation.
    step (int): Step size for SMA calculation.

    Returns:
    pd.DataFrame: DataFrame with added SMA columns.
    """
    # Loop to calculate SMAs for the specified periods and add them to the DataFrame
    for period in range(start, end + 1, step):
        column_name = f'sma_{period}'  # Construct the column name based on the period
        if column_name not in df.columns:  # Check if the column already exists
            df[column_name] = df[price_col].rolling(window=period).mean()
            
    return df

def calculate_wmas(df, price_col, start=10, end=300, step=10):
    """
    Calculate WMAs for the specified periods and add them to the DataFrame.

    Parameters:
    df (pd.DataFrame): DataFrame containing the 'close' price column.
    start (int): Starting period for WMA calculation.
    end (int): Ending period for WMA calculation.
    step (int): Step size for WMA calculation.

    Returns:
    pd.DataFrame: DataFrame with added WMA columns.
    """
    # Loop to calculate WMAs for the specified periods and add them to the DataFrame
    for period in range(start, end + 1, step): # Starts at 10, ends at 100, steps by 10
        weights = np.arange(1, period + 1) # Creates an array of weights [1, 2, ..., period]
        column_name = f'wma_{period}' # Construct the column name for WMA based on the period
        if column_name not in df.columns:
            df[column_name] = df[price_col].rolling(window=period).apply(lambda prices: np.dot(prices, weights) / weights.sum(), raw=True)

    return df

def calculate_marsod(df, initial_periods, final_periods, steps):
    """
    Calculate multiple MARSOD (Moving Average Ribbon of Sum of Differences) indicators for a DataFrame.

    The MARSOD indicator is calculated by taking the differences between successive simple moving averages (SMAs) and summing them.
    This indicator provides insight into the momentum and trend strength by comparing different time periods' SMAs.

    Parameters:
    df (pd.DataFrame): DataFrame containing the columns for close prices and SMAs.
    initial_periods (list of int): List of initial periods for the first SMAs.
    final_periods (list of int): List of final periods for the last SMAs to be compared.
    steps (list of int): List of step sizes to define the range of periods for SMAs.

    The DataFrame should have columns named 'close' and 'sma_{period}' where period varies based on the SMAs used.

    Returns:
    pd.DataFrame: The input DataFrame with additional 'marsod_{initial_period}' columns representing the calculated MARSOD indicators.
    """
    for ip, fp, step in zip(initial_periods, final_periods, steps):
        marsod_col_name = f'marsod_{ip}'
        df[marsod_col_name] = 0

        for period_start in range(ip, fp, step):
            period_end = period_start + step
            column_name_start = f'sma_{period_start}'
            column_name_end = f'sma_{period_end}'
            df[marsod_col_name] += df[column_name_start] - df[column_name_end]

    return df

def calculate_weighted_marsod(df, initial_periods, final_periods, steps):
    """
    Calculate the Weighted MARSOD (Weighted Moving Average Ribbon of Sum of Differences) indicator for a DataFrame.

    The Weighted MARSOD indicator is calculated by taking the differences between successive weighted moving averages (WMAs) and summing them.
    This indicator provides insight into the momentum and trend strength by comparing different time periods' WMAs.

    Parameters:
    df (pd.DataFrame): DataFrame containing the columns for close prices and WMAs.
    initial_period (int): The initial period for the first WMA (default is 10).
    final_period (int): The final period for the last WMA to be compared (default is 40).
    step (int): The step size to define the range of periods for WMAs (default is 10).

    The DataFrame should have columns named 'close' and 'wma_{period}' where period varies based on the WMAs used.

    Returns:
    pd.DataFrame: The input DataFrame with an additional 'weighted_marsod' column representing the calculated Weighted MARSOD indicator.
    """
    for ip, fp, step in zip(initial_periods, final_periods, steps):
        weighted_marsod_col_name = f'weighted_marsod_{ip}'
        df[weighted_marsod_col_name] = 0
        
        for period_start in range(ip, fp, step):
            period_end = period_start + step
            column_name_start = f'wma_{period_start}'
            column_name_end = f'wma_{period_end}'
            df[weighted_marsod_col_name] += df[column_name_start] - df[column_name_end]

    return df

def calculate_sqz(df):
    """
    """
    df.ta.squeeze(high='high', low='low', close='close', keltner=20, bollinger=20, mamode='ema', append=True)
    df.rename(columns={'SQZ_20_2.0_20_1.5': 'sqz'}, inplace=True)
    df.drop(columns=['SQZ_ON', 'SQZ_OFF', 'SQZ_NO'], inplace=True)

    return df

def calculate_momentum(df, start, end, increment):
    """
    Calculates momentum indicators using the linear regression slope over specified rolling window periods.
    Adds new columns to the DataFrame with the calculated momentum for each period.

    Parameters:
    - df (pd.DataFrame): The DataFrame containing the 'ohlc_avg' column.

    Returns:
    - pd.DataFrame: The DataFrame with additional columns for each calculated momentum.
    """
    # Function to calculate the linear regression slope
    def linear_regression_slope(series):
        x = np.arange(len(series))
        y = series.values
        slope = np.polyfit(x, y, 1)[0]
        return slope

    # Iterate through the specified period lengths
    for period in range(start, end, increment):
        column_name_momentum = f'momentum_{period}'
        df[column_name_momentum] = df['ohlc_avg'].rolling(window=period).apply(linear_regression_slope, raw=False)

    return df

def calculate_rsi(df, price_metric):
    """
    Calculates the RSI for specified price metrics over different rolling windows.
    
    Parameters:
    - df (pd.DataFrame): The DataFrame containing the data.
    - price_metric (str): The column name to use for price data in the DataFrame.
    
    Returns:
    - pd.DataFrame: The DataFrame with new columns for each RSI calculation.
    """
    # Calculate price changes from one period to the next
    df['price_change'] = df[price_metric].diff()

    # Separate gains and losses
    df['gain'] = df['price_change'].clip(lower=0)  # Positive changes
    df['loss'] = -df['price_change'].clip(upper=0)  # Negative changes (as positive values)

    calculation_columns = ['price_change', 'gain', 'loss']
    # Iterate through rolling windows from 10 to 20 in increments of 2
    for period in range(6, 21, 2):  # End at 21 to include 20
        # Calculate the average gains and losses over each period
        df[f'avg_gain_{period}'] = df['gain'].rolling(window=period, min_periods=1).mean()
        df[f'avg_loss_{period}'] = df['loss'].rolling(window=period, min_periods=1).mean()

        # Calculate the Relative Strength (RS) and the RSI
        df[f'RS_{period}'] = df[f'avg_gain_{period}'] / df[f'avg_loss_{period}']
        df[f'RSI_{period}'] = 50 - (100 / (1 + df[f'RS_{period}']))

        # Keep track of the calculation columns to drop later
        calculation_columns.extend([f'avg_gain_{period}', f'avg_loss_{period}', f'RS_{period}'])

    # Drop the intermediate calculation columns
    df.drop(columns=calculation_columns, inplace=True)

    return df

def calculate_keltner_channels(df, price_col='close', high_col='high', low_col='low', period=20, multiplier=1.5):
    """
    Calculate the Keltner Channels for a given DataFrame.

    The Keltner Channels are volatility-based envelopes set above and below an exponential moving average (EMA).
    This function calculates the 20-period EMA, the Average True Range (ATR), and the upper and lower Keltner Channels.

    Parameters:
    df (pd.DataFrame): DataFrame containing the price data.
    price_col (str): Name of the column containing the close prices (default is 'close').
    high_col (str): Name of the column containing the high prices (default is 'high').
    low_col (str): Name of the column containing the low prices (default is 'low').
    period (int): Period for the EMA and ATR calculations (default is 20).
    multiplier (float): Multiplier for the ATR to calculate the Keltner Channels (default is 1.5).

    Returns:
    pd.DataFrame: The input DataFrame with additional columns for EMA, ATR, and Keltner Channels.
    """
    # Calculate the 20-period EMA for the 'Close' price
    df['ema_kc'] = df[price_col].ewm(span=period, adjust=False).mean()

    # Calculate the True Range
    df['high_low'] = df[high_col] - df[low_col]
    df['high_close'] = (df[high_col] - df[price_col].shift()).abs()
    df['low_close'] = (df[low_col] - df[price_col].shift()).abs()
    df['true_range'] = df[['high_low', 'high_close', 'low_close']].max(axis=1)

    # Calculate the ATR
    df['atr'] = df['true_range'].rolling(window=period).mean()

    # Calculate Keltner Channels
    df['upper_kc'] = df['ema_kc'] + multiplier * df['atr']
    df['lower_kc'] = df['ema_kc'] - multiplier * df['atr']

    # Drop the intermediate columns
    df.drop(['high_low', 'high_close', 'low_close', 'true_range'], axis=1, inplace=True)

    return df

def calculate_bollinger_bands(df, price_col='close', period=20, std_multiplier=2):
    """
    Calculate the Bollinger Bands for a given DataFrame.

    Bollinger Bands are a volatility indicator that consist of a middle band (EMA), an upper band, and a lower band.
    The upper and lower bands are calculated based on the standard deviation of the price from the EMA.

    Parameters:
    df (pd.DataFrame): DataFrame containing the price data.
    price_col (str): Name of the column containing the close prices (default is 'close').
    period (int): Period for the EMA and standard deviation calculations (default is 20).
    std_multiplier (float): Multiplier for the standard deviation to calculate the Bollinger Bands (default is 2).

    Returns:
    pd.DataFrame: The input DataFrame with additional columns for the Bollinger Bands.
    """
    # Calculate the 20-period SMA for the 'Close' price
    df['sma_bb'] = df[price_col].rolling(window=period).mean()

    # Calculate the rolling standard deviation (also over a 20-period window)
    df['std_bb'] = df[price_col].rolling(window=period).std()

    # Calculate the upper Bollinger Band
    df['upper_bb'] = df['sma_bb'] + (df['std_bb'] * std_multiplier)

    # Calculate the lower Bollinger Band
    df['lower_bb'] = df['sma_bb'] - (df['std_bb'] * std_multiplier)

    return df

def calculate_vwap_and_bands(df, high_col='high', low_col='low', close_col='close', volume_col='volume'):
    """
    Calculate VWAP and its associated bands for a given DataFrame.

    This function calculates the VWAP (Volume Weighted Average Price) and its associated bands based on the standard deviation of the typical price from the VWAP.

    Parameters:
    df (pd.DataFrame): DataFrame containing the price and volume data.
    high_col (str): Name of the column containing the high prices (default is 'high').
    low_col (str): Name of the column containing the low prices (default is 'low').
    close_col (str): Name of the column containing the close prices (default is 'close').
    volume_col (str): Name of the column containing the volume (default is 'volume').

    Returns:
    pd.DataFrame: The input DataFrame with additional columns for VWAP, standard deviation, and VWAP bands.
    """
    # Calculate the typical price
    df['typical_price'] = (df[high_col] + df[low_col] + df[close_col]) / 3

    # Calculate the product of typical price and volume
    df['tpv'] = df['typical_price'] * df[volume_col]

    # Group by date and then calculate the cumulative sums within each group
    df['cumulative_tpv'] = df.groupby(df.index.date)['tpv'].cumsum()
    df['cumulative_volume'] = df.groupby(df.index.date)['volume'].cumsum()

    # Calculate the VWAP, which evolves throughout each day and restarts the next day
    df['vwap'] = df['cumulative_tpv'] / df['cumulative_volume']

    # Calculate the squared deviation of typical price from VWAP
    df['squared_deviation'] = (df['typical_price'] - df['vwap']) ** 2

    # Group by date and calculate the cumulative sum of squared deviations
    df['cumulative_squared_deviation'] = df.groupby(df.index.date)['squared_deviation'].cumsum()

    # Calculate the variance by dividing the cumulative squared deviation by the number of observations
    df['variance'] = df['cumulative_squared_deviation'] / df.groupby(df.index.date).cumcount()  # Avoid division by zero

    # Standard deviation is the square root of variance
    df['std_dev'] = np.sqrt(df['variance'])

    # Calculate the bands
    df['vwap_band_0.5_up'] = df['vwap'] + (df['std_dev'] * 0.5)
    df['vwap_band_0.5_down'] = df['vwap'] - (df['std_dev'] * 0.5)
    df['vwap_band_1_up'] = df['vwap'] + (df['std_dev'] * 1)
    df['vwap_band_1_down'] = df['vwap'] - (df['std_dev'] * 1)
    df['vwap_band_1.5_up'] = df['vwap'] + (df['std_dev'] * 1.5)
    df['vwap_band_1.5_down'] = df['vwap'] - (df['std_dev'] * 1.5)
    df['vwap_band_2_up'] = df['vwap'] + (df['std_dev'] * 2)
    df['vwap_band_2_down'] = df['vwap'] - (df['std_dev'] * 2)
    df['vwap_band_2.5_up'] = df['vwap'] + (df['std_dev'] * 2.5)
    df['vwap_band_2.5_down'] = df['vwap'] - (df['std_dev'] * 2.5)
    df['vwap_band_3_up'] = df['vwap'] + (df['std_dev'] * 3)
    df['vwap_band_3_down'] = df['vwap'] - (df['std_dev'] * 3)

    # Drop intermediate columns to clean up the DataFrame
    df.drop(['typical_price', 'tpv', 'cumulative_tpv', 'cumulative_volume', 'squared_deviation', 'cumulative_squared_deviation', 'variance'], axis=1, inplace=True)

    return df

def calculate_pivot_points(df, high_col='high', low_col='low', close_col='close'):
    """
    Calculate Pivot Points and associated support and resistance levels for a given DataFrame.

    Pivot Points are technical analysis indicators used to determine the overall trend of the market over different time frames.

    Parameters:
    df (pd.DataFrame): DataFrame containing the price data.
    high_col (str): Name of the column containing the high prices (default is 'high').
    low_col (str): Name of the column containing the low prices (default is 'low').
    close_col (str): Name of the column containing the close prices (default is 'close').

    Returns:
    pd.DataFrame: The input DataFrame with additional columns for pivot points and support/resistance levels.
    """
    # Ensure DataFrame is sorted by index (date)
    df = df.sort_index()

    # Resample to get daily high, low, and close values, forward fill for Mondays
    daily_high = df[high_col].resample('B').max().shift(1).ffill(limit=2)
    daily_low = df[low_col].resample('B').min().shift(1).ffill(limit=2)
    daily_close = df[close_col].resample('B').last().shift(1).ffill(limit=2)

    # Calculate Pivot Point levels
    pivot_point = (daily_high + daily_low + daily_close) / 3
    r1 = (2 * pivot_point) - daily_low
    s1 = (2 * pivot_point) - daily_high
    r2 = pivot_point + (daily_high - daily_low)
    s2 = pivot_point - (daily_high - daily_low)
    r3 = daily_high + 2 * (pivot_point - daily_low)
    s3 = daily_low - 2 * (daily_high - pivot_point)

    # Create a new DataFrame for pivot points
    pivot_points = pd.DataFrame({
        'pivot_point': pivot_point,
        'r1': r1,
        's1': s1,
        'r2': r2,
        's2': s2,
        'r3': r3,
        's3': s3
    })

    # Map the pivot points DataFrame back to the original DataFrame using the date
    df['pivot_point'] = df.index.normalize().map(pivot_points['pivot_point'])
    df['r1'] = df.index.normalize().map(pivot_points['r1'])
    df['s1'] = df.index.normalize().map(pivot_points['s1'])
    df['r2'] = df.index.normalize().map(pivot_points['r2'])
    df['s2'] = df.index.normalize().map(pivot_points['s2'])
    df['r3'] = df.index.normalize().map(pivot_points['r3'])
    df['s3'] = df.index.normalize().map(pivot_points['s3'])

    return df

def calculate_adx(df, n=14):
    """
    Calculate the Average Directional Index (ADX), Positive Directional Indicator (+DI),
    and Negative Directional Indicator (-DI) for a given DataFrame.

    Parameters:
    - df (pd.DataFrame): DataFrame containing 'high', 'low', and 'close' prices.
    - n (int): The period over which to calculate the ADX, +DI, and -DI (default is 14).

    Returns:
    - pd.DataFrame: DataFrame with added columns 'ADX', '+DI', and '-DI'.
    """
    # Calculate the True Range (TR)
    df['tr'] = np.maximum(df['high'] - df['low'], 
                          np.maximum(abs(df['high'] - df['close'].shift(1)),
                                     abs(df['low'] - df['close'].shift(1))))
    
    # Calculate the Directional Movements (+DM and -DM)
    df['+dm'] = np.where((df['high'] - df['high'].shift(1)) > (df['low'].shift(1) - df['low']), 
                         np.maximum(df['high'] - df['high'].shift(1), 0), 0)
    df['-dm'] = np.where((df['low'].shift(1) - df['low']) > (df['high'] - df['high'].shift(1)), 
                         np.maximum(df['low'].shift(1) - df['low'], 0), 0)

    # Calculate the smoothed values of TR, +DM, and -DM
    df['tr_smooth'] = df['tr'].rolling(window=n).sum()
    df['+dm_smooth'] = df['+dm'].rolling(window=n).sum()
    df['-dm_smooth'] = df['-dm'].rolling(window=n).sum()

    # Calculate the +DI and -DI
    df['+di'] = 100 * (df['+dm_smooth'] / df['tr_smooth'])
    df['-di'] = 100 * (df['-dm_smooth'] / df['tr_smooth'])

    # Calculate the Directional Index (DX)
    df['dx'] = 100 * abs((df['+di'] - df['-di']) / (df['+di'] + df['-di']))

    # Calculate the ADX
    df['adx'] = df['dx'].rolling(window=n).mean()

    # Clean up intermediate columns (optional)
    df.drop(columns=['tr', '+dm', '-dm', 'tr_smooth', '+dm_smooth', '-dm_smooth', 'dx'], inplace=True)

    return df

def calculate_all_gradients(df, variables):
    def backward_difference(array):
        """
        Calculate the gradient of the array using backward differences.
        
        Parameters:
        - array (np.ndarray): Input array.
        
        Returns:
        - np.ndarray: Gradient array.
        """
        backward_diff = np.diff(array, prepend=array[0])
        return backward_diff

    def apply_gradients(data, variables):
        for var in variables:
            # Check if the variable contains numeric values
            if pd.api.types.is_numeric_dtype(data[var]):
                # Calculate true gradient
                true_gradient = np.gradient(data[var])
                data[var + '_true_derivative_1'] = true_gradient
                
                # Calculate shifted gradient
                shifted_gradient = np.roll(true_gradient, 1)
                shifted_gradient[-1] = np.nan  # Set the last value to NaN as it is shifted out of bounds
                data[var + '_shifted_derivative_1'] = shifted_gradient
                
                # Calculate backward gradient
                backward_gradient = backward_difference(data[var])
                data[var + '_backward_derivative_1'] = backward_gradient
                
                # Calculate second true gradient
                true_second_gradient = np.gradient(data[var + '_true_derivative_1'])
                data[var + '_true_derivative_2'] = true_second_gradient
                
                # Calculate shifted second gradient
                shifted_second_gradient_raw = np.gradient(shifted_gradient)
                shifted_second_gradient = np.roll(shifted_second_gradient_raw, 1)
                shifted_second_gradient[-1] = np.nan  # Set the last value to NaN as it is shifted out of bounds
                data[var + '_shifted_derivative_2'] = shifted_second_gradient
                
                # Calculate backward second gradient
                backward_second_gradient = backward_difference(data[var + '_backward_derivative_1'])
                data[var + '_backward_derivative_2'] = backward_second_gradient

        return data

    # Apply gradients directly without grouping since there's no 'ticker' grouping
    df_with_gradients = apply_gradients(df, variables)
    return df_with_gradients

# Signal Generation and Profit/Loss Calculation

def generate_trading_signals(df, variables):
    """
    Adds trading signal columns to the DataFrame for given variables, shifted by one row down.
    A signal value of 1 represents a long position (go long when the variable is positive the previous day),
    and -1 represents a short position (go short when the variable is negative the previous day).
    A value of 0 represents no action.

    Parameters:
    - df (pd.DataFrame): The DataFrame containing the variables.
    - variables (list of str): List of column names in df for which to generate trading signals.

    Returns:
    - pd.DataFrame: The DataFrame with additional columns for each variable's trading signals,
                    where each signal is shifted one row down.
    """
    for var in variables:
        # Create signal column for each variable
        df[f'signal_{var}'] = np.where(df[var] > 0, 1, np.where(df[var] < 0, -1, 0))
        # Shift signal columns down by one row
        # df[f'signal_{var}'] = df[f'signal_{var}'].shift(1)

    return df

def generate_dynamic_trading_signals(df, variables):
    """
    Generates trading signals based on the values of base columns and their first derivatives.
    A long signal (1) is generated if both values are positive, a short signal (-1) if both are negative,
    and a neutral signal (0) otherwise.

    Parameters:
    - df (pd.DataFrame): The DataFrame containing the columns.
    - variables (list of str): List of base column names to generate signals from.

    Returns:
    - pd.DataFrame: The DataFrame with additional columns for the trading signals.
    """
    for base_col_name in variables:
        # Construct derivative column name
        derivative_1_col_name = f'{base_col_name}_derivative_1'
        
        # Check if both columns exist in the DataFrame
        if base_col_name in df.columns and derivative_1_col_name in df.columns:
            # Generate the signals
            df[f'signal_{base_col_name}_&_{derivative_1_col_name}'] = np.where(
                (df[base_col_name] > 0) & (df[derivative_1_col_name] > 0), 1,
                np.where((df[base_col_name] < 0) & (df[derivative_1_col_name] < 0), -1, 0)
            )
    
    return df

def generate_dynamic_trading_signals_2(df, variables):
    """
    Generates trading signals based on the values of base columns and their first derivatives.
    A long signal (1) is generated if both values are positive, a short signal (-1) if both are negative,
    and a neutral signal (0) otherwise.

    Parameters:
    - df (pd.DataFrame): The DataFrame containing the columns.
    - variables (list of str): List of base column names to generate signals from.

    Returns:
    - pd.DataFrame: The DataFrame with additional columns for the trading signals.
    """
    for base_col_name in variables:
        # Construct derivative column name
        derivative_2_col_name = f'{base_col_name}_derivative_2'
        
        # Check if both columns exist in the DataFrame
        if base_col_name in df.columns and derivative_2_col_name in df.columns:
            # Generate the signals
            df[f'signal_{base_col_name}_&_{derivative_2_col_name}'] = np.where(
                (df[base_col_name] > 0) & (df[derivative_2_col_name] > 0), 1,
                np.where((df[base_col_name] < 0) & (df[derivative_2_col_name] < 0), -1, 0)
            )
    
    return df

def generate_dynamic_trading_signals_3(df, variables):
    """
    Generates trading signals based on the values of base columns and their first derivatives.
    A long signal (1) is generated if both values are positive, a short signal (-1) if both are negative,
    and a neutral signal (0) otherwise.

    Parameters:
    - df (pd.DataFrame): The DataFrame containing the columns.
    - variables (list of str): List of base column names to generate signals from.

    Returns:
    - pd.DataFrame: The DataFrame with additional columns for the trading signals.
    """
    for base_col_name in variables:
        # Construct derivative column names
        derivative_1_col_name =f'{base_col_name}_derivative_1'
        derivative_2_col_name = f'{base_col_name}_derivative_2'
        
        # Check if both columns exist in the DataFrame
        if base_col_name in df.columns and derivative_1_col_name in df.columns and derivative_2_col_name in df.columns:
            # Generate the signals
            df[f'signal_{derivative_1_col_name}_&_{derivative_2_col_name}'] = np.where(
                (df[derivative_1_col_name] > 0) & (df[derivative_2_col_name] > 0), 1,
                np.where((df[derivative_1_col_name] < 0) & (df[derivative_2_col_name] < 0), -1, 0)
            )
    
    return df

def rsi_trigger_based_trading(df, upper_trigger=75, lower_trigger=25, suffix=""):
    """
    Generates trading signals based on multiple RSI columns with state management for entering and exiting trades.
    - Arm short position when RSI > upper_trigger and enter short when RSI < upper_trigger.
    - Close short position when RSI < lower_trigger or RSI > upper_trigger before reaching lower_trigger.
    - Arm long position when RSI < lower_trigger and enter long when RSI > lower_trigger.
    - Close long position when RSI > upper_trigger or RSI < lower_trigger before reaching upper_trigger.

    Parameters:
    - df (pd.DataFrame): The DataFrame containing 'RSI_' columns.
    - upper_trigger (int): Upper trigger threshold for RSI.
    - lower_trigger (int): Lower trigger threshold for RSI.
    - suffix (str): Suffix to add to the signal column names for distinguishing different trigger sets.

    Returns:
    - pd.DataFrame: The DataFrame with additional 'signal_' columns indicating the current position for each RSI column.
    """
    columns_rsi = [col for col in df.columns if col.startswith('RSI_') and 'derivative' not in col]
    
    for rsi_col in columns_rsi:
        signal_col = f'signal_{rsi_col}_{suffix}'
        
        # Convert the column to a numpy array for faster processing
        rsi_values = df[rsi_col].values
        signals = np.zeros_like(rsi_values)
        
        # Initialize states
        armed_short = False
        armed_long = False
        position = 0  # 0 = neutral, 1 = long, -1 = short

        # Iterate through the numpy array
        for i in range(1, len(rsi_values)):
            if armed_short:
                if rsi_values[i] < upper_trigger:
                    position = -1
                    armed_short = False
            elif armed_long:
                if rsi_values[i] > lower_trigger:
                    position = 1
                    armed_long = False
            elif position == -1:
                if rsi_values[i] < lower_trigger:
                    position = 0
                elif rsi_values[i] > upper_trigger:
                    position = 0
            elif position == 1:
                if rsi_values[i] > upper_trigger:
                    position = 0
                elif rsi_values[i] < lower_trigger:
                    position = 0

            # else: # This piece disables frequent rearming when RSI oscillates around trigger levels
            #     if rsi_values[i] > upper_trigger:
            #          armed_short = True
            #     elif rsi_values[i] < lower_trigger:
            #          armed_long = True
            
            # Allow re-arming after disarming
            if position == 0:
                if rsi_values[i] > upper_trigger:
                    armed_short = True
                elif rsi_values[i] < lower_trigger:
                    armed_long = True
            
            signals[i] = position
        
        # Assign the computed signals to the DataFrame
        df[signal_col] = signals

    return df

def moving_average_crossover_strategy(df, short_ma, long_ma):
    """
    Implements a moving average crossover trading strategy.

    Parameters:
    - df (pd.DataFrame): DataFrame containing the trading data and moving averages.
    - short_ma (str): The column name of the short-term moving average.
    - long_ma (str): The column name of the long-term moving average.

    Returns:
    - np.ndarray: Signals for the moving average crossover strategy.
    """
    short_ma_values = df[short_ma].values
    long_ma_values = df[long_ma].values

    signals = np.zeros(len(df))
    signals[short_ma_values > long_ma_values] = 1
    signals[short_ma_values < long_ma_values] = -1

    # Ensure there are no signals where one of the MAs is missing
    signals[np.isnan(short_ma_values) | np.isnan(long_ma_values)] = 0

    # # Shift the signals to prevent look-ahead bias
    # signals = np.roll(signals, 1)
    # signals[0] = 0  # First element shifted incorrectly

    return signals

def mean_reversion_strategy(df, short_ma, long_ma):
    """
    Implements a mean reversion trading strategy.

    Parameters:
    - df (pd.DataFrame): DataFrame containing the trading data and moving averages.
    - short_ma (str): The column name of the short-term moving average.
    - long_ma (str): The column name of the long-term moving average.

    Returns:
    - np.ndarray: Signals for the mean reversion strategy.
    """
    short_ma_values = df[short_ma].values
    long_ma_values = df[long_ma].values

    signals = np.zeros(len(df))
    signals[short_ma_values < long_ma_values] = 1
    signals[short_ma_values > long_ma_values] = -1

    # Ensure there are no signals where one of the MAs is missing
    signals[np.isnan(short_ma_values) | np.isnan(long_ma_values)] = 0

    # # Shift the signals to prevent look-ahead bias
    # signals = np.roll(signals, 1)
    # signals[0] = 0  # First element shifted incorrectly

    return signals

# def hybrid_strategy(df, short_ma, long_ma, atr_threshold_col):
#     """
#     Combines a moving average crossover strategy with a mean reversion strategy based on ATR.

#     Parameters:
#     - df (pd.DataFrame): DataFrame containing the trading data.
#     - short_ma (str): The column name of the short-term moving average.
#     - long_ma (str): The column name of the long-term moving average.
#     - atr_threshold_col (str): The column name of the ATR threshold to switch between strategies.

#     Returns:
#     - pd.DataFrame: DataFrame with new columns for trading signals.
#     """
#     atr_threshold = df[atr_threshold_col].iloc[0]  # Get the ATR threshold value

#     # Calculate signals for both strategies
#     trend_signals = moving_average_crossover_strategy(df, short_ma, long_ma)
#     reversion_signals = mean_reversion_strategy(df, short_ma, long_ma)
    
#     # Determine which strategy to use based on ATR
#     signals = np.where(df['atr'] > atr_threshold, trend_signals, reversion_signals)
    
#     # Add the signals to the DataFrame
#     signal_col_name = f'signal_{short_ma}_X_{long_ma}_crossrev_{atr_threshold_col}'
#     df[signal_col_name] = signals

#     return df

def hybrid_strategy(df, short_ma, long_ma, atr_threshold_col):
    """
    Combines a moving average crossover strategy with a mean reversion strategy based on ATR.
    Goes market neutral at the end of every trading session.

    Parameters:
    - df (pd.DataFrame): DataFrame containing the trading data.
    - short_ma (str): The column name of the short-term moving average.
    - long_ma (str): The column name of the long-term moving average.
    - atr_threshold_col (str): The column name of the ATR threshold to switch between strategies.

    Returns:
    - pd.DataFrame: DataFrame with new columns for trading signals.
    """
    atr_threshold = df[atr_threshold_col].iloc[0]  # Get the ATR threshold value

    # Calculate signals for both strategies
    trend_signals = moving_average_crossover_strategy(df, short_ma, long_ma)
    reversion_signals = mean_reversion_strategy(df, short_ma, long_ma)
    
    # Determine which strategy to use based on ATR
    signals = np.where(df['atr'] > atr_threshold, trend_signals, reversion_signals)
    
    # Go market neutral at the end of each trading day
    df['date'] = df.index.date  # Extract date from datetime index
    last_time_each_day = df.groupby('date').tail(1).index  # Find last timestamp of each trading day
    
    # Create a boolean mask to set signals to 0 at the end of each trading day
    mask = df.index.isin(last_time_each_day)
    signals[mask] = 0

    # Add the signals to the DataFrame
    signal_col_name = f'signal_{short_ma}_X_{long_ma}_crossrev_{atr_threshold_col}'
    df[signal_col_name] = signals

    # Clean up the temporary 'date' column
    df.drop(columns=['date'], inplace=True)

    return df


def calculate_atr_statistics(df):
    """
    Calculate and add various ATR statistics to the DataFrame.

    Parameters:
    - df (pd.DataFrame): The DataFrame containing the ATR data.

    Returns:
    - pd.DataFrame: DataFrame with added ATR statistics.
    """
    atr = df['atr'].dropna()

    # Calculate statistics
    df['atr_mean'] = atr.mean()
    df['atr_std_dev'] = atr.std()
    atr_percentiles = np.percentile(atr, [1, 5, 10, 25, 50, 75, 90, 95, 99])
    df['atr_median'] = atr.median()
    df['atr_mode'] = stats.mode(atr, keepdims=True).mode[0]
    df['atr_maximum'] = atr.max()
    df['atr_minimum'] = atr.min()

    # Add the percentiles to the DataFrame
    percentile_columns = ['atr_percentile_1', 'atr_percentile_5', 'atr_percentile_10', 'atr_percentile_25', 
                          'atr_percentile_50', 'atr_percentile_75', 'atr_percentile_90', 'atr_percentile_95', 
                          'atr_percentile_99']

    for col, percentile in zip(percentile_columns, atr_percentiles):
        df[col] = percentile

    # Print the statistics
    print(f"Mean: {df['atr_mean'].iloc[0]}")
    print(f"Standard Deviation: {df['atr_std_dev'].iloc[0]}")
    print(f"1st Percentile: {df['atr_percentile_1'].iloc[0]}")
    print(f"5th Percentile: {df['atr_percentile_5'].iloc[0]}")
    print(f"10th Percentile: {df['atr_percentile_10'].iloc[0]}")
    print(f"25th Percentile: {df['atr_percentile_25'].iloc[0]}")
    print(f"50th Percentile (Median): {df['atr_percentile_50'].iloc[0]}")
    print(f"75th Percentile: {df['atr_percentile_75'].iloc[0]}")
    print(f"90th Percentile: {df['atr_percentile_90'].iloc[0]}")
    print(f"95th Percentile: {df['atr_percentile_95'].iloc[0]}")
    print(f"99th Percentile: {df['atr_percentile_99'].iloc[0]}")
    print(f"Mode: {df['atr_mode'].iloc[0]}")
    print(f"Max: {df['atr_maximum'].iloc[0]}")
    print(f"Min: {df['atr_minimum'].iloc[0]}")

    return df

def calculate_position_percent_changes(df, price_col='close'):
    signal_cols = [col for col in df.columns if col.startswith('signal_')]
    
    for signal_col in signal_cols:
        change_col = signal_col.replace('signal_', 'position_percent_change_')
        df[change_col] = 0.0

        signal_changes = df[signal_col].ne(df[signal_col].shift()).cumsum()

        start_prices = df.groupby(signal_changes)[price_col].transform('first')
        
        # Calculate percentage changes in a vectorized manner
        pct_changes = (df[price_col].shift(-1) - start_prices) / start_prices

        # Ensure only the last row in each group gets the pct_change value
        is_last_in_group = df.groupby(signal_changes).cumcount(ascending=False) == 0
        df.loc[is_last_in_group, change_col] = pct_changes[is_last_in_group]

        # Handle the final group separately
        last_group = df.groupby(signal_changes).tail(1).index
        for idx in last_group:
            if idx == df.index[-1]:
                end_price = df.loc[idx, price_col]
                start_price = start_prices[idx]
                pct_change = (end_price - start_price) / start_price
                df.loc[idx, change_col] = pct_change

    return df

def calculate_profit_loss_contributions(df, cash_bet=1000):
    """
    Calculates the contributions to profit/loss for each trading signal by multiplying
    each 'signal_' column by the corresponding 'position_percent_change_' column. The result is stored 
    in new columns with 'profit_loss_' replacing 'signal_' in the column names.

    Parameters:
    - df (pd.DataFrame): The DataFrame containing the 'signal_' columns and 'position_percent_change_' columns.
    - cash_bet: The nominal dollar amount to be allocated to each new trade

    Returns:
    - pd.DataFrame: The DataFrame with additional columns showing the profit/loss contributions.
    """
    # Find all columns starting with 'signal_' and 'position_percent_change_'
    columns_signal = [col for col in df.columns if col.startswith('signal_')]
    columns_position_percent_change = [col for col in df.columns if col.startswith('position_percent_change_')]
    
    # Calculate profit/loss contributions for each signal column
    for column_sig, column_ppc in zip(columns_signal, columns_position_percent_change):
        new_col_name = column_sig.replace('signal_', 'profit_loss_')
        new_col_name_2 = column_sig.replace('signal_', 'intratrade_profit_loss_')
        df[new_col_name] = (((df[column_sig] * df[column_ppc]) + 1) * cash_bet) - cash_bet
        df[new_col_name_2] = df[column_sig] * df['cash_vertical']
    
    return df

# Visualizing Monthly CV Against Monthly Profit by Scatterplot

def calculate_monthly_stats_2(df, time_frame, indicator):
    """
    Calculate the mean, standard deviation, and coefficient of variation (CV) of the 'close' column from month to month.
    Also calculates the monthly profit and the coefficient of variation (CV) of the 'volume' column.

    Parameters:
    - df (pd.DataFrame): DataFrame containing the 'close' column and 'date_time' index.
    - time_frame (str): The resampling frequency (e.g., 'M' for month, 'Q' for quarter).
    - indicator (str): The suffix for the profit column.

    Returns:
    - pd.DataFrame: DataFrame with columns 'date', 'mean_close', 'stddev_close', 'cv_close', 'monthly_profit', 'cv_volume'.
    """
    # Ensure the index is a datetime index
    if not pd.api.types.is_datetime64_any_dtype(df.index):
        df.index = pd.to_datetime(df.index)

    # Resample the data by the specified time frame and calculate the mean and standard deviation for 'close'
    monthly_stats_close = df['close'].resample(time_frame).agg(['mean', 'std'])
    monthly_stats_close.columns = ['mean_close', 'stddev_close']
    monthly_stats_close['cv_close'] = monthly_stats_close['stddev_close'] / monthly_stats_close['mean_close']

    # Resample the data by the specified time frame and calculate the mean and standard deviation for 'volume'
    monthly_stats_volume = df['volume'].resample(time_frame).agg(['mean', 'std'])
    monthly_stats_volume.columns = ['mean_volume', 'stddev_volume']
    monthly_stats_volume['cv_volume'] = monthly_stats_volume['stddev_volume'] / monthly_stats_volume['mean_volume']

    # Merge the close and volume stats
    monthly_stats = monthly_stats_close.join(monthly_stats_volume)

    # Add a 'date' column representing the time frame
    monthly_stats['date'] = monthly_stats.index

    # Reset the index to make 'date' a column
    monthly_stats.reset_index(drop=True, inplace=True)

    # Resample profit by the specified time frame with sum aggregation
    profit_column = f'profit_loss_{indicator}'
    if profit_column in df.columns:
        monthly_profit = df[profit_column].resample(time_frame).sum()
        monthly_stats['monthly_profit'] = monthly_profit.values
    else:
        monthly_stats['monthly_profit'] = np.nan  # Assign NaN if the profit column is not found

    monthly_stats['constant'] = 1

    return monthly_stats

def plot_scatter_with_size_2(monthly_df_dict, x_var, y_var, z_var=None, x_threshold=None):
    """
    Plots a scatterplot with one x variable, one y variable, and optionally one z variable expressed by the size of the data point mark.
    Each ticker's dots will be in a different color.

    Parameters:
    - monthly_df_dict (dict of pd.DataFrame): Dictionary containing DataFrames with the columns to be plotted.
    - x_var (str): The name of the column to be used as the x variable.
    - y_var (str): The name of the column to be used as the y variable.
    - z_var (str, optional): The name of the column to be used as the z variable (size of the data point mark). Default is None.
    - x_threshold (float, optional): The threshold value for the x variable. Points with x values beyond this threshold will be excluded.
    """
    plt.figure(figsize=(20, 10))
    colors = plt.cm.get_cmap('tab10', len(monthly_df_dict))  # Get a color map with enough colors

    for i, (ticker, df) in enumerate(monthly_df_dict.items()):
        if x_threshold is not None:
            df = df[df[x_var] <= x_threshold]  # Exclude points with x values beyond the threshold

        if z_var:
            # Apply square root transformation to the z variable to handle skewness
            transformed_z_var = np.sqrt(df[z_var])

            # Scale the transformed z variable for better size visibility
            size = 1000 * (transformed_z_var - transformed_z_var.min()) / (transformed_z_var.max() - transformed_z_var.min())

            # Plot scatterplot for each ticker with z_var
            plt.scatter(df[x_var], df[y_var], s=size, alpha=0.5, label=ticker, color=colors(i))
        else:
            # Plot scatterplot for each ticker without z_var
            plt.scatter(df[x_var], df[y_var], alpha=0.5, label=ticker, color=colors(i))

    # Add title and labels
    plt.title(f'Scatterplot of {x_var} vs {y_var}' + (f' with {z_var} as size' if z_var else '') + ' for Multiple Tickers')
    plt.xlabel(x_var)
    plt.ylabel(y_var)

    # Add legend
    plt.legend()

    # Show plot
    plt.show()

# Performance Summaries

def create_sum_dataframe(df, profit_col):
    """
    Creates a DataFrame containing the sums of columns that start with 'gain_loss_',
    where sums are calculated and sorted numerically, then formatted in US dollars.
    The 'gain_loss_' prefix from the column names is also removed.
    Adds additional columns for trading statistics and average position durations.
    """
    # Creating column lists for iteration
    columns_signal = [col for col in df.columns if col.startswith('signal_')]
    columns_position_percent_change = [col for col in df.columns if col.startswith('position_percent_change_')]
    columns_profit_loss = [col for col in df.columns if col.startswith('profit_loss_')]
    columns_intratrade_profit_loss = [col for col in df.columns if col.startswith('intratrade_profit_loss_')]

    # Initiating lists to store metadata
    indicators = []
    position_profit_list = []
    total_trades_list = []
    profit_per_trade_list = []
    adjusted_lows = []
    adjusted_mids = []
    adjusted_highs = []
    winning_longs_list = []
    losing_longs_list = []
    winning_shorts_list = []
    losing_shorts_list = []
    max_drawdowns_list = []
    intratrade_drawdowns_list = []
    total_periods_long_list = []
    total_periods_neutral_list = []
    total_periods_short_list = []
    captured_c2c_verticality_list = []
    avg_long_duration_list = []
    avg_short_duration_list = []
    avg_neutral_duration_list = []

    ticker_name = df["ticker"].iloc[0]
    c2c_verticality = df[profit_col].abs().sum()
    print(f'Total close-to-close verticality for {ticker_name} is {c2c_verticality}')
    
    # Creating all of our metadata
    for column_sig, column_ppc, column_pl, column_itpl in zip(columns_signal, columns_position_percent_change, columns_profit_loss, columns_intratrade_profit_loss):
        clean_column_name = column_sig.replace('signal_', '')
        total_position_profit = df[column_pl].sum()

        signals = df[column_sig]
        ppcs = df[column_ppc]

        total_trades = (ppcs != 0).sum()
        winning_longs = ((signals > 0) & (ppcs > 0)).sum()
        losing_longs = ((signals > 0) & (ppcs < 0)).sum()
        winning_shorts = ((signals < 0) & (ppcs < 0)).sum()
        losing_shorts = ((signals < 0) & (ppcs > 0)).sum()

        # total_trades = 0
        # winning_longs = 0
        # losing_longs = 0
        # winning_shorts = 0 
        # losing_shorts = 0
        
        # for _, row in df.iterrows():
        #     sig = row[column_sig]
        #     ppc = row[column_ppc]
            
        #     if ppc != 0:
        #         total_trades += 1
        #         if sig > 0 and ppc > 0:
        #             winning_longs += 1
        #         elif sig > 0 and ppc < 0:
        #             losing_longs += 1
        #         elif sig < 0 and ppc < 0:
        #             winning_shorts += 1
        #         elif sig < 0 and ppc > 0:
        #             losing_shorts += 1x

        profit_per_trade = total_position_profit / total_trades if total_trades != 0 else 0
        adjusted_low = total_position_profit * 0.5
        adjusted_mid = total_position_profit * 0.7
        adjusted_high = total_position_profit * 0.9

        cumulative_profit_loss = df[column_pl].cumsum()
        running_max = cumulative_profit_loss.cummax()
        drawdown = running_max - cumulative_profit_loss 
        max_drawdown = drawdown.max()

        it_cumulative_profit_loss = df[column_itpl].cumsum()
        it_running_max = it_cumulative_profit_loss.cummax()
        it_drawdown = it_running_max - it_cumulative_profit_loss 
        intratrade_drawdown = it_drawdown.max()

        # df['cumulative_profit_loss'] = df[column_pl].cumsum()
        # df['running_max'] = df['cumulative_profit_loss'].cummax()
        # df['drawdown'] = df['running_max'] - df['cumulative_profit_loss'] 
        # max_drawdown = df['drawdown'].max()
        # df.drop(['cumulative_profit_loss', 'running_max', 'drawdown'], axis=1, inplace=True)

        # df['it_cumulative_profit_loss'] = df[column_itpl].cumsum()
        # df['it_running_max'] = df['it_cumulative_profit_loss'].cummax()
        # df['it_drawdown'] = df['it_running_max'] - df['it_cumulative_profit_loss'] 
        # intratrade_drawdown = df['it_drawdown'].max()
        # df.drop(['it_cumulative_profit_loss', 'it_running_max', 'it_drawdown'], axis=1, inplace=True)

        total_periods_long = (signals > 0).sum()
        total_periods_short = (signals < 0).sum()
        total_periods_neutral = (signals == 0).sum()

        captured_c2c_verticality = total_position_profit / c2c_verticality

        # Calculate average durations
        long_durations = signals[signals > 0].groupby((signals != signals.shift()).cumsum()).transform('size').unique()
        short_durations = signals[signals < 0].groupby((signals != signals.shift()).cumsum()).transform('size').unique()
        neutral_durations = signals[signals == 0].groupby((signals != signals.shift()).cumsum()).transform('size').unique()

        avg_long_duration = long_durations.mean() if len(long_durations) > 0 else 0
        avg_short_duration = short_durations.mean() if len(short_durations) > 0 else 0
        avg_neutral_duration = neutral_durations.mean() if len(neutral_durations) > 0 else 0

        # Appending all algo metadata to existing lists
        indicators.append(clean_column_name)
        position_profit_list.append(total_position_profit)
        total_trades_list.append(total_trades)
        profit_per_trade_list.append(profit_per_trade)
        adjusted_lows.append(adjusted_low)
        adjusted_mids.append(adjusted_mid)
        adjusted_highs.append(adjusted_high)        
        winning_longs_list.append(winning_longs)
        losing_longs_list.append(losing_longs)
        winning_shorts_list.append(winning_shorts)
        losing_shorts_list.append(losing_shorts)
        max_drawdowns_list.append(max_drawdown)
        intratrade_drawdowns_list.append(intratrade_drawdown)
        total_periods_long_list.append(total_periods_long)
        total_periods_neutral_list.append(total_periods_neutral)
        total_periods_short_list.append(total_periods_short)
        captured_c2c_verticality_list.append(captured_c2c_verticality)
        avg_long_duration_list.append(avg_long_duration)
        avg_short_duration_list.append(avg_short_duration)
        avg_neutral_duration_list.append(avg_neutral_duration)

    # Creating metadataframe 
    sum_df = pd.DataFrame({
        'Indicator': indicators,
        'Profit': position_profit_list,
        'Trades': total_trades_list,
        'Profit_Per_Trade': profit_per_trade_list,
        'Adjusted_Low_50': adjusted_lows,
        'Adjusted_Mid_70': adjusted_mids,
        'Adjusted_High_90': adjusted_highs,        
        'Winning_Longs': winning_longs_list,
        'Losing_Longs': losing_longs_list,
        'Winning_Shorts': winning_shorts_list,
        'Losing_Shorts': losing_shorts_list,
        'Max_Drawdown': max_drawdowns_list,
        'Intratrade_Max_Drawdown': intratrade_drawdowns_list,
        'Periods_Long': total_periods_long_list,
        'Periods_Neutral': total_periods_neutral_list,
        'Periods_Short': total_periods_short_list,
        'Captured_Verticality': captured_c2c_verticality_list,
        'Avg_Long_Duration': avg_long_duration_list,
        'Avg_Short_Duration': avg_short_duration_list,
        'Avg_Neutral_Duration': avg_neutral_duration_list
    })

    sum_df.insert(0, 'Ticker', ticker_name)

    return sum_df

# The Performance Summary Data for All Strategies and All Tickers

def build_profit_cv_df(df, performance_df_dict, indicator_value):
    """
    Builds a summary DataFrame by extracting 'Indicator', 'Profit', and 'cv_close_price' for a given 'Indicator' value
    from each DataFrame in the provided dictionary, along with the corresponding ticker.

    Parameters:
    - performance_df_dict (dict): A dictionary of DataFrames containing performance data.
    - indicator_value (str): The value under the 'Indicator' column to search for in each DataFrame.

    Returns:
    - pd.DataFrame: A DataFrame containing the extracted information.
    """
    data = [
        {
            'Ticker': ticker,
            'Indicator': indicator_value,
            'Profit': df.loc[df['Indicator'] == indicator_value, 'Profit'].values[0],
            'cv_close_price': df.loc[df['Indicator'] == indicator_value, 'cv_close_price'].values[0],
            'cv_volume': df.loc[df['Indicator'] == indicator_value, 'cv_volume'].values[0]
        }
        for ticker, df in performance_df_dict.items()
        if indicator_value in df['Indicator'].values
    ]
    return pd.DataFrame(data)

# Profit Curve and Trade Visualizations

def cumulative_profit_chart_2(df, pl_col_type, indicator, suffix, x_axis=10, y_axis=6):
    """
    """
    columns_profit_loss = [col for col in df.columns if col.startswith(f'{pl_col_type}{indicator}') and col.endswith(f'{suffix}')]

    for column_idpl in columns_profit_loss:
        # Calculate cumulative profit
        df['cumulative_profit'] = df[column_idpl].cumsum()
        df['cumulative_profit_adjusted'] = df['cumulative_profit'] * 0.5

        # Plotting the line chart
        plt.figure(figsize=(x_axis, y_axis))
        plt.plot(df.index, df['cumulative_profit'], linestyle='-', color='b')
        plt.plot(df.index, df['cumulative_profit_adjusted'], linestyle='-', color='g')

        # Adding title and labels
        plt.title(f'Cumulative Profit Over Time for {column_idpl}')
        plt.xlabel('Date')
        plt.ylabel('Cumulative Profit')

        # Display the plot
        plt.grid(True)
        plt.show()   

def smooth_column(df, column, window_size=2):
    """
    Applies a rolling mean to a specified column in a pandas DataFrame.

    Parameters:
    - df (pd.DataFrame): The DataFrame containing the data.
    - column (str): The name of the column to apply the smoothing to.
    - window_size (int): The window size for the rolling mean. Defaults to 2.

    Returns:
    - pd.DataFrame: The DataFrame with the smoothed column added. The new column's name is prefixed with 'smooth_'.
    """
    if column in df.columns:
        # Create the smoothed column
        new_column_name = f'smooth_{column}'
        df[new_column_name] = df[column].rolling(window=window_size).mean()
    else:
        print(f"Column '{column}' does not exist in the DataFrame.")
    return df

# External Viz Functions/Newest Trade Plot

def add_pivot_points(df_timeframe):
    """
    Adds pivot point lines to the plot.

    Parameters:
    - df_timeframe (pd.DataFrame): DataFrame containing the time range to plot.
    
    Returns:
    - add_plots (list): List of addplot objects for pivot points.
    """
    pivot_points = ['pivot_point', 'r1', 's1', 'r2', 's2', 'r3', 's3']
    colors = ['black', 'red', 'green', 'red', 'green', 'red', 'green']
    
    add_plots = []
    for feature, color in zip(pivot_points, colors):
        add_plots.append(mpf.make_addplot(df_timeframe[feature], color=color, linestyle='--', width=0.75, panel=0))
    
    return add_plots

def add_vwap_and_bands(df):
    """
    Adds VWAP and VWAP bands to the plot.

    Parameters:
    - df (pd.DataFrame): The DataFrame containing the trading data.

    Returns:
    - List of additional plots to be added to the main plot.
    """
    
    # Prepare the additional plots
    add_plots = [
        mpf.make_addplot(df['vwap'], color='teal', linestyle='--', width=0.75, panel=0, secondary_y=False),
        mpf.make_addplot(df['vwap_band_1_up'], color='orange', linestyle='--', width=0.75, panel=0, secondary_y=False),
        mpf.make_addplot(df['vwap_band_1_down'], color='orange', linestyle='--', width=0.75, panel=0, secondary_y=False),
        mpf.make_addplot(df['vwap_band_2_up'], color='orange', linestyle='--', width=0.75, panel=0, secondary_y=False),
        mpf.make_addplot(df['vwap_band_2_down'], color='orange', linestyle='--', width=0.75, panel=0, secondary_y=False),
        mpf.make_addplot(df['vwap_band_3_up'], color='orange', linestyle='--', width=0.75, panel=0, secondary_y=False),
        mpf.make_addplot(df['vwap_band_3_down'], color='orange', linestyle='--', width=0.75, panel=0, secondary_y=False),
    ]
    
    return add_plots

# def add_signal_markers(df, signal_column):
#     """
#     Adds columns for buy and sell signal markers in the DataFrame.

#     Parameters:
#     - df (pd.DataFrame): The DataFrame containing the trading data.
#     - signal_column (str): The name of the column containing the trading signals.
#     """
#     df['buy_marker'] = None
#     df['sell_marker'] = None

#     previous_signal = df[signal_column].shift(1)
#     signal_changes = df[signal_column] != previous_signal

#     for idx, signal in df[signal_changes].iterrows():
#         if signal[signal_column] == 1:
#             df.at[idx, 'buy_marker'] = signal['Close']
#         elif signal[signal_column] == -1:
#             df.at[idx, 'sell_marker'] = signal['Close']

#     return df

def add_signal_markers(df, signal_column):
    """
    Adds columns for buy, sell, and market neutral signal markers in the DataFrame.

    Parameters:
    - df (pd.DataFrame): The DataFrame containing the trading data.
    - signal_column (str): The name of the column containing the trading signals.
    """
    df['buy_marker'] = None
    df['sell_marker'] = None
    df['neutral_marker'] = None  # Add a column for market neutral markers

    previous_signal = df[signal_column].shift(1)
    signal_changes = df[signal_column] != previous_signal

    for idx, signal in df[signal_changes].iterrows():
        if signal[signal_column] == 1:
            df.at[idx, 'buy_marker'] = signal['Close']
        elif signal[signal_column] == -1:
            df.at[idx, 'sell_marker'] = signal['Close']
        elif signal[signal_column] == 0:
            df.at[idx, 'neutral_marker'] = signal['Close']

    return df

def add_indicator_plot(df, signal_column):
    """
    Prepares indicator columns for plotting on the main plot, panel 2, and panel 3, including black dashed zero lines.

    Parameters:
    - df (pd.DataFrame): The DataFrame containing the trading data.
    - signal_column (str): The name of the column containing the trading signals.

    Returns:
    - list of addplot objects for the indicators and the zero lines.
    """
    add_plots = []

    percentile_viz = re.sub(r'.*(atr_percentile_\d+)', r'\1', signal_column)

    if '_crossrev' in signal_column:
        # Add ATR to panel 2
        if 'atr' in df.columns:
            atr_plot = mpf.make_addplot(df['atr'], panel=2, color='red', type='scatter', marker='o', label='ATR')
            atr_plot_2 = mpf.make_addplot(df['atr'], panel=2, color='red', linestyle='-', width=0.75)
            atr_threshold = mpf.make_addplot(df[percentile_viz], panel=2, color='black', linestyle=':', secondary_y=False)
            add_plots.extend([atr_plot, atr_plot_2, atr_threshold])

        # Add RSI to panel 3
        rsi_columns = ['RSI_14', 'RSI_12', 'RSI_16']
        df['zero_line'] = 0  # Ensure zero_line is in the DataFrame
        for rsi_col in rsi_columns:
            if rsi_col in df.columns:
                rsi_plot = mpf.make_addplot(df[rsi_col], panel=3, linestyle='-', width=0.75, label=f'{rsi_col}')
                add_plots.append(rsi_plot)
        # Add zero line to panel 3
        zero_line_rsi = mpf.make_addplot(df['zero_line'], panel=3, color='black', linestyle=':', secondary_y=False)
        add_plots.append(zero_line_rsi)

        # Add moving averages to main panel 0
        # moving_averages = signal_column.replace('signal_', '').replace('_crossrev_atr_percentile_95', '').split('_X_')
        moving_averages = re.sub(r'signal_|_crossrev_.*$', '', signal_column).split('_X_')
        for ma in moving_averages:
            if ma in df.columns:
                ma_plot = mpf.make_addplot(df[ma], panel=0, type='scatter', marker='o', markersize=10, label=f'{ma}')
                add_plots.append(ma_plot)
                
    else:
        # Determine the indicator columns
        base_indicator_column = signal_column.replace('signal_', '').replace('_derivative_2', '').replace('_derivative_1', '').replace('_shifted', '').replace('_backward', '').replace('_true', '')
        indicator_column = signal_column.replace('signal_', '')
        derivative_1_column = indicator_column.replace('_derivative_2', '_derivative_1')
        derivative_2_column = indicator_column.replace('_derivative_1', '_derivative_2')

        # Check if columns exist in the DataFrame
        missing_columns = []
        if base_indicator_column not in df.columns:
            missing_columns.append(base_indicator_column)
        if indicator_column not in df.columns:
            missing_columns.append(indicator_column)
        if derivative_1_column not in df.columns:
            missing_columns.append(derivative_1_column)
        if derivative_2_column not in df.columns:
            missing_columns.append(derivative_2_column)

        if missing_columns:
            raise ValueError(f"Indicator columns '{', '.join(missing_columns)}' not found in the DataFrame.")

        df['zero_line'] = 0

        # Add base indicator to the main plot
        if base_indicator_column in df.columns:
            base_indicator_plot = mpf.make_addplot(df[base_indicator_column], panel=0, color='purple', type='scatter', marker='o', markersize=10, label=f'{base_indicator_column}')
            base_indicator_plot_2 = mpf.make_addplot(df[base_indicator_column], panel=0, color='purple', linestyle='-', width=0.5)
            add_plots.extend([base_indicator_plot, base_indicator_plot_2])

        # Add first derivative to panel 2
        if derivative_1_column in df.columns:
            derivative_1_plot = mpf.make_addplot(df[derivative_1_column], panel=2, color='blue', type='scatter', marker='o', label=f'{derivative_1_column}')
            derivative_1_plot_2 = mpf.make_addplot(df[derivative_1_column], panel=2, color='blue', linestyle='-', width=0.75)
            zero_line_2 = mpf.make_addplot(df['zero_line'], panel=2, color='black', linestyle=':', secondary_y=False)
            add_plots.extend([derivative_1_plot, derivative_1_plot_2, zero_line_2])

        # Add second derivative to panel 3
        if derivative_2_column in df.columns:
            derivative_2_plot = mpf.make_addplot(df[derivative_2_column], panel=3, color='red', type='scatter', marker='o', label=f'{derivative_2_column}')
            derivative_2_plot_2 = mpf.make_addplot(df[derivative_2_column], panel=3, color='red', linestyle='-', width=0.75)
            zero_line_3 = mpf.make_addplot(df['zero_line'], panel=3, color='black', linestyle=':', secondary_y=False)
            add_plots.extend([derivative_2_plot, derivative_2_plot_2, zero_line_3])

    return add_plots

# def plot_for_time_range(df, start_day, end_day, start_time='09:30', end_time='16:00', include_pivot_points=False, include_vwap=False, signal_column=None, include_indicator=False):
#     """
#     Plots an OHLC candlestick chart for a given time range with price and volume.
#     Optionally includes pivot points, VWAP bands, trading signals, and an indicator.

#     Parameters:
#     - df (pd.DataFrame): The DataFrame containing the trading data.
#     - start_day (str): The start day for the visualization in 'YYYY-MM-DD' format.
#     - end_day (str): The end day for the visualization in 'YYYY-MM-DD' format.
#     - start_time (str): The start time for the time range visualization.
#     - end_time (str): The end time for the time range visualization.
#     - include_pivot_points (bool): Whether to include pivot points in the plot.
#     - include_vwap (bool): Whether to include VWAP and VWAP bands in the plot.
#     - signal_column (str): The name of the column containing the trading signals.
#     - include_indicator (bool): Whether to include the indicator in panel 2.
#     """
    
#     df.rename(columns={'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'volume': 'Volume'}, inplace=True)
    
#     # Combine start day and time, and end day and time into datetime strings
#     start_datetime = f"{start_day} {start_time}"
#     end_datetime = f"{end_day} {end_time}"
    
#     df_timeframe = df[start_datetime:end_datetime].copy()

#     add_plots = []

#     if include_pivot_points:
#         add_plots.extend(add_pivot_points(df_timeframe))

#     if include_vwap:
#         add_plots.extend(add_vwap_and_bands(df_timeframe))
    
#     # Add trading signals
#     if signal_column:
#         df_timeframe = add_signal_markers(df_timeframe, signal_column)
#         add_plots.append(mpf.make_addplot(df_timeframe['buy_marker'], type='scatter', marker='^', markersize=100, color='green', panel=0, secondary_y=False))
#         add_plots.append(mpf.make_addplot(df_timeframe['sell_marker'], type='scatter', marker='v', markersize=100, color='red', panel=0, secondary_y=False))
    
#     # Add indicator plot in panel 2
#     if include_indicator and signal_column:
#         try:
#             indicator_plots = add_indicator_plot(df_timeframe, signal_column)
#             add_plots.extend(indicator_plots)
#         except ValueError as e:
#             print(e)

#     # Calculate accumulative profit and add to the main plot with a secondary y-axis
#     profit_column = signal_column.replace('signal_', 'profit_loss_')
#     if profit_column in df.columns:
#         df_timeframe['cumulative_profit'] = df_timeframe[profit_column].cumsum()
#         add_plots.append(mpf.make_addplot(df_timeframe['cumulative_profit'], panel=0, color='black', linestyle='--', secondary_y=True, label='Cumulative Profit'))

#     mpf.plot(df_timeframe, type='candle', style='charles',
#              title=f"{signal_column} : {start_day} {start_time} to {end_day} {end_time}",
#              addplot=add_plots,
#              figratio=(36, 18),
#              figscale=2.5,
#              volume=True,
#              volume_panel=1,
#              panel_ratios=(8, 2, 2, 2),  # Adjust panel_ratios to accommodate the additional panel
#              show_nontrading=False,
#              tight_layout=True)
    
def plot_for_time_range(df, start_day, end_day, start_time='09:30', end_time='16:00', include_pivot_points=False, include_vwap=False, signal_column=None, include_indicator=False):
    """
    Plots an OHLC candlestick chart for a given time range with price and volume.
    Optionally includes pivot points, VWAP bands, trading signals, and an indicator.

    Parameters:
    - df (pd.DataFrame): The DataFrame containing the trading data.
    - start_day (str): The start day for the visualization in 'YYYY-MM-DD' format.
    - end_day (str): The end day for the visualization in 'YYYY-MM-DD' format.
    - start_time (str): The start time for the time range visualization.
    - end_time (str): The end time for the time range visualization.
    - include_pivot_points (bool): Whether to include pivot points in the plot.
    - include_vwap (bool): Whether to include VWAP and VWAP bands in the plot.
    - signal_column (str): The name of the column containing the trading signals.
    - include_indicator (bool): Whether to include the indicator in panel 2.
    """
    
    df.rename(columns={'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'volume': 'Volume'}, inplace=True)
    
    # Combine start day and time, and end day and time into datetime strings
    start_datetime = f"{start_day} {start_time}"
    end_datetime = f"{end_day} {end_time}"
    
    df_timeframe = df[start_datetime:end_datetime].copy()

    add_plots = []

    if include_pivot_points:
        add_plots.extend(add_pivot_points(df_timeframe))

    if include_vwap:
        add_plots.extend(add_vwap_and_bands(df_timeframe))
    
    # Add trading signals
    if signal_column:
        df_timeframe = add_signal_markers(df_timeframe, signal_column)
        add_plots.append(mpf.make_addplot(df_timeframe['buy_marker'], type='scatter', marker='^', markersize=100, color='green', panel=0, secondary_y=False))
        add_plots.append(mpf.make_addplot(df_timeframe['sell_marker'], type='scatter', marker='v', markersize=100, color='red', panel=0, secondary_y=False))
        add_plots.append(mpf.make_addplot(df_timeframe['neutral_marker'], type='scatter', marker='o', markersize=100, color='black', panel=0, secondary_y=False))  # Add neutral markers
    
    # Add indicator plot in panel 2
    if include_indicator and signal_column:
        try:
            indicator_plots = add_indicator_plot(df_timeframe, signal_column)
            add_plots.extend(indicator_plots)
        except ValueError as e:
            print(e)

    # Calculate accumulative profit and add to the main plot with a secondary y-axis
    profit_column = signal_column.replace('signal_', 'profit_loss_')
    if profit_column in df.columns:
        df_timeframe['cumulative_profit'] = df_timeframe[profit_column].cumsum()
        add_plots.append(mpf.make_addplot(df_timeframe['cumulative_profit'], panel=0, color='black', linestyle='--', secondary_y=True, label='Cumulative Profit'))

    # Add ADX, +DI, and -DI to new panels at the bottom
    add_plots.append(mpf.make_addplot(df_timeframe['adx'], panel=4, color='blue', title='adx'))
    add_plots.append(mpf.make_addplot(df_timeframe['+di'], panel=5, color='green', title='+di'))
    add_plots.append(mpf.make_addplot(df_timeframe['-di'], panel=6, color='red', title='-di'))

    mpf.plot(df_timeframe, type='candle', style='charles',
             title=f"{signal_column} : {start_day} {start_time} to {end_day} {end_time}",
             addplot=add_plots,
             figratio=(36, 18),
             figscale=2.5,
             volume=True,
             volume_panel=1,
             panel_ratios=(8, 2, 2, 2),  # Adjust panel_ratios to accommodate the additional panel
             show_nontrading=False,
             tight_layout=True)

# Old functions not currently in use

# def transaction_counts(df):
#     """
#     This function calculates the difference between consecutive rows for each column in the DataFrame
#     that starts with 'signal_' and adds these differences as new columns prefixed with 'diff_'.

#     Parameters:
#     df (pd.DataFrame): The input DataFrame containing signal columns.

#     Returns:
#     pd.DataFrame: The DataFrame with additional columns representing the differences of signal columns.
#     """
#     signal_columns = [col for col in df.columns if col.startswith('signal_')]

#     for signal_column in signal_columns:
#         # Calculate the difference between consecutive rows in the signal column
#         df[f'diff_{signal_column}'] = df[signal_column].diff()

#     return df

# The inefficient RSI trigger function

# def rsi_trigger_based_trading(df, upper_trigger=75, lower_trigger=25, suffix=""):
#     """
#     Generates trading signals based on multiple RSI columns with state management for entering and exiting trades.
#     - Arm short position when RSI > upper_trigger and enter short when RSI < upper_trigger.
#     - Close short position when RSI < lower_trigger or RSI > upper_trigger before reaching lower_trigger.
#     - Arm long position when RSI < lower_trigger and enter long when RSI > lower_trigger.
#     - Close long position when RSI > upper_trigger or RSI < lower_trigger before reaching upper_trigger.

#     Parameters:
#     - df (pd.DataFrame): The DataFrame containing 'RSI_' columns.
#     - upper_trigger (int): Upper trigger threshold for RSI.
#     - lower_trigger (int): Lower trigger threshold for RSI.
#     - suffix (str): Suffix to add to the signal column names for distinguishing different trigger sets.

#     Returns:
#     - pd.DataFrame: The DataFrame with additional 'signal_' columns indicating the current position for each RSI column.
#     """
#     columns_rsi = [col for col in df.columns if col.startswith('RSI_')]
    
#     for rsi_col in columns_rsi:
#         signal_col = f'signal_{rsi_col}_{suffix}'
#         df[signal_col] = 0

#         # Initialize states
#         armed_short = False
#         armed_long = False
#         position = 0  # 0 = neutral, 1 = long, -1 = short

#         # Iterate through the DataFrame
#         for i in range(1, len(df)):
#             if armed_short:
#                 if df[rsi_col].iloc[i] < upper_trigger:
#                     position = -1
#                     armed_short = False
#             elif armed_long:
#                 if df[rsi_col].iloc[i] > lower_trigger:
#                     position = 1
#                     armed_long = False
#             elif position == -1:
#                 if df[rsi_col].iloc[i] < lower_trigger:
#                     position = 0
#                 elif df[rsi_col].iloc[i] > upper_trigger:
#                     position = 0
#             elif position == 1:
#                 if df[rsi_col].iloc[i] > upper_trigger:
#                     position = 0
#                 elif df[rsi_col].iloc[i] < lower_trigger:
#                     position = 0
                    
#             # else: # This piece disables frequent rearming when RSI oscillates around trigger levels
#             #     if df[rsi_col].iloc[i] > upper_trigger:
#             #         armed_short = True
#             #     elif df[rsi_col].iloc[i] < lower_trigger:
#             #         armed_long = True

#                 # Allow re-arming after disarming
#             if position == 0: # Comment this part out and uncomment the chunk above to reduce response to noise around triggers
#                 if df[rsi_col].iloc[i] > upper_trigger:
#                     armed_short = True
#                 elif df[rsi_col].iloc[i] < lower_trigger:
#                     armed_long = True
            
#             df.at[i, signal_col] = position

#     return df

# The inefficient Position Profit function

# def calculate_position_percent_changes(df, price_col='close'):
#     """
#     Calculate the percentage change in close price from the beginning of each held position to the end
#     for all columns starting with 'signal_'.

#     Parameters:
#     df (pd.DataFrame): DataFrame containing the signal and close price columns.
#     price_col (str): Name of the close price column (default 'close').

#     Returns:
#     pd.DataFrame: DataFrame with additional columns for percentage change.
#     """
#     # Iterate through all columns that start with 'signal_'
#     signal_cols = [col for col in df.columns if col.startswith('signal_')]
    
#     for signal_col in signal_cols:
#         # Initialize new column with zeros
#         change_col = signal_col.replace('signal_', 'position_percent_change_')
#         df[change_col] = 0.0

#         # Find the indices where the signal changes
#         signal_changes = df[signal_col].ne(df[signal_col].shift()).cumsum()

#         # Group by these changes and process each group
#         for _, group in df.groupby(signal_changes):
#             start_price = group[price_col].iloc[0]
#             end_index = df.index.get_loc(group.index[-1]) + 1
            
#             if end_index < len(df):
#                 end_price = df.iloc[end_index][price_col]
#                 pct_change = (end_price - start_price) / start_price
#                 df.loc[group.index[-1], change_col] = pct_change
#             else:
#                 end_price = group[price_col].iloc[-1]
#                 pct_change = (end_price - start_price) / start_price
#                 df.loc[group.index[-1], change_col] = pct_change
            
#     return df



# atr_thresholds = ['atr_percentile_1', 'atr_percentile_5', 'atr_percentile_10', 'atr_percentile_25', 
#                   'atr_percentile_50', 'atr_percentile_75', 'atr_percentile_90', 'atr_percentile_95', 
#                   'atr_percentile_99']

# # Iterate over short moving average periods
# for short_period in range(1, 11):
#     short_ma = f'wma_{short_period}'
    
#     # Iterate over long moving average periods
#     for long_period in range(20, 51, 2):
#         long_ma = f'wma_{long_period}'
        
#         # Iterate over each ATR threshold column
#         for atr_threshold in atr_thresholds:
#             df = hybrid_strategy(df, short_ma, long_ma, atr_threshold)

# # for ticker in df_dict:
# #     df_dict[ticker] = hybrid_strategy(df_dict[ticker], 'wma_3', 'wma_20', atr_threshold=atr_threshold)

# # Iterate over each ticker's DataFrame in df_dict
# for ticker in df_dict:
#     # df = df_dict[ticker]
    
#     # Iterate over short moving average periods (1 to 10)
#     for short_period in range(1, 11):
#         short_ma = f'wma_{short_period}'
        
#         # Iterate over long moving average periods (20 to 50 by 2)
#         for long_period in range(20, 51, 2):
#             long_ma = f'wma_{long_period}'
            
#             # Iterate over each ATR threshold column
#             for atr_threshold in atr_thresholds:
#                 df_dict[ticker] = hybrid_strategy(df_dict[ticker], short_ma, long_ma, atr_threshold)
    
#     # Update the DataFrame back to the dictionary
#     # df_dict[ticker] = df



# def get_vwap_bands_by_date_range(df, start_date, end_date):
#     """
#     Returns VWAP and VWAP bands within a specified date range.

#     Parameters:
#     - df (pd.DataFrame): The DataFrame containing the trading data.
#     - start_date (str): The start date in 'YYYY-MM-DD' format.
#     - end_date (str): The end date in 'YYYY-MM-DD' format.

#     Returns:
#     - pd.DataFrame: The filtered DataFrame with VWAP and VWAP bands.
#     """
    
#     # Ensure the DataFrame's index is in datetime format
#     df.index = pd.to_datetime(df.index)
    
#     # Filter the DataFrame by the date range
#     df_filtered = df.loc[start_date:end_date, ['vwap_band_3_up', 'vwap_band_2_up', 'vwap_band_1_up', 'vwap', 'vwap_band_1_down', 'vwap_band_2_down', 'vwap_band_3_down']]
    
#     return df_filtered

# # Example usage:
# # Assuming df_viz is your DataFrame
# start_date = '2020-02-12'
# end_date = '2020-02-12'
# filtered_df = get_vwap_bands_by_date_range(df_viz, start_date, end_date)
# print(filtered_df)