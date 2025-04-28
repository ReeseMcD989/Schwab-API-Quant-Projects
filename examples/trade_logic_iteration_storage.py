##############################################################################################################################################
# Vectorized without iterrows- Primary candidate for live trade logic- Obeys stop losses, 
# update_position_open and on needs to be dynamically calculated according to window size of indicators
# This logic is to be converted to live buy/sell logic

def generate_trading_signals(candles, ma_name1='wma_5', ma_name2='sma_5', rsi_column='rsi_5'):
    """ Vectorized without iterrows
    Generate buy/sell signals based on moving averages and RSI indicators and save position states.

    Parameters:
    - candles (pd.DataFrame): The DataFrame containing candle data.
    - ma_name1 (str): Column name for the first moving average.
    - ma_name2 (str): Column name for the second moving average.
    - rsi_column (str): Column name for the RSI indicator.

    Returns:
    - candles (pd.DataFrame): The DataFrame with updated signal and position state columns.
    """
    # Dynamically generate signal column names
    ma_signal_column = f'signal_{ma_name1}_{ma_name2}'
    rsi_signal_column = f'signal_{rsi_column}'
    ma_position_open_col = f'position_open_{ma_name1}_{ma_name2}'
    rsi_position_open_col = f'position_open_{rsi_column}'

    # Initialize signal and position state columns
    candles[ma_signal_column] = 0
    candles[rsi_signal_column] = 0
    candles[ma_position_open_col] = False
    candles[rsi_position_open_col] = False

    # Moving average signal and position state generation
    ma_position_open = False  # Local variable for tracking state
    ma_signals = []
    ma_positions = []
    for ma1, ma2 in zip(candles[ma_name1], candles[ma_name2]):
        if not ma_position_open and ma1 <= ma2:
            ma_position_open = True
            ma_signals.append(1)
        elif ma_position_open and ma1 > ma2:
            ma_position_open = False
            ma_signals.append(0)
        else:
            ma_signals.append(ma_signals[-1] if ma_signals else 0)
        ma_positions.append(ma_position_open)
    candles[ma_signal_column] = ma_signals
    candles[ma_position_open_col] = ma_positions

    # RSI signal and position state generation
    rsi_position_open = False  # Local variable for tracking state
    rsi_signals = []
    rsi_positions = []
    for rsi in candles[rsi_column]:
        if not rsi_position_open and rsi < 50:
            rsi_position_open = True
            rsi_signals.append(1)
        elif rsi_position_open and rsi >= 50:
            rsi_position_open = False
            rsi_signals.append(0)
        else:
            rsi_signals.append(rsi_signals[-1] if rsi_signals else 0)
        rsi_positions.append(rsi_position_open)
    candles[rsi_signal_column] = rsi_signals
    candles[rsi_position_open_col] = rsi_positions

    return candles

def update_position_open(candles, ma_name1='wma_5', ma_name2='sma_5', rsi_column='rsi_5'):
    """
    Update the 'ma_position_open' and 'rsi_position_open' columns for a given DataFrame.

    Parameters:
    - candles (pd.DataFrame): The DataFrame containing the signals and position columns.
    - ma_name1 (str): Column name for the first moving average.
    - ma_name2 (str): Column name for the second moving average.
    - rsi_column (str): Column name for the RSI signal.

    Returns:
    - pd.DataFrame: The updated DataFrame with 'ma_position_open' and 'rsi_position_open' columns.
    """
    # Dynamically generate signal column names
    ma_signal_column = f'signal_{ma_name1}_{ma_name2}'
    rsi_signal_column = f'signal_{rsi_column}'
    ma_position_open = f'position_open_{ma_name1}_{ma_name2}'
    rsi_position_open = f'position_open_{rsi_column}'
    
    # Update position open columns
    candles[ma_position_open] = candles[ma_signal_column] == 1
    candles[rsi_position_open] = candles[rsi_signal_column] == 1
    
    return candles

def determine_entry_prices(candles, ma_name1='wma_5', ma_name2='sma_5', rsi_column='rsi_5', ticker_to_tick_size=None, ticker=None):
    """
    Determine entry prices for MA and RSI strategies based on signals.

    Parameters:
    - candles (pd.DataFrame): The DataFrame containing candle data.
    - ma_name1 (str): Column name for the first moving average.
    - ma_name2 (str): Column name for the second moving average.
    - rsi_column (str): Column name for the RSI indicator.
    - ticker_to_tick_size (dict): Mapping of tickers to their tick sizes.
    - ticker (str): The ticker for which the tick size applies.

    Returns:
    - candles (pd.DataFrame): The DataFrame with updated entry price columns.
    """
    # Dynamically generate column names
    ma_signal_column = f'signal_{ma_name1}_{ma_name2}'
    rsi_signal_column = f'signal_{rsi_column}'
    ma_entry_price = f'entry_price_{ma_name1}_{ma_name2}'
    rsi_entry_price = f'entry_price_{rsi_column}'

    # Initialize entry price columns
    candles[ma_entry_price] = None
    candles[rsi_entry_price] = None

    # Get tick size
    tick_size = ticker_to_tick_size.get(ticker, 0) if ticker_to_tick_size else 0

    # Moving Average Strategy
    ma_signals = candles[ma_signal_column]
    ma_close_prices = candles['close']
    ma_entry_mask = (ma_signals == 1) & (ma_signals.shift(1) != 1)
    candles.loc[ma_entry_mask, ma_entry_price] = ma_close_prices[ma_entry_mask] + tick_size

    # RSI Strategy
    rsi_signals = candles[rsi_signal_column]
    rsi_entry_mask = (rsi_signals == 1) & (rsi_signals.shift(1) != 1)
    candles.loc[rsi_entry_mask, rsi_entry_price] = ma_close_prices[rsi_entry_mask] + tick_size

    return candles

def determine_exit_prices(candles, ma_name1='wma_5', ma_name2='sma_5', rsi_column='rsi_5', ticker_to_tick_size=None, ticker=None):
    """
    Determine exit prices for MA and RSI strategies based on signals.

    Parameters:
    - candles (pd.DataFrame): The DataFrame containing candle data.
    - ma_name1 (str): Column name for the first moving average.
    - ma_name2 (str): Column name for the second moving average.
    - rsi_column (str): Column name for the RSI indicator.
    - ticker_to_tick_size (dict): Mapping of tickers to their tick sizes.
    - ticker (str): The ticker for which the tick size applies.

    Returns:
    - candles (pd.DataFrame): The DataFrame with updated exit price columns.
    """
    # Dynamically generate column names
    ma_signal_column = f'signal_{ma_name1}_{ma_name2}'
    rsi_signal_column = f'signal_{rsi_column}'
    ma_exit_price = f'exit_price_{ma_name1}_{ma_name2}'
    rsi_exit_price = f'exit_price_{rsi_column}'

    # Initialize exit price columns
    candles[ma_exit_price] = None
    candles[rsi_exit_price] = None

    # Get tick size
    tick_size = ticker_to_tick_size.get(ticker, 0) if ticker_to_tick_size else 0

    # Moving Average Strategy
    ma_signals = candles[ma_signal_column]
    ma_close_prices = candles['close']
    ma_exit_mask = (ma_signals == 0) & (ma_signals.shift(1) == 1)
    candles.loc[ma_exit_mask, ma_exit_price] = ma_close_prices[ma_exit_mask] - tick_size

    # RSI Strategy
    rsi_signals = candles[rsi_signal_column]
    rsi_exit_mask = (rsi_signals == 0) & (rsi_signals.shift(1) == 1)
    candles.loc[rsi_exit_mask, rsi_exit_price] = ma_close_prices[rsi_exit_mask] - tick_size

    return candles

def calculate_stop_losses(candles, ma_name1='wma_5', ma_name2='sma_5', rsi_column='rsi_5'):
    """
    Dynamically calculate stop loss levels for MA and RSI strategies and ensure they persist while positions are open.

    Parameters:
    - candles (pd.DataFrame): The DataFrame containing candle data.
    - ma_name1 (str): Column name for the first moving average.
    - ma_name2 (str): Column name for the second moving average.
    - rsi_column (str): Column name for the RSI indicator.

    Returns:
    - pd.DataFrame: The updated DataFrame with dynamically named stop loss columns.
    """
    # Dynamically generate column names
    ma_entry_price = f'entry_price_{ma_name1}_{ma_name2}'
    ma_exit_price = f'exit_price_{ma_name1}_{ma_name2}'
    rsi_entry_price = f'entry_price_{rsi_column}'
    rsi_exit_price = f'exit_price_{rsi_column}'
    stop_loss_ma = f'stop_loss_{ma_name1}_{ma_name2}'
    stop_loss_rsi = f'stop_loss_{rsi_column}'

    # Initialize stop loss columns
    candles[stop_loss_ma] = None
    candles[stop_loss_rsi] = None

    # Moving Average Stop Loss
    ma_entry_mask = candles[ma_entry_price].notnull()
    ma_exit_mask = candles[ma_exit_price].notnull()
    
    # Set stop loss where positions open
    candles.loc[ma_entry_mask, stop_loss_ma] = candles[ma_entry_price] - candles['candle_span_max']

    # Reset stop loss and close position where positions close
    candles.loc[ma_exit_mask, stop_loss_ma] = None

    # RSI Stop Loss
    rsi_entry_mask = candles[rsi_entry_price].notnull()
    rsi_exit_mask = candles[rsi_exit_price].notnull()
    
    # Set stop loss where positions open
    candles.loc[rsi_entry_mask, stop_loss_rsi] = candles[rsi_entry_price] - candles['candle_span_max']

    # Reset stop loss and close position where positions close
    candles.loc[rsi_exit_mask, stop_loss_rsi] = None

    # Forward-fill stop loss for both strategies
    candles[stop_loss_ma] = candles[stop_loss_ma].ffill()
    candles[stop_loss_rsi] = candles[stop_loss_rsi].ffill()

    return candles

def track_stop_loss_hits(candles, ma_name1='wma_5', ma_name2='sma_5', rsi_column='rsi_5', ticker_to_tick_size=None, ticker=None): # Tick size is not being used, address this here and in the function calling
    """
    Track whether stop losses have been hit for MA and RSI strategies and update dynamically named columns.

    Parameters:
    - candles (pd.DataFrame): The DataFrame containing candle data.
    - ma_name1 (str): Column name for the first moving average.
    - ma_name2 (str): Column name for the second moving average.
    - rsi_column (str): Column name for the RSI indicator.
    - ticker_to_tick_size (dict): Mapping of tickers to their tick sizes.
    - ticker (str): The ticker for which the tick size applies.

    Returns:
    - pd.DataFrame: The updated DataFrame with dynamically named stop loss hit flags.
    """
    # Dynamically generate column names
    stop_loss_ma = f'stop_loss_{ma_name1}_{ma_name2}'
    stop_loss_rsi = f'stop_loss_{rsi_column}'
    ma_stop_loss_hit = f'stop_loss_hit_{ma_name1}_{ma_name2}'
    rsi_stop_loss_hit = f'stop_loss_hit_{rsi_column}'
    ma_position_open = f'position_open_{ma_name1}_{ma_name2}'
    rsi_position_open = f'position_open_{rsi_column}'
    
    # Initialize stop loss hit columns
    candles[ma_stop_loss_hit] = False
    candles[rsi_stop_loss_hit] = False

    # Get tick size
    tick_size = ticker_to_tick_size.get(ticker, 0) if ticker_to_tick_size else 0

    # Ensure stop loss values are numerical (convert None to NaN)
    candles[stop_loss_ma] = candles[stop_loss_ma].fillna(float('inf'))
    candles[stop_loss_rsi] = candles[stop_loss_rsi].fillna(float('inf'))

    # Moving Average Stop Loss Hit Logic
    ma_hit_condition = (candles[stop_loss_ma].notnull()) & (candles['close'] <= candles[stop_loss_ma]) & candles[ma_position_open]
    candles.loc[ma_hit_condition, ma_stop_loss_hit] = True

    # RSI Stop Loss Hit Logic
    rsi_hit_condition = (candles[stop_loss_rsi].notnull()) & (candles['close'] <= candles[stop_loss_rsi]) & candles[rsi_position_open]
    candles.loc[rsi_hit_condition, rsi_stop_loss_hit] = True

    return candles

def adjust_signals_for_stop_loss(candles, ma_name1='wma_5', ma_name2='sma_5', rsi_column='rsi_5'):
    """
    Adjust MA and RSI signals to 0 where stop loss has been hit.

    Parameters:
    - candles (pd.DataFrame): The DataFrame containing candle data.
    - ma_name1 (str): Column name for the first moving average.
    - ma_name2 (str): Column name for the second moving average.
    - rsi_column (str): Column name for the RSI indicator.

    Returns:
    - pd.DataFrame: The updated DataFrame with adjusted signals.
    """
    # Dynamically generate column names
    ma_signal_column = f'signal_{ma_name1}_{ma_name2}'
    rsi_signal_column = f'signal_{rsi_column}'
    ma_stop_loss_hit_column = f'stop_loss_hit_{ma_name1}_{ma_name2}'
    rsi_stop_loss_hit_column = f'stop_loss_hit_{rsi_column}'

    # Adjust MA and RSI signals where stop loss has been hit
    candles.loc[candles[ma_stop_loss_hit_column], ma_signal_column] = 0
    candles.loc[candles[rsi_stop_loss_hit_column], rsi_signal_column] = 0

    return candles

def update_stop_loss(candles, ma_name1='wma_5', ma_name2='sma_5', rsi_column='rsi_5'):
    """
    Dynamically set stop loss columns to NaN where corresponding signal columns are 0.

    Parameters:
    - candles (pd.DataFrame): The DataFrame containing candle data.
    - ma_name1 (str): Column name for the first moving average.
    - ma_name2 (str): Column name for the second moving average.
    - rsi_column (str): Column name for the RSI indicator.

    Returns:
    - pd.DataFrame: The updated DataFrame with dynamically named stop loss columns modified.
    """
    # Dynamically generate column names
    ma_signal_column = f'signal_{ma_name1}_{ma_name2}'
    rsi_signal_column = f'signal_{rsi_column}'
    stop_loss_ma = f'stop_loss_{ma_name1}_{ma_name2}'
    stop_loss_rsi = f'stop_loss_{rsi_column}'

    # Update stop loss columns to NaN where signals are 0
    candles.loc[candles[ma_signal_column] == 0, stop_loss_ma] = float('nan')
    candles.loc[candles[rsi_signal_column] == 0, stop_loss_rsi] = float('nan')

    return candles

def calculate_profit_loss_1(candles, multiplier=1, ma_name1='wma_5', ma_name2='sma_5', rsi_column='rsi_5'):
    """ 
    Dynamically calculate profit and loss based on entry and exit price columns.

    Parameters:
    - candles (pd.DataFrame): The DataFrame containing candle data with entry and exit price columns.
    - multiplier (float): The multiplier for PnL calculation (e.g., contract size).
    - ma_name1 (str): Column name for the first moving average.
    - ma_name2 (str): Column name for the second moving average.
    - rsi_column (str): Column name for the RSI indicator.

    Returns:
    - pd.DataFrame: The DataFrame with dynamically named profit/loss columns.
    """
    # Dynamically generate column names
    ma_entry_price = f'entry_price_{ma_name1}_{ma_name2}'
    ma_exit_price = f'exit_price_{ma_name1}_{ma_name2}'
    rsi_entry_price = f'entry_price_{rsi_column}'
    rsi_exit_price = f'exit_price_{rsi_column}'
    pnl_ma_col = f'pnl_{ma_name1}_{ma_name2}'
    pnl_rsi_col = f'pnl_{rsi_column}'
    cum_pnl_ma_col = f'cum_{pnl_ma_col}'
    cum_pnl_rsi_col = f'cum_{pnl_rsi_col}'
    cum_pnl_all_col = f'cum_pnl_all_{ma_name1}_{ma_name2}_{rsi_column}'

    # Initialize PnL columns
    candles[pnl_ma_col] = 0.0
    candles[pnl_rsi_col] = 0.0

    # Moving Average Strategy PnL Calculation
    ma_entry_indices = candles.index[candles[ma_entry_price].notnull()]
    ma_exit_indices = candles.index[candles[ma_exit_price].notnull()]

    # Pair up entry and exit prices
    valid_pairs_ma = min(len(ma_entry_indices), len(ma_exit_indices))
    ma_entry_prices = candles.loc[ma_entry_indices[:valid_pairs_ma], ma_entry_price].values
    ma_exit_prices = candles.loc[ma_exit_indices[:valid_pairs_ma], ma_exit_price].values

    # Calculate PnL for MA strategy
    ma_pnl = (ma_exit_prices - ma_entry_prices) * multiplier
    candles.loc[ma_exit_indices[:valid_pairs_ma], pnl_ma_col] = ma_pnl

    # RSI Strategy PnL Calculation
    rsi_entry_indices = candles.index[candles[rsi_entry_price].notnull()]
    rsi_exit_indices = candles.index[candles[rsi_exit_price].notnull()]

    # Pair up entry and exit prices
    valid_pairs_rsi = min(len(rsi_entry_indices), len(rsi_exit_indices))
    rsi_entry_prices = candles.loc[rsi_entry_indices[:valid_pairs_rsi], rsi_entry_price].values
    rsi_exit_prices = candles.loc[rsi_exit_indices[:valid_pairs_rsi], rsi_exit_price].values

    # Calculate PnL for RSI strategy
    rsi_pnl = (rsi_exit_prices - rsi_entry_prices) * multiplier
    candles.loc[rsi_exit_indices[:valid_pairs_rsi], pnl_rsi_col] = rsi_pnl

    # Calculate cumulative PnL for both strategies
    candles[cum_pnl_ma_col] = candles[pnl_ma_col].cumsum()
    candles[cum_pnl_rsi_col] = candles[pnl_rsi_col].cumsum()

    # Calculate combined cumulative PnL
    candles[cum_pnl_all_col] = candles[cum_pnl_ma_col] + candles[cum_pnl_rsi_col]

    return candles

for ticker, df in minute_candles_1.items():
    for sig_ma, con_ma, rsi_col in ma_combinations:
        # Generate trading signals
        minute_candles_1[ticker] = generate_trading_signals(
            df,
            ma_name1=sig_ma,
            ma_name2=con_ma,
            rsi_column=rsi_col
        )

        # Update position_open columns to be 1:1 verbal boolean with the signal
        minute_candles_1[ticker] = update_position_open(
            df,
            ma_name1=sig_ma,
            ma_name2=con_ma,
            rsi_column=rsi_col
        )

        # Determine entry prices for each ticker
        minute_candles_1[ticker] = determine_entry_prices(
            df,
            ma_name1=sig_ma,
            ma_name2=con_ma,
            rsi_column=rsi_col,
            ticker_to_tick_size=ticker_to_tick_size,
            ticker=ticker
        )

        # Determine exit prices for each ticker
        minute_candles_1[ticker] = determine_exit_prices(
            df,
            ma_name1=sig_ma,
            ma_name2=con_ma,
            rsi_column=rsi_col,
            ticker_to_tick_size=ticker_to_tick_size,
            ticker=ticker
        )

        # Stop loss calculation
        minute_candles_1[ticker] = calculate_stop_losses(
            df,
            ma_name1=sig_ma,
            ma_name2=con_ma,
            rsi_column=rsi_col
        )

        # Track stop loss hits
        minute_candles_1[ticker] = track_stop_loss_hits(
            df,
            ma_name1=sig_ma,
            ma_name2=con_ma,
            rsi_column=rsi_col,
            ticker_to_tick_size=ticker_to_tick_size,
            ticker=ticker
        )

        # Adjust signals from stop loss hits
        minute_candles_1[ticker] = adjust_signals_for_stop_loss(
            df,
            ma_name1=sig_ma,
            ma_name2=con_ma,
            rsi_column=rsi_col
        )

        # Re-update position_open column after stop loss hits
        minute_candles_1[ticker] = update_position_open(
            df,
            ma_name1=sig_ma,
            ma_name2=con_ma,
            rsi_column=rsi_col
        )

        # Re-determine entry prices after stop loss hits
        minute_candles_1[ticker] = determine_entry_prices(
            df,
            ma_name1=sig_ma,
            ma_name2=con_ma,
            rsi_column=rsi_col,
            ticker_to_tick_size=ticker_to_tick_size,
            ticker=ticker
        )

        # Re-determine exit prices after stop loss hits
        minute_candles_1[ticker] = determine_exit_prices(
            df,
            ma_name1=sig_ma,
            ma_name2=con_ma,
            rsi_column=rsi_col,
            ticker_to_tick_size=ticker_to_tick_size,
            ticker=ticker
        )

        # Update stop loss levels after stop loss hits
        minute_candles_1[ticker] = update_stop_loss(
            df,
            ma_name1=sig_ma,
            ma_name2=con_ma,
            rsi_column=rsi_col
        )

        # Calculate profit/loss for each ticker's DataFrame
        minute_candles_1[ticker] = calculate_profit_loss_1(
            df,
            multiplier=1,
            ma_name1=sig_ma,
            ma_name2=con_ma,
            rsi_column=rsi_col
        )

def plot_moving_averages_1(candles, 
                           ma_name1='wma_5', 
                           ma_name2='sma_5', 
                           rsi_column='rsi_5',  
                           figsize=(40, 20), 
                           font_size=10, 
                           ma_markersize=50, 
                           signal_markersize_y=400, 
                           signal_markersize_b=250
                           ):
    """
    Plots the minute_candles DataFrame with two selected moving averages and optional RSI.
    Also plots cumulative profit for MA and RSI strategies on a secondary axis.

    Parameters:
    - ma_name1 (str): The column name of the first moving average to plot.
    - ma_name2 (str): The column name of the second moving average to plot.
    - signal_column (str): The column name of the signal data (default is 'signal').
    - figsize (tuple): The size of the plot (width, height) in inches (default is (30, 20)).
    """

    try:
        # Clean the data to ensure numeric columns are valid
        columns_to_convert = ['open', 'high', 'low', 'close', 'volume', ma_name1, ma_name2, rsi_column] 
        candles[columns_to_convert] = candles[columns_to_convert].apply(pd.to_numeric, errors='coerce')

        # Generate dynamic column names for PnL and signals
        ma_entry_price = f'entry_price_{ma_name1}_{ma_name2}'
        ma_exit_price = f'exit_price_{ma_name1}_{ma_name2}'
        rsi_entry_price = f'entry_price_{rsi_column}'
        rsi_exit_price = f'exit_price_{rsi_column}'
        cum_pnl_ma_col = f'cum_pnl_{ma_name1}_{ma_name2}'
        cum_pnl_rsi_col = f'cum_pnl_{rsi_column}'
        cum_pnl_all_col = f'cum_pnl_all_{ma_name1}_{ma_name2}_{rsi_column}'
        stop_loss_ma = f'stop_loss_{ma_name1}_{ma_name2}'
        stop_loss_rsi = f'stop_loss_{rsi_column}'

        # Select the columns to plot
        plot_data = candles[['datetime', 'open', 'high', 'low', 'close', 'volume', 
                             ma_name1, ma_name2, rsi_column, 
                             ma_entry_price, ma_exit_price, rsi_entry_price, rsi_exit_price,
                             cum_pnl_ma_col, cum_pnl_rsi_col, cum_pnl_all_col,
                             stop_loss_ma, stop_loss_rsi]].copy() # here
        plot_data.set_index('datetime', inplace=True)

        # Create the additional plots for the moving averages and RSI, but only if they are warmed up
        add_plots = []

        # Check if the moving averages have enough valid data to plot
        if not candles[ma_name1].isnull().all() and not candles[ma_name2].isnull().all():
            add_plots.append(mpf.make_addplot(plot_data[ma_name1], color='yellow', type='scatter', marker='o', markersize=ma_markersize, label=f'{ma_name1}'))
            add_plots.append(mpf.make_addplot(plot_data[ma_name1], color='yellow', linestyle='-', width=0.75))
            add_plots.append(mpf.make_addplot(plot_data[ma_name2], color='purple', type='scatter', marker='o', markersize=ma_markersize, label=f'{ma_name2}'))
            add_plots.append(mpf.make_addplot(plot_data[ma_name2], color='purple', linestyle='-', width=0.75))
        else:
            print("Moving averages have not warmed up yet. Plotting without them.")

        # Check if the RSI has enough valid data to plot
        if not candles[rsi_column].isnull().all():
            add_plots.append(mpf.make_addplot(candles[rsi_column], panel=2, color='blue', type='scatter', marker='o', markersize=ma_markersize, label='RSI'))
            add_plots.append(mpf.make_addplot(candles[rsi_column], panel=2, color='blue', linestyle='-', width=0.75))
            add_plots.append(mpf.make_addplot(candles['trend_indicator'], panel=2, color='white', type='scatter', marker='o', markersize=ma_markersize, label='RSI'))
            add_plots.append(mpf.make_addplot(candles['trend_indicator'], panel=2, color='white', linestyle='-', width=0.75))
            add_plots.append(mpf.make_addplot(candles['hundred_line'], panel=2, color='red', linestyle=':', secondary_y=False))
            add_plots.append(mpf.make_addplot(candles['fifty_line'], panel=2, color='yellow', linestyle=':', secondary_y=False))
            add_plots.append(mpf.make_addplot(candles['zero_line'], panel=2, color='green', linestyle=':', secondary_y=False))
            add_plots.append(mpf.make_addplot(candles['trend_high_threshold'], panel=2, color='white', linestyle=':', secondary_y=False))
            add_plots.append(mpf.make_addplot(candles['trend_low_threshold'], panel=2, color='white', linestyle=':', secondary_y=False))
        else:
            print("RSI has not warmed up yet. Plotting without it.")

        # Add buy, sell, and neutral markers if signal_column exists. Eliminate the if else statement to revert to working order
        if ma_entry_price in candles.columns and ma_exit_price in candles.columns:
            add_plots.append(mpf.make_addplot(candles[ma_entry_price], type='scatter', marker='^', markersize=signal_markersize_y, color='yellow', panel=0, secondary_y=False))
            add_plots.append(mpf.make_addplot(candles[ma_exit_price], type='scatter', marker='o', markersize=signal_markersize_y, color='yellow', panel=0, secondary_y=False))
        else:
            print("Buy/Sell markers for MA strat have not warmed up yet. Plotting without them.")

        # Add buy, sell, and neutral markers for RSI strategy
        if rsi_entry_price in candles.columns and rsi_exit_price in candles.columns:
            add_plots.append(mpf.make_addplot(candles[rsi_entry_price], type='scatter', marker='^', markersize=signal_markersize_b, color='blue', panel=0, secondary_y=False))
            add_plots.append(mpf.make_addplot(candles[rsi_exit_price], type='scatter', marker='o', markersize=signal_markersize_b, color='blue', panel=0, secondary_y=False))
        else:
            print("Buy/Sell markers for RSI strat have not warmed up yet. Plotting without them.")

        # Add cumulative profit plots on a secondary y-axis with dynamic names
        add_plots.append(mpf.make_addplot(candles[cum_pnl_ma_col], panel=0, color='yellow', secondary_y=True, label=f'Cumulative PnL (MA: {ma_name1}_{ma_name2})', linestyle='-', width=1.25))
        add_plots.append(mpf.make_addplot(candles[cum_pnl_rsi_col], panel=0, color='blue', secondary_y=True, label=f'Cumulative PnL (RSI: {rsi_column})', linestyle='-', width=1.25))
        add_plots.append(mpf.make_addplot(candles[cum_pnl_all_col], panel=0, color='green', secondary_y=True, label=f'Cumulative PnL (Combined)', linestyle='-', width=1.25))

        # Add stop-loss markers (x) for both MA and RSI strategies
        # if 'stop_loss_ma' in candles.columns:
        add_plots.append(mpf.make_addplot(candles[stop_loss_ma], type='scatter', marker='x', markersize=100, color='yellow', panel=0, secondary_y=False))
        # else:
        #     print("There are no stop loss markers for MA strat")
        # if 'stop_loss_rsi' in candles.columns:
        add_plots.append(mpf.make_addplot(candles[stop_loss_rsi], type='scatter', marker='x', markersize=50, color='blue', panel=0, secondary_y=False))
        # else:
        #     print("There are no stop loss markers for RSI strat")

        # Add price action envelope as white lines
        if 'price_action_upper' in candles.columns and 'price_action_lower' in candles.columns:
            add_plots.append(mpf.make_addplot(candles['price_action_upper'], color='white', linestyle='-', width=0.5, label='Price Action Upper'))
            add_plots.append(mpf.make_addplot(candles['price_action_lower'], color='white', linestyle='-', width=0.5, label='Price Action Lower'))
            # add_plots.append(mpf.make_addplot(candles['ma_price_action_upper'], color='white', linestyle='-', width=0.5, label='Price Action Upper'))
            # add_plots.append(mpf.make_addplot(candles['ma_price_action_lower'], color='white', linestyle='-', width=0.5, label='Price Action Lower'))
        else:
            print("Price action envelope not calculating properly")

        # Create a custom style with a black background
        black_style = mpf.make_mpf_style(
            base_mpf_style='charles',  # Start with the 'charles' style and modify it
            facecolor='black',         # Set the background color to black
            gridcolor='black',          # Set the grid line color
            edgecolor='purple',          # Set the edge color for candles and boxes
            figcolor='black',          # Set the figure background color to black
            rc={'axes.labelcolor': 'yellow', 
                'xtick.color': 'yellow', 
                'ytick.color': 'yellow', 
                'axes.titlecolor': 'yellow',
                'font.size': font_size, 
                'axes.labelsize': font_size,
                'axes.titlesize': font_size,
                'xtick.labelsize': font_size,
                'ytick.labelsize': font_size,
                'legend.fontsize': font_size}  # Set tick and label colors to white
        )

        # Plot using mplfinance
        mpf.plot(plot_data, type='candle', style=black_style, 
                title='',
                ylabel='Price', 
                addplot=add_plots, 
                figsize=figsize,
                volume=True,
                panel_ratios=(8, 2),
                #  panel_ratios=(8, 2, 2),             
                tight_layout=True)
    except Exception as e:
        print(f"Something wrong in the plotting_moving_averages function: {e}")

def visualize_trades_1(candles, ticker_to_tick_size, ticker_to_point_value, ma_name1='wma_5', ma_name2='sma_5', rsi_column='rsi_5', lower_slice=0, upper_slice=-1):
    """
    Visualize trades and print summary statistics, including tick size for each ticker.

    Parameters:
    - candles (dict): Dictionary of DataFrames with candle data for each ticker.
    - ticker_to_tick_size (dict): Dictionary mapping tickers to their respective tick sizes.
    - ticker_to_point_value (dict): Dictionary mapping tickers to their respective point values.
    - ma_name1, ma_name2, rsi_column: Names of MA and RSI columns.
    - lower_slice, upper_slice: Range of rows to visualize.
    """
    # Generate dynamic column names for PnL and signals
    pnl_ma_col = f'pnl_{ma_name1}_{ma_name2}'
    pnl_rsi_col = f'pnl_{rsi_column}'
    cum_pnl_ma_col = f'cum_{pnl_ma_col}'
    cum_pnl_rsi_col = f'cum_{pnl_rsi_col}'
    cum_pnl_all_col = f'cum_pnl_all_{ma_name1}_{ma_name2}_{rsi_column}'
    ma_entry_price = f'entry_price_{ma_name1}_{ma_name2}'
    ma_exit_price = f'exit_price_{ma_name1}_{ma_name2}'
    rsi_entry_price = f'entry_price_{rsi_column}'
    rsi_exit_price = f'exit_price_{rsi_column}'

    # Variable to accumulate total dollar PnL
    total_dollar_pnl_sum = 0.0

    # Iterate through the candles dictionary
    for ticker, minute_candles_df in candles.items():
        # Create a copy of the DataFrame for the specified slice
        minute_candles_viz_1 = minute_candles_df[lower_slice:upper_slice].copy()
        tick_size = ticker_to_tick_size.get(ticker, "Unknown")  # Retrieve tick size or default to "Unknown"
        point_value = ticker_to_point_value.get(ticker, 1)  # Retrieve point value or default to 1

        try:
            # Plot moving averages
            plot_moving_averages_1(
                minute_candles_viz_1,
                ma_name1=ma_name1, ma_name2=ma_name2, rsi_column=rsi_column,
                figsize=(40, 20), font_size=20,
                ma_markersize=50, signal_markersize_y=450, signal_markersize_b=300
            )
            
            # Calculate total dollar PnL for this ticker
            total_dollar_pnl = minute_candles_viz_1[cum_pnl_all_col].iloc[-1] * point_value
            total_dollar_pnl_sum += total_dollar_pnl  # Accumulate dollar PnL

            # Print out the summary for the ticker
            print(
                f"{ticker}: {len(minute_candles_df)} rows, "
                f"{minute_candles_viz_1[ma_exit_price].notna().sum()} MA trades, "
                f"{minute_candles_viz_1[rsi_exit_price].notna().sum()} RSI trades, "
                f"{minute_candles_viz_1[cum_pnl_all_col].iloc[-1]:.2f} total point PnL, "
                f"{total_dollar_pnl:.2f} total dollar PnL, "
                f"Tick Size: {tick_size}"
            )
        except Exception as e:
            # Handle any errors that occur during the plotting
            print(f"Error in visualize_trades_1 for {ticker}: {e}")

    # Print total dollar PnL sum across all tickers
    print(f"\nTotal Dollar PnL Across All Tickers: {total_dollar_pnl_sum:.2f}")

window_size = 5

sig_ma = f'wma_{window_size}'
con_ma = f'sma_{window_size}'
rsi_col = f'rsi_{window_size}'

visualize_trades_1(
    candles=minute_candles_1, 
    ticker_to_tick_size=ticker_to_tick_size,
    ticker_to_point_value=ticker_to_point_value,    
    ma_name1=sig_ma, 
    ma_name2=con_ma, 
    rsi_column=rsi_col, 
    lower_slice=0, 
    upper_slice=-1
)

def print_all_pnls(candles, ma_name1='wma_5', ma_name2='sma_5', rsi_column='rsi_5'):
    # Generate dynamic column names for PnL
    pnl_ma_col = f'pnl_{ma_name1}_{ma_name2}'
    pnl_rsi_col = f'pnl_{rsi_column}'
    cum_pnl_ma_col = f'cum_{pnl_ma_col}'
    cum_pnl_rsi_col = f'cum_{pnl_rsi_col}'
    cum_pnl_all_col = f'cum_pnl_all_{ma_name1}_{ma_name2}_{rsi_column}'
    for ticker, minute_candles_df in candles.items():
        ma_pnl = round(minute_candles_df[cum_pnl_ma_col].iloc[-1], 3)
        rsi_pnl = round(minute_candles_df[cum_pnl_rsi_col].iloc[-1], 3)
        total_pnl = round(minute_candles_df[cum_pnl_all_col].iloc[-1], 3)
        close_price_diff = round(minute_candles_df["close"].iloc[-1] - minute_candles_df["close"].iloc[0], 3)
        point_alpha = round(total_pnl - close_price_diff, 3)
        print(f'{ticker} : {total_pnl} Total pnl, {ma_pnl} MA pnl, {rsi_pnl} RSI pnl, {close_price_diff} Close Price Difference, {point_alpha} Point Alpha, {compression_factor} Compression Factor')

# Call print_all_pnls() to print all profit figures and close price differences
print_all_pnls(minute_candles_1)

########################################################################################################################################
# Blind refactoring by ChatGPT- May not work as expected and may need to be updated as 
# `vectorized without iterrows` is updated. Always closes the position at the next candle close after position is opened (bad)

def generate_signals(candles, ma_name1, ma_name2, rsi_column):
    """ Blind refactoring by ChatGPT
    Generate buy/sell signals for moving averages and RSI indicators.
    """
    # Moving Average Signals
    candles[f'signal_{ma_name1}_{ma_name2}'] = (
        (candles[ma_name1] <= candles[ma_name2]).astype(int).diff().fillna(0).clip(lower=0)
    )
    candles['ma_position_open'] = candles[f'signal_{ma_name1}_{ma_name2}'].astype(bool)

    # RSI Signals
    candles[f'signal_{rsi_column}'] = (
        (candles[rsi_column] < 50).astype(int).diff().fillna(0).clip(lower=0)
    )
    candles['rsi_position_open'] = candles[f'signal_{rsi_column}'].astype(bool)

    return candles

def determine_entry_exit_prices(candles, signal_column, price_column, tick_size, entry_col, exit_col):
    """ Blind refactoring by ChatGPT
    Determine entry and exit prices for a given signal column.
    """
    entry_mask = (candles[signal_column] == 1) & (candles[signal_column].shift(1) != 1)
    exit_mask = (candles[signal_column] == 0) & (candles[signal_column].shift(1) == 1)

    candles[entry_col] = None
    candles[exit_col] = None
    candles.loc[entry_mask, entry_col] = candles[price_column][entry_mask] + tick_size
    candles.loc[exit_mask, exit_col] = candles[price_column][exit_mask] - tick_size

    return candles

def calculate_stop_losses(candles, entry_col, exit_col, stop_loss_col, position_open_col, span_col):
    """ Blind refactoring by ChatGPT
    Calculate stop losses and update position open states.
    """
    entry_mask = candles[entry_col].notnull()
    exit_mask = candles[exit_col].notnull()

    candles[stop_loss_col] = None
    candles.loc[entry_mask, stop_loss_col] = candles[entry_col] - candles[span_col]
    candles.loc[entry_mask, position_open_col] = True

    candles.loc[exit_mask, stop_loss_col] = None
    candles.loc[exit_mask, position_open_col] = False

    # Forward-fill stop loss and position state
    candles[stop_loss_col] = candles[stop_loss_col].ffill()
    candles[position_open_col] = candles[position_open_col].ffill().fillna(False)

    return candles

def track_stop_loss_hits(candles, stop_loss_col, price_col, position_open_col, stop_loss_hit_col, exit_col, signal_col, tick_size):
    """ Blind refactoring by ChatGPT
    Track whether stop losses have been hit and update exit prices and states.
    """
    candles[stop_loss_hit_col] = False

    stop_loss_hit = (candles[stop_loss_col].notnull()) & (candles[price_col] <= candles[stop_loss_col]) & candles[position_open_col]
    candles.loc[stop_loss_hit, stop_loss_hit_col] = True
    candles.loc[stop_loss_hit, exit_col] = candles[price_col][stop_loss_hit] - tick_size
    candles.loc[stop_loss_hit, signal_col] = 0
    candles.loc[stop_loss_hit, position_open_col] = False
    candles.loc[stop_loss_hit, stop_loss_col] = None

    return candles

def calculate_profit_loss(candles, entry_col, exit_col, pnl_col, multiplier):
    """ Blind refactoring by ChatGPT
    Calculate profit and loss for a strategy.
    """
    entry_indices = candles.index[candles[entry_col].notnull()]
    exit_indices = candles.index[candles[exit_col].notnull()]

    valid_pairs = min(len(entry_indices), len(exit_indices))
    entry_prices = candles.loc[entry_indices[:valid_pairs], entry_col].values
    exit_prices = candles.loc[exit_indices[:valid_pairs], exit_col].values

    pnl = (exit_prices - entry_prices) * multiplier
    candles[pnl_col] = 0.0
    candles.loc[exit_indices[:valid_pairs], pnl_col] = pnl

    return candles

def apply_strategy_to_dataframe(candles, ma_name1, ma_name2, rsi_column, tick_size, multiplier):
    """ Blind refactoring by ChatGPT
    Apply the trading strategy logic to a single DataFrame.
    """
    # Step 1: Generate signals
    candles = generate_signals(candles, ma_name1, ma_name2, rsi_column)

    # Step 2: Determine entry and exit prices
    candles = determine_entry_exit_prices(candles, f'signal_{ma_name1}_{ma_name2}', 'close', tick_size, 'ma_entry_price', 'ma_exit_price')
    candles = determine_entry_exit_prices(candles, f'signal_{rsi_column}', 'close', tick_size, 'rsi_entry_price', 'rsi_exit_price')

    # Step 3: Calculate stop losses
    candles = calculate_stop_losses(candles, 'ma_entry_price', 'ma_exit_price', 'stop_loss_ma', 'ma_position_open', 'candle_span_max')
    candles = calculate_stop_losses(candles, 'rsi_entry_price', 'rsi_exit_price', 'stop_loss_rsi', 'rsi_position_open', 'candle_span_max')

    # Step 4: Track stop loss hits
    candles = track_stop_loss_hits(candles, 'stop_loss_ma', 'close', 'ma_position_open', 'ma_stop_loss_hit', 'ma_exit_price', f'signal_{ma_name1}_{ma_name2}', tick_size)
    candles = track_stop_loss_hits(candles, 'stop_loss_rsi', 'close', 'rsi_position_open', 'rsi_stop_loss_hit', 'rsi_exit_price', f'signal_{rsi_column}', tick_size)

    # Step 5: Calculate profit and loss
    candles = calculate_profit_loss(candles, 'ma_entry_price', 'ma_exit_price', f'pnl_{ma_name1}_{ma_name2}', multiplier)
    candles = calculate_profit_loss(candles, 'rsi_entry_price', 'rsi_exit_price', f'pnl_{rsi_column}', multiplier)

    return candles

for key, df in minute_candles_1.items():
    minute_candles_1[key] = apply_strategy_to_dataframe(df, 'wma_5', 'sma_5', 'rsi_5', tick_size=0.01, multiplier=1)

########################################################################################################################################
# With iterrows, revert to if things get funky- works somewhat but is slow, inefficient, repeats exit prices

def check_moving_average_1(candles, ma_name1='wma_5', ma_name2='sma_5', rsi_column='rsi_5'):
    """ With iterrows, revert to if things get funky
    Generate buy/sell signals based on moving averages and RSI indicators.

    Parameters:
    - candles (pd.DataFrame): The DataFrame containing candle data.
    - ma_name1 (str): Column name for the first moving average.
    - ma_name2 (str): Column name for the second moving average.
    - rsi_column (str): Column name for the RSI indicator.

    Returns:
    - candles (pd.DataFrame): The DataFrame with updated signal columns.
    """
    # Initialize signal columns
    candles[f'signal_{ma_name1}_{ma_name2}'] = 0
    candles[f'signal_{rsi_column}'] = 0

    ma_position_open = False
    rsi_position_open = False

    for i, row in candles.iterrows():
        # Moving Average Signal Generation
        if i == 0:  # First row, no previous row to compare
            candles.at[i, f'signal_{ma_name1}_{ma_name2}'] = 0
        elif not ma_position_open and row[ma_name1] <= row[ma_name2]:
            ma_position_open = True
            candles.at[i, f'signal_{ma_name1}_{ma_name2}'] = 1
        elif ma_position_open and row[ma_name1] > row[ma_name2]:
            ma_position_open = False
            candles.at[i, f'signal_{ma_name1}_{ma_name2}'] = 0
        else:
            candles.at[i, f'signal_{ma_name1}_{ma_name2}'] = candles.at[i - 1, f'signal_{ma_name1}_{ma_name2}']

        # RSI Signal Generation
        if i == 0:  # First row, no previous row to compare
            candles.at[i, f'signal_{rsi_column}'] = 0
        elif not rsi_position_open and row[rsi_column] < 50:
            rsi_position_open = True
            candles.at[i, f'signal_{rsi_column}'] = 1
        elif rsi_position_open and row[rsi_column] >= 50:
            rsi_position_open = False
            candles.at[i, f'signal_{rsi_column}'] = 0
        else:
            candles.at[i, f'signal_{rsi_column}'] = candles.at[i - 1, f'signal_{rsi_column}']

    return candles

def determine_entry_exit_prices(candles, ma_name1='wma_5', ma_name2='sma_5', rsi_column='rsi_5', ticker_to_tick_size=None, ticker=None):
    """ Using iterrows, revert to if things get funky
    Determine entry and exit prices for MA and RSI strategies based on signals.

    Returns:
    - candles (pd.DataFrame): The DataFrame with updated entry and exit price columns.
    """
    # Initialize columns
    candles['ma_entry_price'] = None
    candles['ma_exit_price'] = None
    candles['rsi_entry_price'] = None
    candles['rsi_exit_price'] = None

    # Get tick size
    tick_size = ticker_to_tick_size.get(ticker, 0) if ticker_to_tick_size else 0

    # Track positions
    ma_position_open = False
    rsi_position_open = False

    for i, row in candles.iterrows():
        # Moving Average Strategy
        if row[f'signal_{ma_name1}_{ma_name2}'] == 1 and not ma_position_open:
            ma_position_open = True
            candles.at[i, 'ma_entry_price'] = row['close'] + tick_size
        elif row[f'signal_{ma_name1}_{ma_name2}'] == 0 and ma_position_open:
            ma_position_open = False
            candles.at[i, 'ma_exit_price'] = row['close'] - tick_size

        # RSI Strategy
        if row[f'signal_{rsi_column}'] == 1 and not rsi_position_open:
            rsi_position_open = True
            candles.at[i, 'rsi_entry_price'] = row['close'] + tick_size
        elif row[f'signal_{rsi_column}'] == 0 and rsi_position_open:
            rsi_position_open = False
            candles.at[i, 'rsi_exit_price'] = row['close'] - tick_size

    return candles

def calculate_stop_losses(candles):
    """ With iterrows, revert to if things get funky
    Calculate stop loss levels for MA and RSI strategies and ensure they persist while positions are open.

    Returns:
    - candles (pd.DataFrame): The DataFrame with stop loss columns updated.
    """
    # Initialize stop loss columns
    candles['stop_loss_ma'] = None
    candles['stop_loss_rsi'] = None
    candles['ma_position_open'] = False
    candles['rsi_position_open'] = False

    for i, row in candles.iterrows():
        # Persist stop loss if position remains open
        if i > 0:
            # Carry forward stop loss only if the position remains open
            if candles.at[i - 1, 'ma_position_open']:
                candles.at[i, 'stop_loss_ma'] = candles.at[i - 1, 'stop_loss_ma']
                candles.at[i, 'ma_position_open'] = True
            if candles.at[i - 1, 'rsi_position_open']:
                candles.at[i, 'stop_loss_rsi'] = candles.at[i - 1, 'stop_loss_rsi']
                candles.at[i, 'rsi_position_open'] = True

        # Moving Average Stop Loss
        if row['ma_entry_price'] is not None:
            # Set stop loss on entry
            candles.at[i, 'stop_loss_ma'] = row['ma_entry_price'] - row['candle_span_max']
            candles.at[i, 'ma_position_open'] = True  # Mark position as open
        elif row['ma_exit_price'] is not None:
            # Close position and reset stop loss
            candles.at[i, 'ma_position_open'] = False

        # RSI Stop Loss
        if row['rsi_entry_price'] is not None:
            # Set stop loss on entry
            candles.at[i, 'stop_loss_rsi'] = row['rsi_entry_price'] - row['candle_span_max']
            candles.at[i, 'rsi_position_open'] = True  # Mark position as open
        elif row['rsi_exit_price'] is not None:
            # Close position and reset stop loss
            candles.at[i, 'rsi_position_open'] = False

    return candles

def track_stop_loss_hits(candles, ma_name1='wma_5', ma_name2='sma_5', rsi_column='rsi_5', ticker_to_tick_size=None, ticker=None):
    """  With iterrows, revert to if things get funky
    Track whether stop losses have been hit for MA and RSI strategies, and update position open status and exit prices.

    Returns:
    - candles (pd.DataFrame): The DataFrame with stop loss hit flags, adjusted signals, and updated exit prices.
    """
    # Initialize stop loss hit columns
    candles['ma_stop_loss_hit'] = False
    candles['rsi_stop_loss_hit'] = False

    # Get tick size
    tick_size = ticker_to_tick_size.get(ticker, 0) if ticker_to_tick_size else 0

    for i, row in candles.iterrows():
        # Moving Average Stop Loss Hit Logic
        if row['stop_loss_ma'] is not None and row['close'] <= row['stop_loss_ma']:
            if candles.at[i, 'ma_position_open']:  # Only process if the position is still open
                candles.at[i, 'ma_stop_loss_hit'] = True
                candles.at[i, 'ma_exit_price'] = row['close'] - tick_size  # Set exit price at stop loss
                candles.at[i, f'signal_{ma_name1}_{ma_name2}'] = 0  # Reset signal
                candles.at[i, 'ma_position_open'] = False  # Close position
                candles.at[i, 'stop_loss_ma'] = None  # Clear stop loss

        # RSI Stop Loss Hit Logic
        if row['stop_loss_rsi'] is not None and row['close'] <= row['stop_loss_rsi']:
            if candles.at[i, 'rsi_position_open']:  # Only process if the position is still open
                candles.at[i, 'rsi_stop_loss_hit'] = True
                candles.at[i, 'rsi_exit_price'] = row['close'] - tick_size  # Set exit price at stop loss
                candles.at[i, f'signal_{rsi_column}'] = 0  # Reset signal
                candles.at[i, 'rsi_position_open'] = False  # Close position
                candles.at[i, 'stop_loss_rsi'] = None  # Clear stop loss

    return candles

def calculate_profit_loss_1(candles, multiplier=1, ma_name1='wma_5', ma_name2='sma_5', rsi_column='rsi_5'):
    """ With iterrows, revert to if things get funky
    Calculate profit and loss based on entry and exit price columns.

    Parameters:
    - candles (pd.DataFrame): The DataFrame containing candle data with entry and exit price columns.
    - multiplier (float): The multiplier for PnL calculation (e.g., contract size).
    - ma_name1 (str): Column name for the first moving average.
    - ma_name2 (str): Column name for the second moving average.
    - rsi_column (str): Column name for the RSI indicator.

    Returns:
    - candles (pd.DataFrame): The DataFrame with dynamically named profit/loss columns.
    """
    # Generate dynamic column names
    pnl_ma_col = f'pnl_{ma_name1}_{ma_name2}'
    pnl_rsi_col = f'pnl_{rsi_column}'
    cum_pnl_ma_col = f'cum_{pnl_ma_col}'
    cum_pnl_rsi_col = f'cum_{pnl_rsi_col}'
    cum_pnl_all_col = f'cum_pnl_all_{ma_name1}_{ma_name2}_{rsi_column}'  # Dynamically named cumulative PnL column

    # Initialize PnL columns
    candles[pnl_ma_col] = 0.0
    candles[pnl_rsi_col] = 0.0

    # Initialize variables to track state
    ma_position_open = False
    rsi_position_open = False

    ma_entry_price = None
    rsi_entry_price = None

    try:
        for i, row in candles.iterrows():
            # Moving Average Strategy
            if row['ma_entry_price'] is not None and not ma_position_open:
                # Open MA position
                ma_entry_price = row['ma_entry_price']
                ma_position_open = True
                candles.at[i, pnl_ma_col] = 0  # No PnL on opening a position

            elif row['ma_exit_price'] is not None and ma_position_open:
                # Close MA position
                ma_exit_price = row['ma_exit_price']
                ma_position_open = False
                # Calculate PnL
                pnl_ma = (ma_exit_price - ma_entry_price) * multiplier
                candles.at[i, pnl_ma_col] = pnl_ma
                # Reset entry price
                ma_entry_price = None

            # RSI Strategy
            if row['rsi_entry_price'] is not None and not rsi_position_open:
                # Open RSI position
                rsi_entry_price = row['rsi_entry_price']
                rsi_position_open = True
                candles.at[i, pnl_rsi_col] = 0  # No PnL on opening a position

            elif row['rsi_exit_price'] is not None and rsi_position_open:
                # Close RSI position
                rsi_exit_price = row['rsi_exit_price']
                rsi_position_open = False
                # Calculate PnL
                pnl_rsi = (rsi_exit_price - rsi_entry_price) * multiplier
                candles.at[i, pnl_rsi_col] = pnl_rsi
                # Reset entry price
                rsi_entry_price = None

        # Calculate cumulative PnL for MA and RSI strategies
        candles[cum_pnl_ma_col] = candles[pnl_ma_col].cumsum()
        candles[cum_pnl_rsi_col] = candles[pnl_rsi_col].cumsum()

        # Calculate cumulative PnL for both strategies combined, with dynamic naming
        candles[cum_pnl_all_col] = candles[cum_pnl_ma_col] + candles[cum_pnl_rsi_col]

    except Exception as e:
        print(f"Error in calculate_profit_loss_1: {e}")

    return candles

########################################################################################################################################
# Refactored trade logic according to written explanation of states: from ChatGPT, RSI separated from MA

def generate_trading_signals(candles, ma_name1='wma_5', ma_name2='sma_5', rsi_column='rsi_5'):
    """
    Generate trading signals and handle state transitions based on WMA and SMA.

    Parameters:
    - candles (pd.DataFrame): The DataFrame containing candle data.
    - ma_name1 (str): Column name for the WMA.
    - ma_name2 (str): Column name for the SMA.
    - rsi_column (str): Column name for the RSI.

    Returns:
    - pd.DataFrame: Updated DataFrame with signals and state columns.
    """
    # Initialize columns
    candles[f'signal_{ma_name1}_{ma_name2}'] = 0
    candles['ma_state'] = 0  # 0: Initial, 1: Armed, 2: Long Position
    candles['stop_loss_ma'] = None

    # Track state
    armed = False
    position_open = False
    stop_loss = None

    # Iterate through candles
    for i in range(len(candles)):
        wma = candles.at[i, ma_name1]
        sma = candles.at[i, ma_name2]
        close_price = candles.at[i, 'close']

        if not position_open:  # Initial State
            if wma > sma and not armed:
                armed = True
                candles.at[i, 'ma_state'] = 1  # Armed
            elif wma <= sma and armed:
                armed = False  # Reset to Initial
                candles.at[i, 'ma_state'] = 0

        if armed:  # 2nd State: Armed
            if wma <= sma and not position_open:
                position_open = True
                stop_loss = close_price - candles.at[i, 'candle_span_max']
                candles.at[i, f'signal_{ma_name1}_{ma_name2}'] = 1  # Enter Long
                candles.at[i, 'stop_loss_ma'] = stop_loss
                candles.at[i, 'ma_state'] = 2  # Long Position

        if position_open:  # 3rd State: Long Position
            if close_price <= stop_loss:  # Stop Loss Hit
                position_open = False
                armed = False
                candles.at[i, f'signal_{ma_name1}_{ma_name2}'] = -1  # Exit
                candles.at[i, 'stop_loss_ma'] = None
                candles.at[i, 'ma_state'] = 0  # Reset to Initial
            elif wma > sma:  # WMA Above SMA: Revert to Armed
                position_open = False
                armed = True
                candles.at[i, f'signal_{ma_name1}_{ma_name2}'] = -1  # Exit
                candles.at[i, 'stop_loss_ma'] = None
                candles.at[i, 'ma_state'] = 1  # Armed

    return candles

def determine_entry_exit_prices(candles, signal_column, entry_col, exit_col, tick_size):
    """
    Determine entry and exit prices based on trading signals.
    """
    candles[entry_col] = None
    candles[exit_col] = None

    entry_mask = candles[signal_column] == 1
    exit_mask = candles[signal_column] == -1

    candles.loc[entry_mask, entry_col] = candles['close'][entry_mask] + tick_size
    candles.loc[exit_mask, exit_col] = candles['close'][exit_mask] - tick_size

    return candles

def track_stop_loss_hits(candles, stop_loss_col, close_col, position_col, signal_col):
    """
    Track stop loss hits and update positions and signals.
    """
    stop_loss_hit = (candles[close_col] <= candles[stop_loss_col]) & (candles[position_col])
    candles.loc[stop_loss_hit, signal_col] = -1
    candles.loc[stop_loss_hit, stop_loss_col] = None
    candles.loc[stop_loss_hit, position_col] = False
    return candles

def calculate_profit_loss(candles, entry_col, exit_col, pnl_col, multiplier):
    """
    Calculate profit and loss for a strategy.
    """
    candles[pnl_col] = 0.0
    pnl = (candles[exit_col] - candles[entry_col]) * multiplier
    candles.loc[candles[exit_col].notnull(), pnl_col] = pnl
    return candles

for ticker, df in minute_candles_1.items():
    # Generate signals
    for sig_ma, con_ma, rsi_col in ma_combinations:
        df = generate_trading_signals(df, ma_name1=sig_ma, ma_name2=con_ma, rsi_column=rsi_col)

    # Determine entry and exit prices
    df = determine_entry_exit_prices(df, signal_column=f'signal_wma_5_sma_5', entry_col='ma_entry_price', exit_col='ma_exit_price', tick_size=0.01)

    # Track stop loss hits
    df = track_stop_loss_hits(df, stop_loss_col='stop_loss_ma', close_col='close', position_col='ma_position_open', signal_col='signal_wma_5_sma_5')

    # Calculate profit and loss
    df = calculate_profit_loss(df, entry_col='ma_entry_price', exit_col='ma_exit_price', pnl_col='pnl_wma_5_sma_5', multiplier=1)

    # Store the updated DataFrame back
    minute_candles_1[ticker] = df

def generate_rsi_signals(candles, rsi_column='rsi_5', threshold=50):
    """
    Generate RSI trading signals and position states.

    Parameters:
    - candles (pd.DataFrame): The DataFrame containing candle data.
    - rsi_column (str): Column name for the RSI indicator.
    - threshold (int): The RSI threshold for long/short signals.

    Returns:
    - pd.DataFrame: Updated DataFrame with RSI signals and position state.
    """
    candles[f'signal_{rsi_column}'] = 0
    candles['rsi_position_open'] = False

    # Iterate through candles to generate signals
    rsi_position_open = False

    for i in range(len(candles)):
        rsi = candles.at[i, rsi_column]

        if not rsi_position_open and rsi < threshold:  # RSI below threshold, enter long position
            rsi_position_open = True
            candles.at[i, f'signal_{rsi_column}'] = 1
            candles.at[i, 'rsi_position_open'] = True
        elif rsi_position_open and rsi >= threshold:  # RSI crosses above threshold, exit position
            rsi_position_open = False
            candles.at[i, f'signal_{rsi_column}'] = -1
            candles.at[i, 'rsi_position_open'] = False
        else:  # Maintain previous state
            candles.at[i, f'signal_{rsi_column}'] = 0
            candles.at[i, 'rsi_position_open'] = rsi_position_open

    return candles

def determine_rsi_entry_exit_prices(candles, signal_column, entry_col, exit_col, tick_size=0.01):
    """
    Determine entry and exit prices based on RSI trading signals.

    Parameters:
    - candles (pd.DataFrame): The DataFrame containing candle data.
    - signal_column (str): Column name for the RSI signal.
    - entry_col (str): Column name for the RSI entry price.
    - exit_col (str): Column name for the RSI exit price.
    - tick_size (float): Tick size adjustment for entry/exit prices.

    Returns:
    - pd.DataFrame: Updated DataFrame with RSI entry and exit prices.
    """
    candles[entry_col] = None
    candles[exit_col] = None

    entry_mask = candles[signal_column] == 1
    exit_mask = candles[signal_column] == -1

    candles.loc[entry_mask, entry_col] = candles['close'][entry_mask] + tick_size
    candles.loc[exit_mask, exit_col] = candles['close'][exit_mask] - tick_size

    return candles

def track_rsi_stop_loss_hits(candles, stop_loss_col, close_col, signal_column, position_open_col):
    """
    Track stop loss hits for RSI strategy.

    Parameters:
    - candles (pd.DataFrame): The DataFrame containing candle data.
    - stop_loss_col (str): Column name for the RSI stop loss level.
    - close_col (str): Column name for the closing price.
    - signal_column (str): Column name for the RSI signal.
    - position_open_col (str): Column name for the RSI position open state.

    Returns:
    - pd.DataFrame: Updated DataFrame with stop loss information.
    """
    stop_loss_hit = (candles[close_col] <= candles[stop_loss_col]) & candles[position_open_col]
    candles.loc[stop_loss_hit, signal_column] = -1  # Exit signal
    candles.loc[stop_loss_hit, position_open_col] = False
    candles.loc[stop_loss_hit, stop_loss_col] = None

    return candles

def calculate_rsi_profit_loss(candles, entry_col, exit_col, pnl_col, multiplier=1):
    """
    Calculate profit and loss for RSI strategy.

    Parameters:
    - candles (pd.DataFrame): The DataFrame containing candle data.
    - entry_col (str): Column name for RSI entry prices.
    - exit_col (str): Column name for RSI exit prices.
    - pnl_col (str): Column name for the RSI profit/loss values.
    - multiplier (float): Multiplier for profit/loss calculation.

    Returns:
    - pd.DataFrame: Updated DataFrame with RSI profit/loss values.
    """
    candles[pnl_col] = 0.0
    pnl = (candles[exit_col] - candles[entry_col]) * multiplier
    candles.loc[candles[exit_col].notnull(), pnl_col] = pnl
    return candles

for ticker, df in minute_candles_1.items():
    # Step 1: Generate RSI signals
    df = generate_rsi_signals(df, rsi_column='rsi_5', threshold=50)
    
    # Step 2: Determine RSI entry and exit prices
    df = determine_rsi_entry_exit_prices(df, signal_column='signal_rsi_5', 
                                         entry_col='rsi_entry_price', exit_col='rsi_exit_price', 
                                         tick_size=0.01)
    
    # Step 3: Track RSI stop loss hits
    df = track_rsi_stop_loss_hits(df, stop_loss_col='stop_loss_rsi', close_col='close', 
                                  signal_column='signal_rsi_5', position_open_col='rsi_position_open')
    
    # Step 4: Calculate RSI profit and loss
    df = calculate_rsi_profit_loss(df, entry_col='rsi_entry_price', exit_col='rsi_exit_price', 
                                   pnl_col='pnl_rsi_5', multiplier=1)
    
    # Update the DataFrame back into the dictionary
    minute_candles_1[ticker] = df


########################################################################################################################################
# The `old` old logic- everything from trade logic to visualization. Keeping as a safety net

def check_moving_average(candles, ma_name1='wma_5', ma_name2='sma_5', rsi_column='rsi_5'): # , signal_column='signal_wma_5_sma_5', signal_column_2='signal_rsi_5'
    """ An attempt at implementing ma_stop_loss_hit logic into the function. produces some whacky results
    Parameters:
    - candles (pd.DataFrame): The DataFrame containing candle data.
    - ma_name1 (str): The column name for the first moving average (default is 'wma_5').
    - ma_name2 (str): The column name for the second moving average (default is 'sma_5').
    - rsi_column (str): The column name for the RSI indicator (default is 'rsi_5').
    
    Returns:
    - candles (pd.DataFrame): The DataFrame with updated 'signal' and stop loss values.
    """
    current_time_USE = datetime.now(pytz.timezone('US/Eastern')).strftime('%Y-%m-%d %H:%M:%S')

    ma_position_open = False  # Initialize the MA position to neutral
    ma_entry_price = None     # Initialize the MA entry price to None
    ma_stop_loss_hit = False  # Track if a stop loss has been hit

    rsi_position_open = False # Initialize the RSI position to neutral
    rsi_entry_price = None    # Initialize the RSI entry price to None

    try:

        # Iterate through each row in the candles DataFrame
        for i, row in candles.iterrows():

            # # Moving Average Strategy
            if ma_stop_loss_hit: # This is the wonky one with reverse logic. Behaves strangely with respect to stop losses and entries/exits
                # If stop loss was hit, stay neutral as long as ma_name1 remains below ma_name2
                if row[ma_name1] < row[ma_name2]:
                    ma_stop_loss_hit = True  # Reset stop loss hit state
                candles.at[i, f'signal_{ma_name1}_{ma_name2}'] = 0  # Stay market neutral
                candles.at[i, 'stop_loss_ma'] = None  # No active stop loss
                continue

            # if pd.notna(row[ma_name1]) and pd.notna(row[ma_name2]): # Uncomment this line and indent the following chunk to undo fuckery                # If neither MA column is all null, then proceed
            if not ma_position_open and row[ma_name1] <= row[ma_name2]:                                 # If no position is open and wma is below or equal to sma
                ma_position_open = True                                                                 # Set the MA position to open
                ma_entry_price = row['close']                                                           # Record the entry price for use in determining stop loss
                candles.at[i, f'signal_{ma_name1}_{ma_name2}'] = 1                                      # Set the MA signal column to long (1)
                candles.at[i, 'stop_loss_ma'] = (ma_entry_price - row['candle_span_max'])       # Set the stop loss level
            elif ma_position_open and row['close'] < (ma_entry_price - row['candle_span_max']): # If the price drops below the stop loss level
                ma_position_open = False                                                                # Set the MA position to neutral
                candles.at[i, f'signal_{ma_name1}_{ma_name2}'] = 0                                      # Set the MA signal column to neutral (0)
                candles.at[i, 'stop_loss_ma'] = None                                                    # Reset stop loss after it's triggered
            elif ma_position_open and row[ma_name1] > row[ma_name2]:                                    # If a position is open and wma is above sma
                ma_position_open = False                                                                # Set the MA position to neutral
                candles.at[i, f'signal_{ma_name1}_{ma_name2}'] = 0                                      # Set the MA signal column to neutral (0)
                candles.at[i, 'stop_loss_ma'] = None                                                    # Reset stop loss after closing the position due to MA crossover opposite day
            elif ma_position_open:                                                                      # If a position is still open past the closing conditions
                candles.at[i, 'stop_loss_ma'] = candles.at[i-1, 'stop_loss_ma']                         # Retain the stop loss value for risk management and visualization

            # # RSI Strategy
            # if pd.notna(row[rsi_column]): # Uncomment this line and indent the following chunk to undo fuckery                                                               # If the RSI column is not all null, then proceed
            if not rsi_position_open and row[rsi_column] < 50:                                          # If no position is open and RSI is below 50
                rsi_position_open = True                                                                # Set the RSI position to open
                rsi_entry_price = row['close']                                                          # Record the entry price for use in determining stop loss
                candles.at[i, f'signal_{rsi_column}'] = 1                                               # Set the RSI signal column to long (1)
                candles.at[i, 'stop_loss_rsi'] = rsi_entry_price - row['candle_span_max']       # Set the stop loss level
            elif rsi_position_open and row['close'] < rsi_entry_price - row['candle_span_max']: # If a position is open and the price drops below the stop loss level
                rsi_position_open = False                                                               # Set the RSI position to neutral
                candles.at[i, f'signal_{rsi_column}'] = 0                                               # Set the RSI signal column to neutral (0)
                candles.at[i, 'stop_loss_rsi'] = None                                                   # Reset stop loss after it's triggered
            elif rsi_position_open and row[rsi_column] >= 50:                                           # If a position is open and RSI is above or equal to 50
                rsi_position_open = False                                                               # Set the RSI position to neutral
                candles.at[i, f'signal_{rsi_column}'] = 0                                               # Set the RSI signal column to neutral (0)
                candles.at[i, 'stop_loss_rsi'] = None                                                   # Reset stop loss after closing the position due to RSI crossover opposite day
            elif rsi_position_open:                                                                     # If a position is still open past the closing conditions
                candles.at[i, 'stop_loss_rsi'] = candles.at[i-1, 'stop_loss_rsi']                       # Retain the stop loss value for risk management and visualization

    except:
        print('Something weird is happening in the check_moving_average function.')

    return candles

def add_signal_markers_long_only(df, 
                                 signal_column='signal_column', 
                                 price_col='close', 
                                 buy_marker='buy_marker', 
                                #  sell_marker='sell_marker', 
                                 neutral_marker='neutral_marker'):
    """
    Adds columns for buy, sell, and market neutral signal markers in the DataFrame.

    Parameters:
    - df (pd.DataFrame): The DataFrame containing the trading data.
    - signal_column (str): The name of the column containing the trading signals.
    """
    df[buy_marker] = None
    # df[sell_marker] = None
    df[neutral_marker] = None  # Add a column for market neutral markers

    previous_signal = df[signal_column].shift(1)
    signal_changes = df[signal_column] != previous_signal

    for idx, signal in df[signal_changes].iterrows():
        if signal[signal_column] == 1:
            df.at[idx, buy_marker] = signal[price_col]
        # elif signal[signal_column] == -1:
        #     df.at[idx, sell_marker] = signal[price_col]
        elif signal[signal_column] == 0:
            df.at[idx, neutral_marker] = signal[price_col]

    return df

def calculate_profit_loss(candles, signal_ma='signal_wma_5_sma_5', signal_rsi='signal_rsi_5', multiplier=1): # multiplier=?? for stocks, multiplier=50 for ES
    """
    Function to calculate profit and loss based on the buy/sell signals and price movement.
    Assumes you're buying one share at a time and selling it to exit. No short selling.
    
    Parameters:
    - candles (pd.DataFrame): The DataFrame containing candle data and signal columns.

    Returns:
    - candles (pd.DataFrame): The DataFrame with 'pnl_ma' and 'pnl_rsi' columns showing profit/loss for each trade.
    """
    
    # Add columns to store profit/loss for each strategy
    candles['pnl_ma'] = None
    candles['pnl_rsi'] = None
    
    ma_position_open = False
    rsi_position_open = False
    
    ma_buy_price = None
    rsi_buy_price = None

    try:    
        for i, row in candles.iterrows():
            # Moving Average Strategy
            if row[signal_ma] == 1 and not ma_position_open:
                # Buy condition for MA
                ma_buy_price = row['close']
                ma_position_open = True
                candles.at[i, 'pnl_ma'] = 0  # No PnL on the buy
                # print(f"MA Buy at {ma_buy_price}")
                
            elif row[signal_ma] == 0 and ma_position_open:
                # Sell condition for MA (exit position)
                ma_sell_price = row['close']
                ma_position_open = False
                # Calculate profit
                candles.at[i, 'pnl_ma'] = (ma_sell_price - ma_buy_price)*multiplier
                # print(f"MA Sell at {ma_sell_price}, PnL: {ma_sell_price - ma_buy_price}")
                
            # RSI Strategy
            if row[signal_rsi] == 1 and not rsi_position_open:
                # Buy condition for RSI
                rsi_buy_price = row['close']
                rsi_position_open = True
                candles.at[i, 'pnl_rsi'] = 0  # No PnL on the buy
                # print(f"RSI Buy at {rsi_buy_price}")
                
            elif row[signal_rsi] == 0 and rsi_position_open:
                # Sell condition for RSI (exit position)
                rsi_sell_price = row['close']
                rsi_position_open = False
                # Calculate profit
                candles.at[i, 'pnl_rsi'] = (rsi_sell_price - rsi_buy_price)*multiplier
                # print(f"RSI Sell at {rsi_sell_price}, PnL: {rsi_sell_price - rsi_buy_price}")
        
        # Replace any NaN values with 0 (if no trade happened on that row)
        candles['pnl_ma'].fillna(0, inplace=True)
        candles['pnl_rsi'].fillna(0, inplace=True)
        candles['cum_pnl_ma'] = candles['pnl_ma'].cumsum()
        candles['cum_pnl_rsi'] = candles['pnl_rsi'].cumsum()
        candles['cum_pnl_all'] = candles['cum_pnl_ma'] + candles['cum_pnl_rsi']

        # Add buy, sell, and neutral markers if signal_column exists
        if signal_ma:
            candles = add_signal_markers_long_only(candles, 
                                                signal_ma, 
                                                price_col='close')
        else:
            print('signal_ma must be missing')

        if signal_rsi:
            candles = add_signal_markers_long_only(candles, 
                                                signal_rsi, 
                                                price_col='close', 
                                                buy_marker='buy_marker_2', 
                                                #    sell_marker='sell_marker_2', 
                                                neutral_marker='neutral_marker_2')
        else:
            print('signal_rsi must be missing')
    except:
        print('Something weird going on in calculate_profit_loss')
        
    return candles

def plot_moving_averages(df, ma1='wma_5', ma2='sma_5', rsi_plt='rsi_5', signal_column='signal_wma_5_sma_5', signal_column_2='signal_rsi_5', 
                         figsize=(40, 20), font_size=10, ma_markersize=50, signal_markersize_y=600, signal_markersize_b=500, v_if_short='o'): #'v' if short enabled, 'o' if not
    """
    Plots the minute_candles DataFrame with two selected moving averages and optional RSI.
    Also plots cumulative profit for MA and RSI strategies on a secondary axis.

    Parameters:
    - ma1 (str): The column name of the first moving average to plot.
    - ma2 (str): The column name of the second moving average to plot.
    - signal_column (str): The column name of the signal data (default is 'signal').
    - figsize (tuple): The size of the plot (width, height) in inches (default is (30, 20)).
    """
    try:
        # Clean the data to ensure numeric columns are valid
        columns_to_convert = ['open', 'high', 'low', 'close', 'volume', ma1, ma2, rsi_plt]
        df[columns_to_convert] = df[columns_to_convert].apply(pd.to_numeric, errors='coerce')

        # Select the columns to plot
        plot_data = df[['datetime', 'open', 'high', 'low', 'close', 'volume', ma1, ma2, rsi_plt]].copy()
        plot_data.set_index('datetime', inplace=True)

        # Create the additional plots for the moving averages and RSI, but only if they are warmed up
        add_plots = []

        # Check if the moving averages have enough valid data to plot
        if not df[ma1].isnull().all() and not df[ma2].isnull().all():
            add_plots.append(mpf.make_addplot(plot_data[ma1], color='yellow', type='scatter', marker='o', markersize=ma_markersize, label=f'{ma1}'))
            add_plots.append(mpf.make_addplot(plot_data[ma1], color='yellow', linestyle='-', width=0.75))
            add_plots.append(mpf.make_addplot(plot_data[ma2], color='purple', type='scatter', marker='o', markersize=ma_markersize, label=f'{ma2}'))
            add_plots.append(mpf.make_addplot(plot_data[ma2], color='purple', linestyle='-', width=0.75))
        else:
            print("Moving averages have not warmed up yet. Plotting without them.")

        # Check if the RSI has enough valid data to plot
        if not df[rsi_plt].isnull().all():
            add_plots.append(mpf.make_addplot(df[rsi_plt], panel=2, color='blue', type='scatter', marker='o', markersize=ma_markersize, label='RSI'))
            add_plots.append(mpf.make_addplot(df[rsi_plt], panel=2, color='blue', linestyle='-', width=0.75))
            add_plots.append(mpf.make_addplot(df['trend_indicator'], panel=2, color='white', type='scatter', marker='o', markersize=ma_markersize, label='RSI'))
            add_plots.append(mpf.make_addplot(df['trend_indicator'], panel=2, color='white', linestyle='-', width=0.75))
            add_plots.append(mpf.make_addplot(df['hundred_line'], panel=2, color='red', linestyle=':', secondary_y=False))
            add_plots.append(mpf.make_addplot(df['fifty_line'], panel=2, color='yellow', linestyle=':', secondary_y=False))
            add_plots.append(mpf.make_addplot(df['zero_line'], panel=2, color='green', linestyle=':', secondary_y=False))
            add_plots.append(mpf.make_addplot(df['trend_high_threshold'], panel=2, color='white', linestyle=':', secondary_y=False))
            add_plots.append(mpf.make_addplot(df['trend_low_threshold'], panel=2, color='white', linestyle=':', secondary_y=False))
        else:
            print("RSI has not warmed up yet. Plotting without it.")

        # Add buy, sell, and neutral markers if signal_column exists. Eliminate the if else statement to revert to working order
        if 'buy_marker' in df.columns and 'neutral_marker' in df.columns:
            add_plots.append(mpf.make_addplot(df['buy_marker'], type='scatter', marker='^', markersize=signal_markersize_y, color='yellow', panel=0, secondary_y=False))
            # add_plots.append(mpf.make_addplot(df['sell_marker'], type='scatter', marker=v_if_short, markersize=signal_markersize_y, color='yellow', panel=0, secondary_y=False))
            add_plots.append(mpf.make_addplot(df['neutral_marker'], type='scatter', marker='o', markersize=signal_markersize_y, color='yellow', panel=0, secondary_y=False))
        else:
            print("Buy/Sell markers for MA strat have not warmed up yet. Plotting without them.")

        # Add buy, sell, and neutral markers for RSI strategy
        if 'buy_marker_2' in df.columns and 'neutral_marker_2' in df.columns:
            add_plots.append(mpf.make_addplot(df['buy_marker_2'], type='scatter', marker='^', markersize=signal_markersize_b, color='blue', panel=0, secondary_y=False))
            # add_plots.append(mpf.make_addplot(df['sell_marker_2'], type='scatter', marker=v_if_short, markersize=signal_markersize_b, color='blue', panel=0, secondary_y=False))
            add_plots.append(mpf.make_addplot(df['neutral_marker_2'], type='scatter', marker='o', markersize=signal_markersize_b, color='blue', panel=0, secondary_y=False))
        else:
            print("Buy/Sell markers for RSI strat have not warmed up yet. Plotting without them.")

        # Add cumulative profit plots on a secondary y-axis
        add_plots.append(mpf.make_addplot(df['cum_pnl_ma'], panel=0, color='yellow', secondary_y=True, label='Cumulative PnL (MA)', linestyle='-', width=1.25))
        add_plots.append(mpf.make_addplot(df['cum_pnl_rsi'], panel=0, color='blue', secondary_y=True, label='Cumulative PnL (RSI)', linestyle='-', width=1.25))
        add_plots.append(mpf.make_addplot(df['cum_pnl_all'], panel=0, color='green', secondary_y=True, label='Cumulative PnL (Combined)', linestyle='-', width=1.25))

        # Add stop-loss markers (x) for both MA and RSI strategies
        # if 'stop_loss_ma' in df.columns:
        add_plots.append(mpf.make_addplot(df['stop_loss_ma'], type='scatter', marker='x', markersize=200, color='yellow', panel=0, secondary_y=False))
        # else:
        #     print("There are no stop loss markers for MA strat")
        # if 'stop_loss_rsi' in df.columns:
        add_plots.append(mpf.make_addplot(df['stop_loss_rsi'], type='scatter', marker='x', markersize=100, color='blue', panel=0, secondary_y=False))
        # else:
        #     print("There are no stop loss markers for RSI strat")

        # Add price action envelope as white lines
        if 'price_action_upper' in df.columns and 'price_action_lower' in df.columns:
            add_plots.append(mpf.make_addplot(df['price_action_upper'], color='white', linestyle='-', width=0.5, label='Price Action Upper'))
            add_plots.append(mpf.make_addplot(df['price_action_lower'], color='white', linestyle='-', width=0.5, label='Price Action Lower'))
            # add_plots.append(mpf.make_addplot(df['ma_price_action_upper'], color='white', linestyle='-', width=0.5, label='Price Action Upper'))
            # add_plots.append(mpf.make_addplot(df['ma_price_action_lower'], color='white', linestyle='-', width=0.5, label='Price Action Lower'))
        else:
            print("Price action envelope not calculating properly")

        # Create a custom style with a black background
        black_style = mpf.make_mpf_style(
            base_mpf_style='charles',  # Start with the 'charles' style and modify it
            facecolor='black',         # Set the background color to black
            gridcolor='black',          # Set the grid line color
            edgecolor='purple',          # Set the edge color for candles and boxes
            figcolor='black',          # Set the figure background color to black
            rc={'axes.labelcolor': 'yellow', 
                'xtick.color': 'yellow', 
                'ytick.color': 'yellow', 
                'axes.titlecolor': 'yellow',
                'font.size': font_size, 
                'axes.labelsize': font_size,
                'axes.titlesize': font_size,
                'xtick.labelsize': font_size,
                'ytick.labelsize': font_size,
                'legend.fontsize': font_size}  # Set tick and label colors to white
        )

        # Plot using mplfinance
        mpf.plot(plot_data, type='candle', style=black_style, 
                title='',
                ylabel='Price', 
                addplot=add_plots, 
                figsize=figsize,
                volume=True,
                panel_ratios=(8, 2),
                #  panel_ratios=(8, 2, 2),             
                tight_layout=True)
    except Exception as e:
        print(f"Something wrong in the plotting_moving_averages function: {e}")

def visualize_trades(candles, lower_slice=0, upper_slice=-1):
    # Iterate through the minute_candles dictionary
    for ticker, minute_candles_df in candles.items():
        # Create a copy of the minute_candles DataFrame
        minute_candles_viz = minute_candles_df[lower_slice:upper_slice].copy()
        # Calculate profit/loss for each DataFrame
        # calculate_profit_loss(minute_candles_viz, signal_ma=signal_ma, signal_rsi=signal_rsi, multiplier=1)
        try:
            # Plot moving averages
            plot_moving_averages(minute_candles_viz, sig_ma, con_ma, rsi_plt, signal_column='signal_wma_5_sma_5',
                                signal_column_2='signal_rsi_5', figsize=(40, 20), font_size=20,
                                ma_markersize=50, signal_markersize_y=450, signal_markersize_b=300, v_if_short='o')
            # Print out the number of trades and total PnL
            print(f'{ticker} : {len(minute_candles_df)} minutes : {minute_candles_viz["buy_marker"].notna().sum()} ma trades : {minute_candles_viz["buy_marker_2"].notna().sum()} rsi trades : {minute_candles_viz["cum_pnl_all"].iloc[-1]} total pnl')
        except Exception as e:
            # Handle any errors that occur during the plotting
            print(f"Something wrong in the visualize_trades function: {e}")

def handle_data(message):
    global minute_candles

    # Define the periods for SMAs and WMAs
    ma_periods=[5, 10, 20]
    sma_periods = ma_periods
    wma_periods = ma_periods
    rsi_periods = ma_periods
    current_time_USE = datetime.now(pytz.timezone('US/Eastern')).strftime('%Y-%m-%d %H:%M:%S')
    current_hour = datetime.now(pytz.timezone('US/Eastern')).hour
    current_minute = datetime.now(pytz.timezone('US/Eastern')).minute
    current_second = datetime.now(pytz.timezone('US/Eastern')).second

    # Define the list of (sig_ma, con_ma, rsi_col) combinations
    ma_combinations = [
        ('wma_5', 'sma_5', 'rsi_5'),
        ('wma_10', 'sma_10', 'rsi_10'),
        ('wma_20', 'sma_20', 'rsi_20')
    ]

    # Try parsing the incoming message
    try: # Indent the code and uncomment the try-except block to revert
        data = json.loads(message)

        # Call the function to process the data and update todays_price_action
        process_message_data(data)

        # **Update the minute candles in real-time**
        update_minute_candles(ticker_tables, time_frame='min')

        # Iterate through all the tickers in ticker_tables
        for ticker, df in minute_candles.items():
            # Calculate moving averages and indicators for each DataFrame
            minute_candles[ticker] = calculate_indicators(
                df, 
                price_col='close', 
                acc_vol_col='accumulative_volume', 
                sma_periods=sma_periods,
                wma_periods=wma_periods,
                rsi_periods=rsi_periods,
                candle_window=10
            )
        
        # Iterate through all the tickers in minute_candles dictionary
        for ticker, df in minute_candles.items():
            # Iterate over the (sig_ma, con_ma, rsi_col) combinations for each ticker
            for sig_ma, con_ma, rsi_col in ma_combinations:                                    # CHANGE THE SIMULATED STRATEGY HERE
                # Call the check_moving_average function for each combination
                check_moving_average(df,
                                    #  ticker=ticker, # comment this out if not needed or causing problems
                                    #  ticker_to_tick_size=ticker_to_tick_size, # comment this out if not needed or causing problems
                                     ma_name1=sig_ma, 
                                     ma_name2=con_ma, 
                                     rsi_column=rsi_col
                                     ) # CHANGE THE SIMULATED STRATEGY HERE

        # Calculate profit/loss for each ticker's DataFrame
        for ticker, df in minute_candles.items():                                                                                       # CHANGE THE SIMULATED STRATEGY HERE
            minute_candles[ticker] = calculate_profit_loss(df, signal_ma='signal_wma_5_sma_5', signal_rsi='signal_rsi_5', multiplier=1) # CHANGE THE SIMULATED STRATEGY HERE

        # if current_second == 59 or 00 or 1 or 2:
        #     # Call the trade_logic function to execute trades based on the signal
        #     trade_logic(minute_candles, buy_order=buy_order, sell_order=sell_order)

        # if current_second == 59 or 00 or 1 or 2:
        #     # Call the trade_simulator function to simulate trades based on the signal
        #     trade_simulator(minute_candles)

            # print(message)

    except Exception as e:
        print(f"{current_time_USE}: Error processing message: {e}")

key = 0

tickers_list = list(ticker_tables.keys())
display(minute_candles)
display(tickers_list)
display(ticker_tables[tickers_list[key]])
display(minute_candles[tickers_list[key]])

sig_ma = 'wma_5'
con_ma = 'sma_5'
rsi_plt = 'rsi_5'
signal_ma='signal_wma_5_sma_5'
signal_rsi='signal_rsi_5'

visualize_trades(minute_candles, lower_slice=0, upper_slice=-1)

# # Initialize a variable to hold the task reference
# visualization_task = None  # This ensures visualization_task is defined

# async def update_visualization():
#     while True:
#         # Clear the previous output
#         clear_output(wait=True)
       
#         # Call the visualization function
#         visualize_trades(lower_slice=0, upper_slice=-1)
        
#         # Wait for 5 seconds before repeating
#         await asyncio.sleep(60)

# def start_visualization():
#     global visualization_task
#     # Start the update_visualization function if it's not already running
#     if visualization_task is None:
#         loop = asyncio.get_event_loop()
#         visualization_task = loop.create_task(update_visualization())

# def stop_visualization():
#     global visualization_task
#     # Stop the update_visualization function if it's running
#     if visualization_task is not None:
#         visualization_task.cancel()  # Cancel the task
#         visualization_task = None

# # Start the visualization
# start_visualization()

def print_all_pnls(candles):
    for ticker, minute_candles_df in candles.items():
        total_pnl = round(minute_candles_df["cum_pnl_all"].iloc[-1], 3)
        close_price_diff = round(minute_candles_df["close"].iloc[-1] - minute_candles_df["close"].iloc[0], 3)
        point_alpha = round(total_pnl - close_price_diff, 3)
        print(f'{ticker} : {total_pnl} total pnl, Close Price Difference: {close_price_diff}, Point Alpha: {point_alpha}')

# Call print_all_pnls() to print all profit figures and close price differences
print_all_pnls(minute_candles)

def percentage_candles_containing_previous_close(candles):
    """
    Calculates the percentage of candles where the current candle's high and low
    contain the previous candle's close price.

    Parameters:
    - candles (pd.DataFrame): A DataFrame containing 'high', 'low', and 'close' columns.

    Returns:
    - float: The percentage of candles containing the previous candle's close price.
    """
    # Ensure the DataFrame contains necessary columns
    if not {'high', 'low', 'close'}.issubset(candles.columns):
        raise ValueError("The DataFrame must contain 'high', 'low', and 'close' columns.")

    # Shift the close price to align it with the next candle
    previous_close = candles['close'].shift(1)

    # Check if the previous close is within the current candle's range
    contains_previous_close = (candles['low'] <= previous_close) & (candles['high'] >= previous_close)

    # Calculate percentage
    percentage = contains_previous_close.mean() * 100

    return round(percentage, 2)

# Calculate the percentage for the /ES DataFrame
percentage_candles_containing_previous_close(minute_candles['/ES'])

###############################################################################################################################
















def handle_data(message):
    global minute_candles

    # Define the periods for SMAs and WMAs
    ma_periods=[5, 10, 20]
    sma_periods = ma_periods
    wma_periods = ma_periods
    rsi_periods = ma_periods
    current_time_USE = datetime.now(pytz.timezone('US/Eastern')).strftime('%Y-%m-%d %H:%M:%S')
    current_hour = datetime.now(pytz.timezone('US/Eastern')).hour
    current_minute = datetime.now(pytz.timezone('US/Eastern')).minute
    current_second = datetime.now(pytz.timezone('US/Eastern')).second

    # Dynamically create ma_combinations
    ma_combinations = [
        (f'wma_{period}', f'sma_{period}', f'rsi_{period}') for period in ma_periods
    ]

    # Try parsing the incoming message
    try: # Indent the code and uncomment the try-except block to revert
        data = json.loads(message)

        # Call the function to process the data and update todays_price_action
        process_message_data(data)

        # **Update the minute candles in real-time**
        update_minute_candles(ticker_tables, time_frame='min')

        # Iterate through all the tickers in ticker_tables
        for ticker, df in minute_candles.items():
            # Calculate moving averages and indicators for each DataFrame
            minute_candles[ticker] = calculate_indicators(
                df, 
                price_col='close', 
                acc_vol_col='accumulative_volume', 
                sma_periods=sma_periods,
                wma_periods=wma_periods,
                rsi_periods=rsi_periods,
                candle_window=10
            )
        
        # Iterate through all the tickers in the current dictionary
        for ticker, df in minute_candles.items():
            print(f"Processing ticker: {ticker} in {compression_name}")
            
            # Iterate through all the ma_combinations
            for sig_ma, con_ma, rsi_col in ma_combinations:
                # Generate trading signals
                minute_candles[ticker] = generate_trading_signals(
                    df,
                    ma_name1=sig_ma,
                    ma_name2=con_ma,
                    rsi_column=rsi_col
                )

                # Update position_open columns to be 1:1 verbal boolean with the signal
                minute_candles[ticker] = update_position_open(
                    df,
                    ma_name1=sig_ma,
                    ma_name2=con_ma,
                    rsi_column=rsi_col
                )

                # Determine entry prices for each ticker
                minute_candles[ticker] = determine_entry_prices(
                    df,
                    ma_name1=sig_ma,
                    ma_name2=con_ma,
                    rsi_column=rsi_col,
                    ticker_to_tick_size=ticker_to_tick_size,
                    ticker=ticker
                )

                # Determine exit prices for each ticker
                minute_candles[ticker] = determine_exit_prices(
                    df,
                    ma_name1=sig_ma,
                    ma_name2=con_ma,
                    rsi_column=rsi_col,
                    ticker_to_tick_size=ticker_to_tick_size,
                    ticker=ticker
                )

                # Stop loss calculation
                minute_candles[ticker] = calculate_stop_losses(
                    df,
                    ma_name1=sig_ma,
                    ma_name2=con_ma,
                    rsi_column=rsi_col
                )

                # Track stop loss hits
                minute_candles[ticker] = track_stop_loss_hits(
                    df,
                    ma_name1=sig_ma,
                    ma_name2=con_ma,
                    rsi_column=rsi_col,
                    ticker_to_tick_size=ticker_to_tick_size,
                    ticker=ticker
                )

                # Adjust signals from stop loss hits
                minute_candles[ticker] = adjust_signals_for_stop_loss(
                    df,
                    ma_name1=sig_ma,
                    ma_name2=con_ma,
                    rsi_column=rsi_col
                )

                # Re-update position_open column after stop loss hits
                minute_candles[ticker] = update_position_open(
                    df,
                    ma_name1=sig_ma,
                    ma_name2=con_ma,
                    rsi_column=rsi_col
                )

                # Re-determine entry prices after stop loss hits
                minute_candles[ticker] = determine_entry_prices(
                    df,
                    ma_name1=sig_ma,
                    ma_name2=con_ma,
                    rsi_column=rsi_col,
                    ticker_to_tick_size=ticker_to_tick_size,
                    ticker=ticker
                )

                # Re-determine exit prices after stop loss hits
                minute_candles[ticker] = determine_exit_prices(
                    df,
                    ma_name1=sig_ma,
                    ma_name2=con_ma,
                    rsi_column=rsi_col,
                    ticker_to_tick_size=ticker_to_tick_size,
                    ticker=ticker
                )

                # Update stop loss levels after stop loss hits
                minute_candles[ticker] = update_stop_loss(
                    df,
                    ma_name1=sig_ma,
                    ma_name2=con_ma,
                    rsi_column=rsi_col
                )

                # Calculate profit/loss for each ticker's DataFrame
                minute_candles[ticker] = calculate_profit_loss_1(
                    df,
                    contract_multiplier=1,
                    ma_name1=sig_ma,
                    ma_name2=con_ma,
                    rsi_column=rsi_col
                )

        # if current_second == 59 or 00 or 1 or 2:
        #     # Call the trade_logic function to execute trades based on the signal
        #     trade_logic(minute_candles, buy_order=buy_order, sell_order=sell_order)

        # if current_second == 59 or 00 or 1 or 2:
        #     # Call the trade_simulator function to simulate trades based on the signal
        #     trade_simulator(minute_candles)

            # print(message)

    except Exception as e:
        print(f"{current_time_USE}: Error processing message: {e}")





window_size = 5

sig_ma = f'wma_{window_size}'
con_ma = f'sma_{window_size}'
rsi_col = f'rsi_{window_size}'

visualize_trades_1(
    candles=compressed_candles[f'minute_candles_{examination_compression_factor}'], 
    ticker_to_tick_size=ticker_to_tick_size,
    ticker_to_point_value=ticker_to_point_value,    
    ma_name1=sig_ma, 
    ma_name2=con_ma, 
    rsi_column=rsi_col, 
    lower_slice=0, 
    upper_slice=-1,
    compression_factor=examination_compression_factor
)



# Call the updated print_all_pnls function
print_all_pnls(
    candles=ticker_dict,  # Pass the dictionary of DataFrames
    compression_factor=compression_factor,  # Pass the extracted compression factor
    ticker_to_tick_size=ticker_to_tick_size,  # Pass tick size mapping
    ticker_to_point_value=ticker_to_point_value,  # Pass point value mapping
    ma_name1='wma_5',
    ma_name2='sma_5',
    rsi_column='rsi_5'
)






###############################################################################################################################
# Trade logic and implementation for the version that handles the whole dataset before iterating through trend and 
# reversion trading rules

def generate_trading_signals_1(candles, ma_name1='wma_5', ma_name2='sma_5', rsi_column='rsi_5'): # signals are reversed temporarily
    """ Vectorized without iterrows
    Generate buy/sell signals based on moving averages and RSI indicators and save position states.

    Parameters:
    - candles (pd.DataFrame): The DataFrame containing candle data.
    - ma_name1 (str): Column name for the first moving average.
    - ma_name2 (str): Column name for the second moving average.
    - rsi_column (str): Column name for the RSI indicator.

    Returns:
    - candles (pd.DataFrame): The DataFrame with updated signal and position state columns.
    """
    # Dynamically generate signal column names
    ma_signal_column = f'signal_{ma_name1}_{ma_name2}'
    rsi_signal_column = f'signal_{rsi_column}'
    ma_position_open_col = f'position_open_{ma_name1}_{ma_name2}'
    rsi_position_open_col = f'position_open_{rsi_column}'

    # Initialize signal and position state columns
    candles[ma_signal_column] = 0
    candles[rsi_signal_column] = 0
    candles[ma_position_open_col] = False
    candles[rsi_position_open_col] = False

    # Moving average signal and position state generation
    ma_position_open = False  # Local variable for tracking state
    ma_signals = []
    ma_positions = []

    for ma1, ma2 in zip(candles[ma_name1], candles[ma_name2]):
        if not ma_position_open and ma1 <= ma2:
            ma_position_open = True
            ma_signals.append(0) # here
        elif ma_position_open and ma1 > ma2:
            ma_position_open = False
            ma_signals.append(1) # here
        else:
            ma_signals.append(ma_signals[-1] if ma_signals else 0)
        ma_positions.append(ma_position_open)
    candles[ma_signal_column] = ma_signals
    candles[ma_position_open_col] = ma_positions

    # RSI signal and position state generation
    rsi_position_open = False  # Local variable for tracking state
    rsi_signals = []
    rsi_positions = []
    
    for rsi in candles[rsi_column]:
        if not rsi_position_open and rsi < 50:
            rsi_position_open = True
            rsi_signals.append(0) # here
        elif rsi_position_open and rsi >= 50:
            rsi_position_open = False
            rsi_signals.append(1) # here
        else:
            rsi_signals.append(rsi_signals[-1] if rsi_signals else 0)
        rsi_positions.append(rsi_position_open)
    candles[rsi_signal_column] = rsi_signals
    candles[rsi_position_open_col] = rsi_positions

    return candles

def update_position_open_1(candles, ma_name1='wma_5', ma_name2='sma_5', rsi_column='rsi_5'):
    """
    Update the 'ma_position_open' and 'rsi_position_open' columns for a given DataFrame.

    Parameters:
    - candles (pd.DataFrame): The DataFrame containing the signals and position columns.
    - ma_name1 (str): Column name for the first moving average.
    - ma_name2 (str): Column name for the second moving average.
    - rsi_column (str): Column name for the RSI signal.

    Returns:
    - pd.DataFrame: The updated DataFrame with 'ma_position_open' and 'rsi_position_open' columns.
    """
    # Dynamically generate signal column names
    ma_signal_column = f'signal_{ma_name1}_{ma_name2}'
    rsi_signal_column = f'signal_{rsi_column}'
    ma_position_open = f'position_open_{ma_name1}_{ma_name2}'
    rsi_position_open = f'position_open_{rsi_column}'
    
    # Update position open columns
    candles[ma_position_open] = candles[ma_signal_column] == 1
    candles[rsi_position_open] = candles[rsi_signal_column] == 1
    
    return candles

def determine_entry_prices_1(candles, ma_name1='wma_5', ma_name2='sma_5', rsi_column='rsi_5', ticker_to_tick_size=None, ticker=None):
    """
    Determine entry prices for MA and RSI strategies based on signals.

    Parameters:
    - candles (pd.DataFrame): The DataFrame containing candle data.
    - ma_name1 (str): Column name for the first moving average.
    - ma_name2 (str): Column name for the second moving average.
    - rsi_column (str): Column name for the RSI indicator.
    - ticker_to_tick_size (dict): Mapping of tickers to their tick sizes.
    - ticker (str): The ticker for which the tick size applies.

    Returns:
    - candles (pd.DataFrame): The DataFrame with updated entry price columns.
    """
    # Dynamically generate column names
    ma_signal_column = f'signal_{ma_name1}_{ma_name2}'
    rsi_signal_column = f'signal_{rsi_column}'
    ma_entry_price = f'entry_price_{ma_name1}_{ma_name2}'
    rsi_entry_price = f'entry_price_{rsi_column}'

    # Initialize entry price columns
    candles[ma_entry_price] = None
    candles[rsi_entry_price] = None

    # Get tick size
    tick_size = ticker_to_tick_size.get(ticker, 0) if ticker_to_tick_size else 0

    # Moving Average Strategy
    ma_signals = candles[ma_signal_column]
    ma_close_prices = candles['close']
    ma_entry_mask = (ma_signals == 1) & (ma_signals.shift(1) != 1)
    candles.loc[ma_entry_mask, ma_entry_price] = ma_close_prices[ma_entry_mask] + tick_size

    # RSI Strategy
    rsi_signals = candles[rsi_signal_column]
    rsi_entry_mask = (rsi_signals == 1) & (rsi_signals.shift(1) != 1)
    candles.loc[rsi_entry_mask, rsi_entry_price] = ma_close_prices[rsi_entry_mask] + tick_size

    return candles

def determine_exit_prices_1(candles, ma_name1='wma_5', ma_name2='sma_5', rsi_column='rsi_5', ticker_to_tick_size=None, ticker=None):
    """
    Determine exit prices for MA and RSI strategies based on signals.

    Parameters:
    - candles (pd.DataFrame): The DataFrame containing candle data.
    - ma_name1 (str): Column name for the first moving average.
    - ma_name2 (str): Column name for the second moving average.
    - rsi_column (str): Column name for the RSI indicator.
    - ticker_to_tick_size (dict): Mapping of tickers to their tick sizes.
    - ticker (str): The ticker for which the tick size applies.

    Returns:
    - candles (pd.DataFrame): The DataFrame with updated exit price columns.
    """
    # Dynamically generate column names
    ma_signal_column = f'signal_{ma_name1}_{ma_name2}'
    rsi_signal_column = f'signal_{rsi_column}'
    ma_exit_price = f'exit_price_{ma_name1}_{ma_name2}'
    rsi_exit_price = f'exit_price_{rsi_column}'

    # Initialize exit price columns
    candles[ma_exit_price] = None
    candles[rsi_exit_price] = None

    # Get tick size
    tick_size = ticker_to_tick_size.get(ticker, 0) if ticker_to_tick_size else 0

    # Moving Average Strategy
    ma_signals = candles[ma_signal_column]
    ma_close_prices = candles['close']
    ma_exit_mask = (ma_signals == 0) & (ma_signals.shift(1) == 1)
    candles.loc[ma_exit_mask, ma_exit_price] = ma_close_prices[ma_exit_mask] - tick_size

    # RSI Strategy
    rsi_signals = candles[rsi_signal_column]
    rsi_exit_mask = (rsi_signals == 0) & (rsi_signals.shift(1) == 1)
    candles.loc[rsi_exit_mask, rsi_exit_price] = ma_close_prices[rsi_exit_mask] - tick_size

    return candles

def calculate_stop_losses_1(candles, ma_name1='wma_5', ma_name2='sma_5', rsi_column='rsi_5'):
    """
    Dynamically calculate stop loss levels for MA and RSI strategies and ensure they persist while positions are open.

    Parameters:
    - candles (pd.DataFrame): The DataFrame containing candle data.
    - ma_name1 (str): Column name for the first moving average.
    - ma_name2 (str): Column name for the second moving average.
    - rsi_column (str): Column name for the RSI indicator.

    Returns:
    - pd.DataFrame: The updated DataFrame with dynamically named stop loss columns.
    """
    # Dynamically generate column names
    ma_entry_price = f'entry_price_{ma_name1}_{ma_name2}'
    ma_exit_price = f'exit_price_{ma_name1}_{ma_name2}'
    rsi_entry_price = f'entry_price_{rsi_column}'
    rsi_exit_price = f'exit_price_{rsi_column}'
    stop_loss_ma = f'stop_loss_{ma_name1}_{ma_name2}'
    stop_loss_rsi = f'stop_loss_{rsi_column}'

    # Initialize stop loss columns
    candles[stop_loss_ma] = None
    candles[stop_loss_rsi] = None

    # Moving Average Stop Loss
    ma_entry_mask = candles[ma_entry_price].notnull()
    ma_exit_mask = candles[ma_exit_price].notnull()
    
    # Set stop loss where positions open
    candles.loc[ma_entry_mask, stop_loss_ma] = candles[ma_entry_price] - candles['candle_span_max']

    # Reset stop loss and close position where positions close
    candles.loc[ma_exit_mask, stop_loss_ma] = None

    # RSI Stop Loss
    rsi_entry_mask = candles[rsi_entry_price].notnull()
    rsi_exit_mask = candles[rsi_exit_price].notnull()
    
    # Set stop loss where positions open
    candles.loc[rsi_entry_mask, stop_loss_rsi] = candles[rsi_entry_price] - candles['candle_span_max']

    # Reset stop loss and close position where positions close
    candles.loc[rsi_exit_mask, stop_loss_rsi] = None

    # Forward-fill stop loss for both strategies
    candles[stop_loss_ma] = candles[stop_loss_ma].ffill()
    candles[stop_loss_rsi] = candles[stop_loss_rsi].ffill()

    return candles

def track_stop_loss_hits_1(candles, ma_name1='wma_5', ma_name2='sma_5', rsi_column='rsi_5', ticker_to_tick_size=None, ticker=None): # Tick size is not being used, address this here and in the function calling
    """
    Track whether stop losses have been hit for MA and RSI strategies and update dynamically named columns.

    Parameters:
    - candles (pd.DataFrame): The DataFrame containing candle data.
    - ma_name1 (str): Column name for the first moving average.
    - ma_name2 (str): Column name for the second moving average.
    - rsi_column (str): Column name for the RSI indicator.
    - ticker_to_tick_size (dict): Mapping of tickers to their tick sizes.
    - ticker (str): The ticker for which the tick size applies.

    Returns:
    - pd.DataFrame: The updated DataFrame with dynamically named stop loss hit flags.
    """
    # Dynamically generate column names
    stop_loss_ma = f'stop_loss_{ma_name1}_{ma_name2}'
    stop_loss_rsi = f'stop_loss_{rsi_column}'
    ma_stop_loss_hit = f'stop_loss_hit_{ma_name1}_{ma_name2}'
    rsi_stop_loss_hit = f'stop_loss_hit_{rsi_column}'
    ma_position_open = f'position_open_{ma_name1}_{ma_name2}'
    rsi_position_open = f'position_open_{rsi_column}'
    
    # Initialize stop loss hit columns
    candles[ma_stop_loss_hit] = False
    candles[rsi_stop_loss_hit] = False

    # Get tick size
    tick_size = ticker_to_tick_size.get(ticker, 0) if ticker_to_tick_size else 0

    # Ensure stop loss values are numerical (convert None to NaN)
    candles[stop_loss_ma] = candles[stop_loss_ma].fillna(float('inf'))
    candles[stop_loss_rsi] = candles[stop_loss_rsi].fillna(float('inf'))

    # Moving Average Stop Loss Hit Logic
    ma_hit_condition = (candles[stop_loss_ma].notnull()) & (candles['close'] <= candles[stop_loss_ma]) & candles[ma_position_open]
    candles.loc[ma_hit_condition, ma_stop_loss_hit] = True

    # RSI Stop Loss Hit Logic
    rsi_hit_condition = (candles[stop_loss_rsi].notnull()) & (candles['close'] <= candles[stop_loss_rsi]) & candles[rsi_position_open]
    candles.loc[rsi_hit_condition, rsi_stop_loss_hit] = True

    return candles

def adjust_signals_for_stop_loss_1(candles, ma_name1='wma_5', ma_name2='sma_5', rsi_column='rsi_5'):
    """
    Adjust MA and RSI signals to 0 where stop loss has been hit.

    Parameters:
    - candles (pd.DataFrame): The DataFrame containing candle data.
    - ma_name1 (str): Column name for the first moving average.
    - ma_name2 (str): Column name for the second moving average.
    - rsi_column (str): Column name for the RSI indicator.

    Returns:
    - pd.DataFrame: The updated DataFrame with adjusted signals.
    """
    # Dynamically generate column names
    ma_signal_column = f'signal_{ma_name1}_{ma_name2}'
    rsi_signal_column = f'signal_{rsi_column}'
    ma_stop_loss_hit_column = f'stop_loss_hit_{ma_name1}_{ma_name2}'
    rsi_stop_loss_hit_column = f'stop_loss_hit_{rsi_column}'

    # Adjust MA and RSI signals where stop loss has been hit
    candles.loc[candles[ma_stop_loss_hit_column], ma_signal_column] = 0
    candles.loc[candles[rsi_stop_loss_hit_column], rsi_signal_column] = 0

    return candles

def update_stop_loss_1(candles, ma_name1='wma_5', ma_name2='sma_5', rsi_column='rsi_5'):
    """
    Dynamically set stop loss columns to NaN where corresponding signal columns are 0.

    Parameters:
    - candles (pd.DataFrame): The DataFrame containing candle data.
    - ma_name1 (str): Column name for the first moving average.
    - ma_name2 (str): Column name for the second moving average.
    - rsi_column (str): Column name for the RSI indicator.

    Returns:
    - pd.DataFrame: The updated DataFrame with dynamically named stop loss columns modified.
    """
    # Dynamically generate column names
    ma_signal_column = f'signal_{ma_name1}_{ma_name2}'
    rsi_signal_column = f'signal_{rsi_column}'
    stop_loss_ma = f'stop_loss_{ma_name1}_{ma_name2}'
    stop_loss_rsi = f'stop_loss_{rsi_column}'

    # Update stop loss columns to NaN where signals are 0
    candles.loc[candles[ma_signal_column] == 0, stop_loss_ma] = float('nan')
    candles.loc[candles[rsi_signal_column] == 0, stop_loss_rsi] = float('nan')

    return candles

def calculate_profit_loss_1(candles, ma_name1='wma_5', ma_name2='sma_5', rsi_column='rsi_5', contract_multiplier=1, trade_commission=1.5):
    """ 
    Dynamically calculate profit and loss based on entry and exit price columns, including cumulative commission costs.

    Parameters:
    - candles (pd.DataFrame): The DataFrame containing candle data with entry and exit price columns.
    - multiplier (float): The multiplier for PnL calculation (e.g., contract size).
    - trade_commission (float): The commission cost per trade.
    - ma_name1 (str): Column name for the first moving average.
    - ma_name2 (str): Column name for the second moving average.
    - rsi_column (str): Column name for the RSI indicator.

    Returns:
    - pd.DataFrame: The DataFrame with dynamically named profit/loss and commission cost columns.
    """
    # Dynamically generate column names
    ma_entry_price = f'entry_price_{ma_name1}_{ma_name2}'
    ma_exit_price = f'exit_price_{ma_name1}_{ma_name2}'
    rsi_entry_price = f'entry_price_{rsi_column}'
    rsi_exit_price = f'exit_price_{rsi_column}'
    pnl_ma_col = f'pnl_{ma_name1}_{ma_name2}'
    pnl_rsi_col = f'pnl_{rsi_column}'
    cum_pnl_ma_col = f'cum_{pnl_ma_col}'
    cum_pnl_rsi_col = f'cum_{pnl_rsi_col}'
    cum_pnl_all_col = f'cum_pnl_all_{ma_name1}_{ma_name2}_{rsi_column}'
    ma_commission_col = f'commission_cost_{ma_name1}_{ma_name2}'
    rsi_commission_col = f'commission_cost_{rsi_column}'

    # Initialize PnL and commission columns
    candles[pnl_ma_col] = 0.0
    candles[pnl_rsi_col] = 0.0
    candles[ma_commission_col] = 0.0
    candles[rsi_commission_col] = 0.0

    # Moving Average Strategy PnL and Commission Calculation
    ma_entry_indices = candles.index[candles[ma_entry_price].notnull()]
    ma_exit_indices = candles.index[candles[ma_exit_price].notnull()]

    # Pair up entry and exit prices
    valid_pairs_ma = min(len(ma_entry_indices), len(ma_exit_indices))
    ma_entry_prices = candles.loc[ma_entry_indices[:valid_pairs_ma], ma_entry_price].values
    ma_exit_prices = candles.loc[ma_exit_indices[:valid_pairs_ma], ma_exit_price].values

    # Calculate commission costs for MA strategy
    candles[ma_commission_col] = candles[ma_entry_price].notna().astype(int) * trade_commission + \
                                 candles[ma_exit_price].notna().astype(int) * trade_commission
    candles[ma_commission_col] = candles[ma_commission_col].cumsum()  # Accumulate commission costs

    # Calculate PnL for MA strategy
    ma_pnl = (ma_exit_prices - ma_entry_prices) * contract_multiplier
    candles.loc[ma_exit_indices[:valid_pairs_ma], pnl_ma_col] = ma_pnl

    # RSI Strategy PnL and Commission Calculation
    rsi_entry_indices = candles.index[candles[rsi_entry_price].notnull()]
    rsi_exit_indices = candles.index[candles[rsi_exit_price].notnull()]

    # Pair up entry and exit prices
    valid_pairs_rsi = min(len(rsi_entry_indices), len(rsi_exit_indices))
    rsi_entry_prices = candles.loc[rsi_entry_indices[:valid_pairs_rsi], rsi_entry_price].values
    rsi_exit_prices = candles.loc[rsi_exit_indices[:valid_pairs_rsi], rsi_exit_price].values

    # Calculate commission costs for RSI strategy
    candles[rsi_commission_col] = candles[rsi_entry_price].notna().astype(int) * trade_commission + \
                                  candles[rsi_exit_price].notna().astype(int) * trade_commission
    candles[rsi_commission_col] = candles[rsi_commission_col].cumsum()  # Accumulate commission costs

    # Calculate PnL for RSI strategy
    rsi_pnl = (rsi_exit_prices - rsi_entry_prices) * contract_multiplier
    candles.loc[rsi_exit_indices[:valid_pairs_rsi], pnl_rsi_col] = rsi_pnl

    # Calculate cumulative PnL for both strategies
    candles[cum_pnl_ma_col] = candles[pnl_ma_col].cumsum()
    candles[cum_pnl_rsi_col] = candles[pnl_rsi_col].cumsum()

    # Calculate combined cumulative PnL
    candles[cum_pnl_all_col] = candles[cum_pnl_ma_col] + candles[cum_pnl_rsi_col]

    return candles

# Iterate through all tickers in compressed_sessions
for ticker, sessions in compressed_sessions.items():
    print(f"Processing ticker: {ticker}")

    # Iterate through all sessions of the current ticker
    for time_slice, compressions in sessions.items():
        print(f"  Processing session: {time_slice}, Total Compressions: {len(compressions)}")

        # Iterate through all compression levels of the current session
        for compression_name, df in compressions.items():
            print(f"    Processing compression: {compression_name} in session {time_slice}")

            # Iterate through all the ma_combinations
            for sig_ma, con_ma, rsi_col in ma_combinations:
                # Generate trading signals
                compressions[compression_name] = generate_trading_signals_1(
                    df,
                    ma_name1=sig_ma,
                    ma_name2=con_ma,
                    rsi_column=rsi_col
                )

                # Update position_open columns to be 1:1 verbal boolean with the signal
                compressions[compression_name] = update_position_open_1(
                    df,
                    ma_name1=sig_ma,
                    ma_name2=con_ma,
                    rsi_column=rsi_col
                )

                # Determine entry prices for each ticker
                compressions[compression_name] = determine_entry_prices_1(
                    df,
                    ma_name1=sig_ma,
                    ma_name2=con_ma,
                    rsi_column=rsi_col,
                    ticker_to_tick_size=ticker_to_tick_size,
                    ticker=ticker
                )

                # Determine exit prices for each ticker
                compressions[compression_name] = determine_exit_prices_1(
                    df,
                    ma_name1=sig_ma,
                    ma_name2=con_ma,
                    rsi_column=rsi_col,
                    ticker_to_tick_size=ticker_to_tick_size,
                    ticker=ticker
                )

                # Stop loss calculation
                compressions[compression_name] = calculate_stop_losses_1(
                    df,
                    ma_name1=sig_ma,
                    ma_name2=con_ma,
                    rsi_column=rsi_col
                )

                # Track stop loss hits
                compressions[compression_name] = track_stop_loss_hits_1(
                    df,
                    ma_name1=sig_ma,
                    ma_name2=con_ma,
                    rsi_column=rsi_col,
                    ticker_to_tick_size=ticker_to_tick_size,
                    ticker=ticker
                )

                # Adjust signals from stop loss hits
                compressions[compression_name] = adjust_signals_for_stop_loss_1(
                    df,
                    ma_name1=sig_ma,
                    ma_name2=con_ma,
                    rsi_column=rsi_col
                )

                # Re-update position_open column after stop loss hits
                compressions[compression_name] = update_position_open_1(
                    df,
                    ma_name1=sig_ma,
                    ma_name2=con_ma,
                    rsi_column=rsi_col
                )

                # Re-determine entry prices after stop loss hits
                compressions[compression_name] = determine_entry_prices_1(
                    df,
                    ma_name1=sig_ma,
                    ma_name2=con_ma,
                    rsi_column=rsi_col,
                    ticker_to_tick_size=ticker_to_tick_size,
                    ticker=ticker
                )

                # Re-determine exit prices after stop loss hits
                compressions[compression_name] = determine_exit_prices_1(
                    df,
                    ma_name1=sig_ma,
                    ma_name2=con_ma,
                    rsi_column=rsi_col,
                    ticker_to_tick_size=ticker_to_tick_size,
                    ticker=ticker
                )

                # Update stop loss levels after stop loss hits
                compressions[compression_name] = update_stop_loss_1(
                    df,
                    ma_name1=sig_ma,
                    ma_name2=con_ma,
                    rsi_column=rsi_col
                )

                # Calculate profit/loss for each ticker's DataFrame
                compressions[compression_name] = calculate_profit_loss_1(
                    df,
                    contract_multiplier=1,
                    ma_name1=sig_ma,
                    ma_name2=con_ma,
                    rsi_column=rsi_col
                )

# Trade logic and implementation for the version that handles the whole dataset before iterating through trend and 
# reversion trading rules
###############################################################################################################################

###############################################################################################################################
# Trade logic before implementing market vs limit order iteration

def generate_trading_signals_1(candles, ma_name1='wma_5', ma_name2='sma_5', rsi_column='rsi_5', strategy_type='trend'):
    """
    Generate buy/sell signals based on moving averages and RSI indicators for either
    a trend-following or mean-reversion strategy.

    Parameters:
    - candles (pd.DataFrame): The DataFrame containing candle data.
    - ma_name1 (str): Column name for the first moving average.
    - ma_name2 (str): Column name for the second moving average.
    - rsi_column (str): Column name for the RSI indicator.
    - strategy_type (str): 'trend' or 'reversion'.

    Returns:
    - candles (pd.DataFrame): The DataFrame with updated signal and position state columns.
    """

    # Append strategy type to column names
    ma_signal_column = f'signal_{ma_name1}_{ma_name2}_{strategy_type}'
    rsi_signal_column = f'signal_{rsi_column}_{strategy_type}'
    ma_position_open_col = f'position_open_{ma_name1}_{ma_name2}_{strategy_type}'
    rsi_position_open_col = f'position_open_{rsi_column}_{strategy_type}'

    # Initialize columns
    candles[ma_signal_column] = 0
    candles[rsi_signal_column] = 0
    candles[ma_position_open_col] = False
    candles[rsi_position_open_col] = False

    # ======= MOVING AVERAGE LOGIC =======
    ma_position_open = False
    ma_signals = []
    ma_positions = []

    for ma1, ma2 in zip(candles[ma_name1], candles[ma_name2]):
        if strategy_type == 'trend':
            # Trend-following logic
            if not ma_position_open and ma1 <= ma2:
                ma_position_open = True
                ma_signals.append(0)  # Enter
            elif ma_position_open and ma1 > ma2:
                ma_position_open = False
                ma_signals.append(1)  # Exit
            else:
                ma_signals.append(ma_signals[-1] if ma_signals else 0)
        elif strategy_type == 'reversion':
            # Mean-reversion logic (inverted)
            if not ma_position_open and ma1 > ma2:
                ma_position_open = True
                ma_signals.append(0)  # Enter
            elif ma_position_open and ma1 <= ma2:
                ma_position_open = False
                ma_signals.append(1)  # Exit
            else:
                ma_signals.append(ma_signals[-1] if ma_signals else 0)

        ma_positions.append(ma_position_open)

    candles[ma_signal_column] = ma_signals
    candles[ma_position_open_col] = ma_positions

    # ======= RSI LOGIC =======
    rsi_position_open = False
    rsi_signals = []
    rsi_positions = []

    for rsi in candles[rsi_column]:
        if strategy_type == 'trend':
            if not rsi_position_open and rsi < 50:
                rsi_position_open = True
                rsi_signals.append(0)  # Enter
            elif rsi_position_open and rsi >= 50:
                rsi_position_open = False
                rsi_signals.append(1)  # Exit
            else:
                rsi_signals.append(rsi_signals[-1] if rsi_signals else 0)
        elif strategy_type == 'reversion':
            if not rsi_position_open and rsi >= 50:
                rsi_position_open = True
                rsi_signals.append(0)  # Enter
            elif rsi_position_open and rsi < 50:
                rsi_position_open = False
                rsi_signals.append(1)  # Exit
            else:
                rsi_signals.append(rsi_signals[-1] if rsi_signals else 0)

        rsi_positions.append(rsi_position_open)

    candles[rsi_signal_column] = rsi_signals
    candles[rsi_position_open_col] = rsi_positions

    return candles

def update_position_open_1(candles, ma_name1='wma_5', ma_name2='sma_5', rsi_column='rsi_5', strategy_type='trend'):
    """
    Update the 'position_open' columns for MA and RSI strategies for a specific strategy type (trend or reversion).

    Parameters:
    - candles (pd.DataFrame): The DataFrame containing the signals and position columns.
    - ma_name1 (str): Column name for the first moving average.
    - ma_name2 (str): Column name for the second moving average.
    - rsi_column (str): Column name for the RSI signal.
    - strategy_type (str): Either 'trend' or 'reversion'

    Returns:
    - pd.DataFrame: The updated DataFrame with 'position_open' columns for the chosen strategy type.
    """
    # Append strategy type to column names
    ma_signal_column = f'signal_{ma_name1}_{ma_name2}_{strategy_type}'
    rsi_signal_column = f'signal_{rsi_column}_{strategy_type}'
    ma_position_open = f'position_open_{ma_name1}_{ma_name2}_{strategy_type}'
    rsi_position_open = f'position_open_{rsi_column}_{strategy_type}'
    
    # Update position open columns based on the signals
    candles[ma_position_open] = candles[ma_signal_column] == 1
    candles[rsi_position_open] = candles[rsi_signal_column] == 1
    
    return candles

def determine_entry_prices_1(candles, ma_name1='wma_5', ma_name2='sma_5', rsi_column='rsi_5', ticker_to_tick_size=None, ticker=None, strategy_type='trend'):
    """
    Determine entry prices for MA and RSI strategies based on signals.

    Parameters:
    - candles (pd.DataFrame): The DataFrame containing candle data.
    - ma_name1 (str): Column name for the first moving average.
    - ma_name2 (str): Column name for the second moving average.
    - rsi_column (str): Column name for the RSI indicator.
    - ticker_to_tick_size (dict): Mapping of tickers to their tick sizes.
    - ticker (str): The ticker for which the tick size applies.
    - strategy_type (str): Either 'trend' or 'reversion', affecting the signal logic.

    Returns:
    - candles (pd.DataFrame): The DataFrame with updated entry price columns.
    """
    # Dynamically generate column names including the strategy type
    ma_signal_column = f'signal_{ma_name1}_{ma_name2}_{strategy_type}'
    rsi_signal_column = f'signal_{rsi_column}_{strategy_type}'
    ma_entry_price = f'entry_price_{ma_name1}_{ma_name2}_{strategy_type}'
    rsi_entry_price = f'entry_price_{rsi_column}_{strategy_type}'

    # Initialize entry price columns
    candles[ma_entry_price] = None
    candles[rsi_entry_price] = None

    # Get tick size
    tick_size = ticker_to_tick_size.get(ticker, 0) if ticker_to_tick_size else 0

    # Moving Average Strategy
    ma_signals = candles[ma_signal_column]
    ma_close_prices = candles['close']
    ma_entry_mask = (ma_signals == 1) & (ma_signals.shift(1) != 1)
    candles.loc[ma_entry_mask, ma_entry_price] = ma_close_prices[ma_entry_mask] + tick_size

    # RSI Strategy
    rsi_signals = candles[rsi_signal_column]
    rsi_entry_mask = (rsi_signals == 1) & (rsi_signals.shift(1) != 1)
    candles.loc[rsi_entry_mask, rsi_entry_price] = ma_close_prices[rsi_entry_mask] + tick_size

    return candles

def determine_exit_prices_1(candles, ma_name1='wma_5', ma_name2='sma_5', rsi_column='rsi_5', ticker_to_tick_size=None, ticker=None, strategy_type='trend'):
    """
    Determine exit prices for MA and RSI strategies based on signals.

    Parameters:
    - candles (pd.DataFrame): The DataFrame containing candle data.
    - ma_name1 (str): Column name for the first moving average.
    - ma_name2 (str): Column name for the second moving average.
    - rsi_column (str): Column name for the RSI indicator.
    - ticker_to_tick_size (dict): Mapping of tickers to their tick sizes.
    - ticker (str): The ticker for which the tick size applies.
    - strategy_type (str): Either 'trend' or 'reversion', affecting the signal logic.

    Returns:
    - candles (pd.DataFrame): The DataFrame with updated exit price columns.
    """
    # Dynamically generate column names including the strategy type
    ma_signal_column = f'signal_{ma_name1}_{ma_name2}_{strategy_type}'
    rsi_signal_column = f'signal_{rsi_column}_{strategy_type}'
    ma_exit_price = f'exit_price_{ma_name1}_{ma_name2}_{strategy_type}'
    rsi_exit_price = f'exit_price_{rsi_column}_{strategy_type}'

    # Initialize exit price columns
    candles[ma_exit_price] = None
    candles[rsi_exit_price] = None

    # Get tick size
    tick_size = ticker_to_tick_size.get(ticker, 0) if ticker_to_tick_size else 0

    # Moving Average Strategy
    ma_signals = candles[ma_signal_column]
    ma_close_prices = candles['close']
    ma_exit_mask = (ma_signals == 0) & (ma_signals.shift(1) == 1)
    candles.loc[ma_exit_mask, ma_exit_price] = ma_close_prices[ma_exit_mask] - tick_size

    # RSI Strategy
    rsi_signals = candles[rsi_signal_column]
    rsi_exit_mask = (rsi_signals == 0) & (rsi_signals.shift(1) == 1)
    candles.loc[rsi_exit_mask, rsi_exit_price] = ma_close_prices[rsi_exit_mask] - tick_size

    return candles

def calculate_stop_losses_1(candles, ma_name1='wma_5', ma_name2='sma_5', rsi_column='rsi_5', strategy_type='trend'):
    """
    Dynamically calculate stop loss levels for MA and RSI strategies and ensure they persist while positions are open.

    Parameters:
    - candles (pd.DataFrame): The DataFrame containing candle data.
    - ma_name1 (str): Column name for the first moving average.
    - ma_name2 (str): Column name for the second moving average.
    - rsi_column (str): Column name for the RSI indicator.
    - strategy_type (str): Either 'trend' or 'reversion', affecting column naming.

    Returns:
    - pd.DataFrame: The updated DataFrame with dynamically named stop loss columns.
    """
    # Dynamically generate column names including the strategy type
    ma_entry_price = f'entry_price_{ma_name1}_{ma_name2}_{strategy_type}'
    ma_exit_price = f'exit_price_{ma_name1}_{ma_name2}_{strategy_type}'
    rsi_entry_price = f'entry_price_{rsi_column}_{strategy_type}'
    rsi_exit_price = f'exit_price_{rsi_column}_{strategy_type}'
    stop_loss_ma = f'stop_loss_{ma_name1}_{ma_name2}_{strategy_type}'
    stop_loss_rsi = f'stop_loss_{rsi_column}_{strategy_type}'

    # Initialize stop loss columns
    candles[stop_loss_ma] = None
    candles[stop_loss_rsi] = None

    # Moving Average Stop Loss
    ma_entry_mask = candles[ma_entry_price].notnull()
    ma_exit_mask = candles[ma_exit_price].notnull()
    
    # Set stop loss where positions open
    candles.loc[ma_entry_mask, stop_loss_ma] = candles[ma_entry_price] - candles['candle_span_max']

    # Reset stop loss and close position where positions close
    candles.loc[ma_exit_mask, stop_loss_ma] = None

    # RSI Stop Loss
    rsi_entry_mask = candles[rsi_entry_price].notnull()
    rsi_exit_mask = candles[rsi_exit_price].notnull()
    
    # Set stop loss where positions open
    candles.loc[rsi_entry_mask, stop_loss_rsi] = candles[rsi_entry_price] - candles['candle_span_max']

    # Reset stop loss and close position where positions close
    candles.loc[rsi_exit_mask, stop_loss_rsi] = None

    # Forward-fill stop loss for both strategies
    candles[stop_loss_ma] = candles[stop_loss_ma].ffill()
    candles[stop_loss_rsi] = candles[stop_loss_rsi].ffill()

    return candles

def track_stop_loss_hits_1(candles, ma_name1='wma_5', ma_name2='sma_5', rsi_column='rsi_5', ticker_to_tick_size=None, ticker=None, strategy_type='trend'):
    """
    Track whether stop losses have been hit for MA and RSI strategies and update dynamically named columns.

    Parameters:
    - candles (pd.DataFrame): The DataFrame containing candle data.
    - ma_name1 (str): Column name for the first moving average.
    - ma_name2 (str): Column name for the second moving average.
    - rsi_column (str): Column name for the RSI indicator.
    - ticker_to_tick_size (dict): Mapping of tickers to their tick sizes.
    - ticker (str): The ticker for which the tick size applies.
    - strategy_type (str): Either 'trend' or 'reversion', affecting column naming.

    Returns:
    - pd.DataFrame: The updated DataFrame with dynamically named stop loss hit flags.
    """
    # Dynamically generate column names including strategy type
    stop_loss_ma = f'stop_loss_{ma_name1}_{ma_name2}_{strategy_type}'
    stop_loss_rsi = f'stop_loss_{rsi_column}_{strategy_type}'
    ma_stop_loss_hit = f'stop_loss_hit_{ma_name1}_{ma_name2}_{strategy_type}'
    rsi_stop_loss_hit = f'stop_loss_hit_{rsi_column}_{strategy_type}'
    ma_position_open = f'position_open_{ma_name1}_{ma_name2}_{strategy_type}'
    rsi_position_open = f'position_open_{rsi_column}_{strategy_type}'

    # Initialize stop loss hit columns
    candles[ma_stop_loss_hit] = False
    candles[rsi_stop_loss_hit] = False

    # Get tick size (ensure it's non-zero)
    tick_size = ticker_to_tick_size.get(ticker, 0) if ticker_to_tick_size else 0

    # Ensure stop loss values are numerical (convert None to NaN)
    candles[stop_loss_ma] = candles[stop_loss_ma].fillna(float('inf'))
    candles[stop_loss_rsi] = candles[stop_loss_rsi].fillna(float('inf'))

    # Moving Average Stop Loss Hit Logic
    ma_hit_condition = (
        (candles[stop_loss_ma].notnull()) & 
        (candles['close'] <= (candles[stop_loss_ma])) & 
        candles[ma_position_open]
    )
    candles.loc[ma_hit_condition, ma_stop_loss_hit] = True

    # RSI Stop Loss Hit Logic
    rsi_hit_condition = (
        (candles[stop_loss_rsi].notnull()) & 
        (candles['close'] <= (candles[stop_loss_rsi])) & 
        candles[rsi_position_open]
    )
    candles.loc[rsi_hit_condition, rsi_stop_loss_hit] = True

    return candles

def adjust_signals_for_stop_loss_1(candles, ma_name1='wma_5', ma_name2='sma_5', rsi_column='rsi_5', strategy_type='trend'):
    """
    Adjust MA and RSI signals to 0 where stop loss has been hit.

    Parameters:
    - candles (pd.DataFrame): The DataFrame containing candle data.
    - ma_name1 (str): Column name for the first moving average.
    - ma_name2 (str): Column name for the second moving average.
    - rsi_column (str): Column name for the RSI indicator.
    - strategy_type (str): Either 'trend' or 'reversion', affecting column naming.

    Returns:
    - pd.DataFrame: The updated DataFrame with adjusted signals.
    """
    # Dynamically generate column names including strategy type
    ma_signal_column = f'signal_{ma_name1}_{ma_name2}_{strategy_type}'
    rsi_signal_column = f'signal_{rsi_column}_{strategy_type}'
    ma_stop_loss_hit_column = f'stop_loss_hit_{ma_name1}_{ma_name2}_{strategy_type}'
    rsi_stop_loss_hit_column = f'stop_loss_hit_{rsi_column}_{strategy_type}'

    # Adjust MA and RSI signals where stop loss has been hit
    candles.loc[candles[ma_stop_loss_hit_column], ma_signal_column] = 0
    candles.loc[candles[rsi_stop_loss_hit_column], rsi_signal_column] = 0

    return candles

def update_stop_loss_1(candles, ma_name1='wma_5', ma_name2='sma_5', rsi_column='rsi_5', strategy_type='trend'):
    """
    Dynamically set stop loss columns to NaN where corresponding signal columns are 0.

    Parameters:
    - candles (pd.DataFrame): The DataFrame containing candle data.
    - ma_name1 (str): Column name for the first moving average.
    - ma_name2 (str): Column name for the second moving average.
    - rsi_column (str): Column name for the RSI indicator.
    - strategy_type (str): Either 'trend' or 'reversion', affecting column naming.

    Returns:
    - pd.DataFrame: The updated DataFrame with dynamically named stop loss columns modified.
    """
    # Dynamically generate column names including strategy type
    ma_signal_column = f'signal_{ma_name1}_{ma_name2}_{strategy_type}'
    rsi_signal_column = f'signal_{rsi_column}_{strategy_type}'
    stop_loss_ma = f'stop_loss_{ma_name1}_{ma_name2}_{strategy_type}'
    stop_loss_rsi = f'stop_loss_{rsi_column}_{strategy_type}'

    # Update stop loss columns to NaN where signals are 0
    candles.loc[candles[ma_signal_column] == 0, stop_loss_ma] = float('nan')
    candles.loc[candles[rsi_signal_column] == 0, stop_loss_rsi] = float('nan')

    return candles

def calculate_profit_loss_1(candles, ma_name1='wma_5', ma_name2='sma_5', rsi_column='rsi_5', contract_multiplier=1, trade_commission=1.5, strategy_type='trend'):
    """ 
    Dynamically calculate profit and loss based on entry and exit price columns, including cumulative commission costs.

    Parameters:
    - candles (pd.DataFrame): The DataFrame containing candle data with entry and exit price columns.
    - contract_multiplier (float): The multiplier for PnL calculation (e.g., contract size).
    - trade_commission (float): The commission cost per trade.
    - ma_name1 (str): Column name for the first moving average.
    - ma_name2 (str): Column name for the second moving average.
    - rsi_column (str): Column name for the RSI indicator.
    - strategy_type (str): The type of strategy ('trend' or 'reversion').

    Returns:
    - pd.DataFrame: The DataFrame with dynamically named profit/loss and commission cost columns.
    """
    # Dynamically generate column names with strategy type
    ma_entry_price = f'entry_price_{ma_name1}_{ma_name2}_{strategy_type}'
    ma_exit_price = f'exit_price_{ma_name1}_{ma_name2}_{strategy_type}'
    rsi_entry_price = f'entry_price_{rsi_column}_{strategy_type}'
    rsi_exit_price = f'exit_price_{rsi_column}_{strategy_type}'
    pnl_ma_col = f'pnl_{ma_name1}_{ma_name2}_{strategy_type}'
    pnl_rsi_col = f'pnl_{rsi_column}_{strategy_type}'
    cum_pnl_ma_col = f'cum_{pnl_ma_col}'
    cum_pnl_rsi_col = f'cum_{pnl_rsi_col}'
    cum_pnl_all_col = f'cum_pnl_all_{ma_name1}_{ma_name2}_{rsi_column}_{strategy_type}'
    ma_commission_col = f'commission_cost_{ma_name1}_{ma_name2}_{strategy_type}'
    rsi_commission_col = f'commission_cost_{rsi_column}_{strategy_type}'

    # Initialize PnL and commission columns
    candles[pnl_ma_col] = 0.0
    candles[pnl_rsi_col] = 0.0
    candles[ma_commission_col] = 0.0
    candles[rsi_commission_col] = 0.0

    # Moving Average Strategy PnL and Commission Calculation
    ma_entry_indices = candles.index[candles[ma_entry_price].notnull()]
    ma_exit_indices = candles.index[candles[ma_exit_price].notnull()]

    # Pair up entry and exit prices
    valid_pairs_ma = min(len(ma_entry_indices), len(ma_exit_indices))
    ma_entry_prices = candles.loc[ma_entry_indices[:valid_pairs_ma], ma_entry_price].values
    ma_exit_prices = candles.loc[ma_exit_indices[:valid_pairs_ma], ma_exit_price].values

    # Calculate commission costs for MA strategy
    candles[ma_commission_col] = candles[ma_entry_price].notna().astype(int) * trade_commission + \
                                 candles[ma_exit_price].notna().astype(int) * trade_commission
    candles[ma_commission_col] = candles[ma_commission_col].cumsum()  # Accumulate commission costs

    # Calculate PnL for MA strategy
    ma_pnl = (ma_exit_prices - ma_entry_prices) * contract_multiplier
    candles.loc[ma_exit_indices[:valid_pairs_ma], pnl_ma_col] = ma_pnl

    # RSI Strategy PnL and Commission Calculation
    rsi_entry_indices = candles.index[candles[rsi_entry_price].notnull()]
    rsi_exit_indices = candles.index[candles[rsi_exit_price].notnull()]

    # Pair up entry and exit prices
    valid_pairs_rsi = min(len(rsi_entry_indices), len(rsi_exit_indices))
    rsi_entry_prices = candles.loc[rsi_entry_indices[:valid_pairs_rsi], rsi_entry_price].values
    rsi_exit_prices = candles.loc[rsi_exit_indices[:valid_pairs_rsi], rsi_exit_price].values

    # Calculate commission costs for RSI strategy
    candles[rsi_commission_col] = candles[rsi_entry_price].notna().astype(int) * trade_commission + \
                                  candles[rsi_exit_price].notna().astype(int) * trade_commission
    candles[rsi_commission_col] = candles[rsi_commission_col].cumsum()  # Accumulate commission costs

    # Calculate PnL for RSI strategy
    rsi_pnl = (rsi_exit_prices - rsi_entry_prices) * contract_multiplier
    candles.loc[rsi_exit_indices[:valid_pairs_rsi], pnl_rsi_col] = rsi_pnl

    # Calculate cumulative PnL for both strategies
    candles[cum_pnl_ma_col] = candles[pnl_ma_col].cumsum()
    candles[cum_pnl_rsi_col] = candles[pnl_rsi_col].cumsum()

    # Calculate combined cumulative PnL
    candles[cum_pnl_all_col] = candles[cum_pnl_ma_col] + candles[cum_pnl_rsi_col]

    return candles

# Iterate through all tickers in compressed_sessions
for ticker, sessions in compressed_sessions.items():
    print(f"Processing ticker: {ticker}")

    # Iterate through all sessions of the current ticker
    for time_slice, compressions in sessions.items():
        print(f"  Processing session: {time_slice}, Total Compressions: {len(compressions)}")

        # Iterate through all compression levels of the current session
        for compression_name, df in compressions.items():
            print(f"    Processing compression: {compression_name} in session {time_slice}")

            # Iterate through all the ma_combinations
            for sig_ma, con_ma, rsi_col in ma_combinations:

                # Iterate through both 'trend' and 'reversion' strategies
                for strategy_type in ['trend', 'reversion']:
                    print(f"      Applying {strategy_type} strategy for {sig_ma} and {con_ma}")

                    # Generate trading signals for the current strategy
                    compressions[compression_name] = generate_trading_signals_1(
                        df,
                        ma_name1=sig_ma,
                        ma_name2=con_ma,
                        rsi_column=rsi_col,
                        strategy_type=strategy_type
                    )

                    # Update position_open columns to be 1:1 verbal boolean with the signal
                    compressions[compression_name] = update_position_open_1(
                        df,
                        ma_name1=sig_ma,
                        ma_name2=con_ma,
                        rsi_column=rsi_col,
                        strategy_type=strategy_type
                    )

                    # Determine entry prices for each ticker
                    compressions[compression_name] = determine_entry_prices_1(
                        df,
                        ma_name1=sig_ma,
                        ma_name2=con_ma,
                        rsi_column=rsi_col,
                        ticker_to_tick_size=ticker_to_tick_size,
                        ticker=ticker,
                        strategy_type=strategy_type
                    )

                    # Determine exit prices for each ticker
                    compressions[compression_name] = determine_exit_prices_1(
                        df,
                        ma_name1=sig_ma,
                        ma_name2=con_ma,
                        rsi_column=rsi_col,
                        ticker_to_tick_size=ticker_to_tick_size,
                        ticker=ticker,
                        strategy_type=strategy_type
                    )

                    # Stop loss calculation
                    compressions[compression_name] = calculate_stop_losses_1(
                        df,
                        ma_name1=sig_ma,
                        ma_name2=con_ma,
                        rsi_column=rsi_col,
                        strategy_type=strategy_type
                    )

                    # Track stop loss hits
                    compressions[compression_name] = track_stop_loss_hits_1(
                        df,
                        ma_name1=sig_ma,
                        ma_name2=con_ma,
                        rsi_column=rsi_col,
                        ticker_to_tick_size=ticker_to_tick_size,
                        ticker=ticker,
                        strategy_type=strategy_type
                    )

                    # Adjust signals from stop loss hits
                    compressions[compression_name] = adjust_signals_for_stop_loss_1(
                        df,
                        ma_name1=sig_ma,
                        ma_name2=con_ma,
                        rsi_column=rsi_col,
                        strategy_type=strategy_type
                    )

                    # Re-update position_open column after stop loss hits
                    compressions[compression_name] = update_position_open_1(
                        df,
                        ma_name1=sig_ma,
                        ma_name2=con_ma,
                        rsi_column=rsi_col,
                        strategy_type=strategy_type
                    )

                    # Re-determine entry prices after stop loss hits
                    compressions[compression_name] = determine_entry_prices_1(
                        df,
                        ma_name1=sig_ma,
                        ma_name2=con_ma,
                        rsi_column=rsi_col,
                        ticker_to_tick_size=ticker_to_tick_size,
                        ticker=ticker,
                        strategy_type=strategy_type
                    )

                    # Re-determine exit prices after stop loss hits
                    compressions[compression_name] = determine_exit_prices_1(
                        df,
                        ma_name1=sig_ma,
                        ma_name2=con_ma,
                        rsi_column=rsi_col,
                        ticker_to_tick_size=ticker_to_tick_size,
                        ticker=ticker,
                        strategy_type=strategy_type
                    )

                    # Update stop loss levels after stop loss hits
                    compressions[compression_name] = update_stop_loss_1(
                        df,
                        ma_name1=sig_ma,
                        ma_name2=con_ma,
                        rsi_column=rsi_col,
                        strategy_type=strategy_type
                        
                    )

                    # Calculate profit/loss for each ticker's DataFrame
                    compressions[compression_name] = calculate_profit_loss_1(
                        df,
                        contract_multiplier=1,
                        ma_name1=sig_ma,
                        ma_name2=con_ma,
                        rsi_column=rsi_col,
                        strategy_type=strategy_type
                    )

# Trade logic before implementing market vs limit order iteration
###############################################################################################################################

def determine_exit_prices_1(candles, ma_name1='wma_5', ma_name2='sma_5', rsi_column='rsi_5', 
                            ticker_to_tick_size=None, ticker=None, strategy_type='trend', order_type='market'):
    """
    Determine exit prices for MA and RSI strategies based on signals.

    Parameters:
    - candles (pd.DataFrame): The DataFrame containing candle data.
    - ma_name1 (str): Column name for the first moving average.
    - ma_name2 (str): Column name for the second moving average.
    - rsi_column (str): Column name for the RSI indicator.
    - ticker_to_tick_size (dict): Mapping of tickers to their tick sizes.
    - ticker (str): The ticker for which the tick size applies.
    - strategy_type (str): Either 'trend' or 'reversion', affecting the signal logic.
    - order_type (str): 'market' or 'limit', defining how exit prices are set.

    Returns:
    - candles (pd.DataFrame): The DataFrame with updated exit price columns.
    """
    # Dynamically generate column names including the strategy type
    ma_signal_column = f'signal_{ma_name1}_{ma_name2}_{strategy_type}_{order_type}'
    rsi_signal_column = f'signal_{rsi_column}_{strategy_type}_{order_type}'
    ma_exit_price = f'exit_price_{ma_name1}_{ma_name2}_{strategy_type}_{order_type}'
    rsi_exit_price = f'exit_price_{rsi_column}_{strategy_type}_{order_type}'

    # Initialize exit price columns
    candles[ma_exit_price] = None
    candles[rsi_exit_price] = None

    # Get tick size
    tick_size = ticker_to_tick_size.get(ticker, 0) if ticker_to_tick_size else 0

    # Determine tick size adjustment based on order type
    subtract_tick_size = order_type == 'market'

    # Moving Average Strategy
    ma_signals = candles[ma_signal_column]
    ma_close_prices = candles['close']
    ma_exit_mask = (ma_signals == 0) & (ma_signals.shift(1) == 1)
    candles.loc[ma_exit_mask, ma_exit_price] = ma_close_prices[ma_exit_mask] - (tick_size if subtract_tick_size else 0)

    # RSI Strategy
    rsi_signals = candles[rsi_signal_column]
    rsi_exit_mask = (rsi_signals == 0) & (rsi_signals.shift(1) == 1)
    candles.loc[rsi_exit_mask, rsi_exit_price] = ma_close_prices[rsi_exit_mask] - (tick_size if subtract_tick_size else 0)

    return candles

###############################################################################################################################
# Trade logic after implementing market vs limit order iteration and before iterating through directional bias

def generate_trading_signals_long(candles, ma_name1='wma_5', ma_name2='sma_5', rsi_column='rsi_5', strategy_type='trend', order_type='market'):
    """
    Generate buy/sell signals based on moving averages and RSI indicators for either
    a trend-following or mean-reversion strategy.

    Parameters:
    - candles (pd.DataFrame): The DataFrame containing candle data.
    - ma_name1 (str): Column name for the first moving average.
    - ma_name2 (str): Column name for the second moving average.
    - rsi_column (str): Column name for the RSI indicator.
    - strategy_type (str): 'trend' or 'reversion'.

    Returns:
    - candles (pd.DataFrame): The DataFrame with updated signal and position state columns.
    """

    # Append strategy type to column names
    ma_signal_column = f'signal_{ma_name1}_{ma_name2}_{strategy_type}_{order_type}'
    rsi_signal_column = f'signal_{rsi_column}_{strategy_type}_{order_type}'
    ma_position_open_col = f'position_open_{ma_name1}_{ma_name2}_{strategy_type}_{order_type}'
    rsi_position_open_col = f'position_open_{rsi_column}_{strategy_type}_{order_type}'

    # Initialize columns
    candles[ma_signal_column] = 0
    candles[rsi_signal_column] = 0
    candles[ma_position_open_col] = False
    candles[rsi_position_open_col] = False

    # ======= MOVING AVERAGE LOGIC =======
    ma_position_open = False
    ma_signals = []
    ma_positions = []

    for ma1, ma2 in zip(candles[ma_name1], candles[ma_name2]):
        if strategy_type == 'trend':
            # Trend-following logic
            if not ma_position_open and ma1 <= ma2:
                ma_position_open = True
                ma_signals.append(0)  # Enter
            elif ma_position_open and ma1 > ma2:
                ma_position_open = False
                ma_signals.append(1)  # Exit
            else:
                ma_signals.append(ma_signals[-1] if ma_signals else 0)
        elif strategy_type == 'reversion':
            # Mean-reversion logic (inverted)
            if not ma_position_open and ma1 > ma2:
                ma_position_open = True
                ma_signals.append(0)  # Enter
            elif ma_position_open and ma1 <= ma2:
                ma_position_open = False
                ma_signals.append(1)  # Exit
            else:
                ma_signals.append(ma_signals[-1] if ma_signals else 0)

        ma_positions.append(ma_position_open)

    candles[ma_signal_column] = ma_signals
    candles[ma_position_open_col] = ma_positions

    # ======= RSI LOGIC =======
    rsi_position_open = False
    rsi_signals = []
    rsi_positions = []

    for rsi in candles[rsi_column]:
        if strategy_type == 'trend':
            if not rsi_position_open and rsi < 50:
                rsi_position_open = True
                rsi_signals.append(0)  # Enter
            elif rsi_position_open and rsi >= 50:
                rsi_position_open = False
                rsi_signals.append(1)  # Exit
            else:
                rsi_signals.append(rsi_signals[-1] if rsi_signals else 0)
        elif strategy_type == 'reversion':
            if not rsi_position_open and rsi >= 50:
                rsi_position_open = True
                rsi_signals.append(0)  # Enter
            elif rsi_position_open and rsi < 50:
                rsi_position_open = False
                rsi_signals.append(1)  # Exit
            else:
                rsi_signals.append(rsi_signals[-1] if rsi_signals else 0)

        rsi_positions.append(rsi_position_open)

    candles[rsi_signal_column] = rsi_signals
    candles[rsi_position_open_col] = rsi_positions

    return candles

def update_position_open_1(candles, ma_name1='wma_5', ma_name2='sma_5', rsi_column='rsi_5', strategy_type='trend', order_type='market'):
    """
    Update the 'position_open' columns for MA and RSI strategies for a specific strategy type (trend or reversion).

    Parameters:
    - candles (pd.DataFrame): The DataFrame containing the signals and position columns.
    - ma_name1 (str): Column name for the first moving average.
    - ma_name2 (str): Column name for the second moving average.
    - rsi_column (str): Column name for the RSI signal.
    - strategy_type (str): Either 'trend' or 'reversion'

    Returns:
    - pd.DataFrame: The updated DataFrame with 'position_open' columns for the chosen strategy type.
    """
    # Append strategy type to column names
    ma_signal_column = f'signal_{ma_name1}_{ma_name2}_{strategy_type}_{order_type}'
    rsi_signal_column = f'signal_{rsi_column}_{strategy_type}_{order_type}'
    ma_position_open = f'position_open_{ma_name1}_{ma_name2}_{strategy_type}_{order_type}'
    rsi_position_open = f'position_open_{rsi_column}_{strategy_type}_{order_type}'
    
    # Update position open columns based on the signals
    candles[ma_position_open] = candles[ma_signal_column] == 1
    candles[rsi_position_open] = candles[rsi_signal_column] == 1
    
    return candles

def determine_entry_prices_1(candles, ma_name1='wma_5', ma_name2='sma_5', rsi_column='rsi_5', 
                             ticker_to_tick_size=None, ticker=None, strategy_type='trend', order_type='market'):
    """
    Determine entry prices for MA and RSI strategies based on signals.

    Parameters:
    - candles (pd.DataFrame): The DataFrame containing candle data.
    - ma_name1 (str): Column name for the first moving average.
    - ma_name2 (str): Column name for the second moving average.
    - rsi_column (str): Column name for the RSI indicator.
    - ticker_to_tick_size (dict): Mapping of tickers to their tick sizes.
    - ticker (str): The ticker for which the tick size applies.
    - strategy_type (str): Either 'trend' or 'reversion', affecting the signal logic.
    - order_type (str): 'market' or 'limit', defining how entry prices are set.

    Returns:
    - candles (pd.DataFrame): The DataFrame with updated entry price columns.
    """
    # Dynamically generate column names including the strategy type
    ma_signal_column = f'signal_{ma_name1}_{ma_name2}_{strategy_type}_{order_type}'
    rsi_signal_column = f'signal_{rsi_column}_{strategy_type}_{order_type}'
    ma_entry_price = f'entry_price_{ma_name1}_{ma_name2}_{strategy_type}_{order_type}'
    rsi_entry_price = f'entry_price_{rsi_column}_{strategy_type}_{order_type}'

    # Initialize entry price columns
    candles[ma_entry_price] = None
    candles[rsi_entry_price] = None

    # Get tick size
    tick_size = ticker_to_tick_size.get(ticker, 0) if ticker_to_tick_size else 0

    # Determine tick size adjustment based on order type
    add_tick_size = order_type == 'market'

    # Moving Average Strategy
    ma_signals = candles[ma_signal_column]
    ma_close_prices = candles['close']
    ma_entry_mask = (ma_signals == 1) & (ma_signals.shift(1) != 1)
    candles.loc[ma_entry_mask, ma_entry_price] = ma_close_prices[ma_entry_mask] + (tick_size if add_tick_size else 0)

    # RSI Strategy
    rsi_signals = candles[rsi_signal_column]
    rsi_entry_mask = (rsi_signals == 1) & (rsi_signals.shift(1) != 1)
    candles.loc[rsi_entry_mask, rsi_entry_price] = ma_close_prices[rsi_entry_mask] + (tick_size if add_tick_size else 0)

    return candles

def determine_exit_prices_1(candles, ma_name1='wma_5', ma_name2='sma_5', rsi_column='rsi_5', 
                            ticker_to_tick_size=None, ticker=None, strategy_type='trend', order_type='market'):
    """
    Determine exit prices for MA and RSI strategies based on signals.

    Parameters:
    - candles (pd.DataFrame): The DataFrame containing candle data.
    - ma_name1 (str): Column name for the first moving average.
    - ma_name2 (str): Column name for the second moving average.
    - rsi_column (str): Column name for the RSI indicator.
    - ticker_to_tick_size (dict): Mapping of tickers to their tick sizes.
    - ticker (str): The ticker for which the tick size applies.
    - strategy_type (str): Either 'trend' or 'reversion', affecting the signal logic.
    - order_type (str): 'market' or 'limit', defining how exit prices are set.

    Returns:
    - candles (pd.DataFrame): The DataFrame with updated exit price columns.
    """
    # Dynamically generate column names including the strategy type
    ma_signal_column = f'signal_{ma_name1}_{ma_name2}_{strategy_type}_{order_type}'
    rsi_signal_column = f'signal_{rsi_column}_{strategy_type}_{order_type}'
    ma_exit_price = f'exit_price_{ma_name1}_{ma_name2}_{strategy_type}_{order_type}'
    rsi_exit_price = f'exit_price_{rsi_column}_{strategy_type}_{order_type}'

    # Initialize exit price columns
    candles[ma_exit_price] = None
    candles[rsi_exit_price] = None

    # Get tick size
    tick_size = ticker_to_tick_size.get(ticker, 0) if ticker_to_tick_size else 0

    # Determine tick size adjustment based on order type
    subtract_tick_size = order_type == 'market'

    # Moving Average Strategy
    ma_signals = candles[ma_signal_column]
    ma_close_prices = candles['close']
    ma_exit_mask = (ma_signals == 0) & (ma_signals.shift(1) == 1)
    candles.loc[ma_exit_mask, ma_exit_price] = ma_close_prices[ma_exit_mask] - (tick_size if subtract_tick_size else 0)

    # RSI Strategy
    rsi_signals = candles[rsi_signal_column]
    rsi_exit_mask = (rsi_signals == 0) & (rsi_signals.shift(1) == 1)
    candles.loc[rsi_exit_mask, rsi_exit_price] = ma_close_prices[rsi_exit_mask] - (tick_size if subtract_tick_size else 0)

    return candles

def calculate_stop_losses_1(candles, ma_name1='wma_5', ma_name2='sma_5', rsi_column='rsi_5', strategy_type='trend', order_type='market'):
    """
    Dynamically calculate stop loss levels for MA and RSI strategies and ensure they persist while positions are open.

    Parameters:
    - candles (pd.DataFrame): The DataFrame containing candle data.
    - ma_name1 (str): Column name for the first moving average.
    - ma_name2 (str): Column name for the second moving average.
    - rsi_column (str): Column name for the RSI indicator.
    - strategy_type (str): Either 'trend' or 'reversion', affecting column naming.

    Returns:
    - pd.DataFrame: The updated DataFrame with dynamically named stop loss columns.
    """
    # Dynamically generate column names including the strategy type
    ma_entry_price = f'entry_price_{ma_name1}_{ma_name2}_{strategy_type}_{order_type}'
    ma_exit_price = f'exit_price_{ma_name1}_{ma_name2}_{strategy_type}_{order_type}'
    rsi_entry_price = f'entry_price_{rsi_column}_{strategy_type}_{order_type}'
    rsi_exit_price = f'exit_price_{rsi_column}_{strategy_type}_{order_type}'
    stop_loss_ma = f'stop_loss_{ma_name1}_{ma_name2}_{strategy_type}_{order_type}'
    stop_loss_rsi = f'stop_loss_{rsi_column}_{strategy_type}_{order_type}'

    # Initialize stop loss columns
    candles[stop_loss_ma] = None
    candles[stop_loss_rsi] = None

    # Moving Average Stop Loss
    ma_entry_mask = candles[ma_entry_price].notnull()
    ma_exit_mask = candles[ma_exit_price].notnull()
    
    # Set stop loss where positions open
    candles.loc[ma_entry_mask, stop_loss_ma] = candles[ma_entry_price] - candles['candle_span_max']

    # Reset stop loss and close position where positions close
    candles.loc[ma_exit_mask, stop_loss_ma] = None

    # RSI Stop Loss
    rsi_entry_mask = candles[rsi_entry_price].notnull()
    rsi_exit_mask = candles[rsi_exit_price].notnull()
    
    # Set stop loss where positions open
    candles.loc[rsi_entry_mask, stop_loss_rsi] = candles[rsi_entry_price] - candles['candle_span_max']

    # Reset stop loss and close position where positions close
    candles.loc[rsi_exit_mask, stop_loss_rsi] = None

    # Forward-fill stop loss for both strategies
    candles[stop_loss_ma] = candles[stop_loss_ma].ffill()
    candles[stop_loss_rsi] = candles[stop_loss_rsi].ffill()

    return candles

def track_stop_loss_hits_1(candles, ma_name1='wma_5', ma_name2='sma_5', rsi_column='rsi_5', ticker_to_tick_size=None, ticker=None, strategy_type='trend', order_type='market'):
    """
    Track whether stop losses have been hit for MA and RSI strategies and update dynamically named columns.

    Parameters:
    - candles (pd.DataFrame): The DataFrame containing candle data.
    - ma_name1 (str): Column name for the first moving average.
    - ma_name2 (str): Column name for the second moving average.
    - rsi_column (str): Column name for the RSI indicator.
    - ticker_to_tick_size (dict): Mapping of tickers to their tick sizes.
    - ticker (str): The ticker for which the tick size applies.
    - strategy_type (str): Either 'trend' or 'reversion', affecting column naming.

    Returns:
    - pd.DataFrame: The updated DataFrame with dynamically named stop loss hit flags.
    """
    # Dynamically generate column names including strategy type
    stop_loss_ma = f'stop_loss_{ma_name1}_{ma_name2}_{strategy_type}_{order_type}'
    stop_loss_rsi = f'stop_loss_{rsi_column}_{strategy_type}_{order_type}'
    ma_stop_loss_hit = f'stop_loss_hit_{ma_name1}_{ma_name2}_{strategy_type}_{order_type}'
    rsi_stop_loss_hit = f'stop_loss_hit_{rsi_column}_{strategy_type}_{order_type}'
    ma_position_open = f'position_open_{ma_name1}_{ma_name2}_{strategy_type}_{order_type}'
    rsi_position_open = f'position_open_{rsi_column}_{strategy_type}_{order_type}'

    # Initialize stop loss hit columns
    candles[ma_stop_loss_hit] = False
    candles[rsi_stop_loss_hit] = False

    # Get tick size (ensure it's non-zero)
    tick_size = ticker_to_tick_size.get(ticker, 0) if ticker_to_tick_size else 0

    # Ensure stop loss values are numerical (convert None to NaN)
    candles[stop_loss_ma] = candles[stop_loss_ma].fillna(float('inf'))
    candles[stop_loss_rsi] = candles[stop_loss_rsi].fillna(float('inf'))

    # Moving Average Stop Loss Hit Logic
    ma_hit_condition = (
        (candles[stop_loss_ma].notnull()) & 
        (candles['close'] <= (candles[stop_loss_ma])) & 
        candles[ma_position_open]
    )
    candles.loc[ma_hit_condition, ma_stop_loss_hit] = True

    # RSI Stop Loss Hit Logic
    rsi_hit_condition = (
        (candles[stop_loss_rsi].notnull()) & 
        (candles['close'] <= (candles[stop_loss_rsi])) & 
        candles[rsi_position_open]
    )
    candles.loc[rsi_hit_condition, rsi_stop_loss_hit] = True

    return candles

def adjust_signals_for_stop_loss_1(candles, ma_name1='wma_5', ma_name2='sma_5', rsi_column='rsi_5', strategy_type='trend', order_type='market'):
    """
    Adjust MA and RSI signals to 0 where stop loss has been hit.

    Parameters:
    - candles (pd.DataFrame): The DataFrame containing candle data.
    - ma_name1 (str): Column name for the first moving average.
    - ma_name2 (str): Column name for the second moving average.
    - rsi_column (str): Column name for the RSI indicator.
    - strategy_type (str): Either 'trend' or 'reversion', affecting column naming.

    Returns:
    - pd.DataFrame: The updated DataFrame with adjusted signals.
    """
    # Dynamically generate column names including strategy type
    ma_signal_column = f'signal_{ma_name1}_{ma_name2}_{strategy_type}_{order_type}'
    rsi_signal_column = f'signal_{rsi_column}_{strategy_type}_{order_type}'
    ma_stop_loss_hit_column = f'stop_loss_hit_{ma_name1}_{ma_name2}_{strategy_type}_{order_type}'
    rsi_stop_loss_hit_column = f'stop_loss_hit_{rsi_column}_{strategy_type}_{order_type}'

    # Adjust MA and RSI signals where stop loss has been hit
    candles.loc[candles[ma_stop_loss_hit_column], ma_signal_column] = 0
    candles.loc[candles[rsi_stop_loss_hit_column], rsi_signal_column] = 0

    return candles

def update_stop_loss_1(candles, ma_name1='wma_5', ma_name2='sma_5', rsi_column='rsi_5', strategy_type='trend', order_type='market'):
    """
    Dynamically set stop loss columns to NaN where corresponding signal columns are 0.

    Parameters:
    - candles (pd.DataFrame): The DataFrame containing candle data.
    - ma_name1 (str): Column name for the first moving average.
    - ma_name2 (str): Column name for the second moving average.
    - rsi_column (str): Column name for the RSI indicator.
    - strategy_type (str): Either 'trend' or 'reversion', affecting column naming.

    Returns:
    - pd.DataFrame: The updated DataFrame with dynamically named stop loss columns modified.
    """
    # Dynamically generate column names including strategy type
    ma_signal_column = f'signal_{ma_name1}_{ma_name2}_{strategy_type}_{order_type}'
    rsi_signal_column = f'signal_{rsi_column}_{strategy_type}_{order_type}'
    stop_loss_ma = f'stop_loss_{ma_name1}_{ma_name2}_{strategy_type}_{order_type}'
    stop_loss_rsi = f'stop_loss_{rsi_column}_{strategy_type}_{order_type}'

    # Update stop loss columns to NaN where signals are 0
    candles.loc[candles[ma_signal_column] == 0, stop_loss_ma] = float('nan')
    candles.loc[candles[rsi_signal_column] == 0, stop_loss_rsi] = float('nan')

    return candles

def calculate_profit_loss_1(candles, ma_name1='wma_5', ma_name2='sma_5', rsi_column='rsi_5', contract_multiplier=1, trade_commission=1.5, strategy_type='trend', order_type='market'):
    """ 
    Dynamically calculate profit and loss based on entry and exit price columns, including cumulative commission costs.

    Parameters:
    - candles (pd.DataFrame): The DataFrame containing candle data with entry and exit price columns.
    - contract_multiplier (float): The multiplier for PnL calculation (e.g., contract size).
    - trade_commission (float): The commission cost per trade.
    - ma_name1 (str): Column name for the first moving average.
    - ma_name2 (str): Column name for the second moving average.
    - rsi_column (str): Column name for the RSI indicator.
    - strategy_type (str): The type of strategy ('trend' or 'reversion').

    Returns:
    - pd.DataFrame: The DataFrame with dynamically named profit/loss and commission cost columns.
    """
    # Dynamically generate column names with strategy type
    ma_entry_price = f'entry_price_{ma_name1}_{ma_name2}_{strategy_type}_{order_type}'
    ma_exit_price = f'exit_price_{ma_name1}_{ma_name2}_{strategy_type}_{order_type}'
    rsi_entry_price = f'entry_price_{rsi_column}_{strategy_type}_{order_type}'
    rsi_exit_price = f'exit_price_{rsi_column}_{strategy_type}_{order_type}'
    pnl_ma_col = f'pnl_{ma_name1}_{ma_name2}_{strategy_type}_{order_type}'
    pnl_rsi_col = f'pnl_{rsi_column}_{strategy_type}_{order_type}'
    cum_pnl_ma_col = f'cum_{pnl_ma_col}'
    cum_pnl_rsi_col = f'cum_{pnl_rsi_col}'
    cum_pnl_all_col = f'cum_pnl_all_{ma_name1}_{ma_name2}_{rsi_column}_{strategy_type}_{order_type}'
    ma_commission_col = f'commission_cost_{ma_name1}_{ma_name2}_{strategy_type}_{order_type}'
    rsi_commission_col = f'commission_cost_{rsi_column}_{strategy_type}_{order_type}'

    # Initialize PnL and commission columns
    candles[pnl_ma_col] = 0.0
    candles[pnl_rsi_col] = 0.0
    candles[ma_commission_col] = 0.0
    candles[rsi_commission_col] = 0.0

    # Moving Average Strategy PnL and Commission Calculation
    ma_entry_indices = candles.index[candles[ma_entry_price].notnull()]
    ma_exit_indices = candles.index[candles[ma_exit_price].notnull()]

    # Pair up entry and exit prices
    valid_pairs_ma = min(len(ma_entry_indices), len(ma_exit_indices))
    ma_entry_prices = candles.loc[ma_entry_indices[:valid_pairs_ma], ma_entry_price].values
    ma_exit_prices = candles.loc[ma_exit_indices[:valid_pairs_ma], ma_exit_price].values

    # Calculate commission costs for MA strategy
    candles[ma_commission_col] = candles[ma_entry_price].notna().astype(int) * trade_commission + \
                                 candles[ma_exit_price].notna().astype(int) * trade_commission
    candles[ma_commission_col] = candles[ma_commission_col].cumsum()  # Accumulate commission costs

    # Calculate PnL for MA strategy
    ma_pnl = (ma_exit_prices - ma_entry_prices) * contract_multiplier
    candles.loc[ma_exit_indices[:valid_pairs_ma], pnl_ma_col] = ma_pnl

    # RSI Strategy PnL and Commission Calculation
    rsi_entry_indices = candles.index[candles[rsi_entry_price].notnull()]
    rsi_exit_indices = candles.index[candles[rsi_exit_price].notnull()]

    # Pair up entry and exit prices
    valid_pairs_rsi = min(len(rsi_entry_indices), len(rsi_exit_indices))
    rsi_entry_prices = candles.loc[rsi_entry_indices[:valid_pairs_rsi], rsi_entry_price].values
    rsi_exit_prices = candles.loc[rsi_exit_indices[:valid_pairs_rsi], rsi_exit_price].values

    # Calculate commission costs for RSI strategy
    candles[rsi_commission_col] = candles[rsi_entry_price].notna().astype(int) * trade_commission + \
                                  candles[rsi_exit_price].notna().astype(int) * trade_commission
    candles[rsi_commission_col] = candles[rsi_commission_col].cumsum()  # Accumulate commission costs

    # Calculate PnL for RSI strategy
    rsi_pnl = (rsi_exit_prices - rsi_entry_prices) * contract_multiplier
    candles.loc[rsi_exit_indices[:valid_pairs_rsi], pnl_rsi_col] = rsi_pnl

    # Calculate cumulative PnL for both strategies
    candles[cum_pnl_ma_col] = candles[pnl_ma_col].cumsum()
    candles[cum_pnl_rsi_col] = candles[pnl_rsi_col].cumsum()

    # Calculate combined cumulative PnL
    candles[cum_pnl_all_col] = candles[cum_pnl_ma_col] + candles[cum_pnl_rsi_col]

    return candles

# Iterate through all tickers in compressed_sessions
for ticker, sessions in compressed_sessions.items():
    print(f"Processing ticker: {ticker}")

    # Iterate through all sessions of the current ticker
    for time_slice, compressions in sessions.items():
        print(f"  Processing session: {time_slice}, Total Compressions: {len(compressions)}")

        # Iterate through all compression levels of the current session
        for compression_name, df in compressions.items():
            print(f"    Processing compression: {compression_name} in session {time_slice}")

            # Iterate through all the ma_combinations
            for sig_ma, con_ma, rsi_col in ma_combinations:

                # Iterate through both 'trend' and 'reversion' strategies
                for strategy_type in ['trend', 'reversion']:
                    print(f"      Applying {strategy_type} strategy for {sig_ma} and {con_ma}")

                    for order_type in ['market', 'limit']:
                        print(f"        Applying {order_type} order_type")

                        # Generate trading signals for the current strategy
                        compressions[compression_name] = generate_trading_signals_long(
                            df,
                            ma_name1=sig_ma,
                            ma_name2=con_ma,
                            rsi_column=rsi_col,
                            strategy_type=strategy_type,
                            order_type=order_type
                        )

                        # Update position_open columns to be 1:1 verbal boolean with the signal
                        compressions[compression_name] = update_position_open_1(
                            df,
                            ma_name1=sig_ma,
                            ma_name2=con_ma,
                            rsi_column=rsi_col,
                            strategy_type=strategy_type,
                            order_type=order_type
                        )

                        # Determine entry prices for each ticker
                        compressions[compression_name] = determine_entry_prices_1(
                            df,
                            ma_name1=sig_ma,
                            ma_name2=con_ma,
                            rsi_column=rsi_col,
                            ticker_to_tick_size=ticker_to_tick_size,
                            ticker=ticker,
                            strategy_type=strategy_type,
                            order_type=order_type
                        )

                        # Determine exit prices for each ticker
                        compressions[compression_name] = determine_exit_prices_1(
                            df,
                            ma_name1=sig_ma,
                            ma_name2=con_ma,
                            rsi_column=rsi_col,
                            ticker_to_tick_size=ticker_to_tick_size,
                            ticker=ticker,
                            strategy_type=strategy_type,
                            order_type=order_type
                        )

                        # Stop loss calculation
                        compressions[compression_name] = calculate_stop_losses_1(
                            df,
                            ma_name1=sig_ma,
                            ma_name2=con_ma,
                            rsi_column=rsi_col,
                            strategy_type=strategy_type,
                            order_type=order_type
                        )

                        # Track stop loss hits
                        compressions[compression_name] = track_stop_loss_hits_1(
                            df,
                            ma_name1=sig_ma,
                            ma_name2=con_ma,
                            rsi_column=rsi_col,
                            ticker_to_tick_size=ticker_to_tick_size,
                            ticker=ticker,
                            strategy_type=strategy_type,
                            order_type=order_type
                        )

                        # Adjust signals from stop loss hits
                        compressions[compression_name] = adjust_signals_for_stop_loss_1(
                            df,
                            ma_name1=sig_ma,
                            ma_name2=con_ma,
                            rsi_column=rsi_col,
                            strategy_type=strategy_type,
                            order_type=order_type
                        )

                        # Re-update position_open column after stop loss hits
                        compressions[compression_name] = update_position_open_1(
                            df,
                            ma_name1=sig_ma,
                            ma_name2=con_ma,
                            rsi_column=rsi_col,
                            strategy_type=strategy_type,
                            order_type=order_type
                        )

                        # Re-determine entry prices after stop loss hits
                        compressions[compression_name] = determine_entry_prices_1(
                            df,
                            ma_name1=sig_ma,
                            ma_name2=con_ma,
                            rsi_column=rsi_col,
                            ticker_to_tick_size=ticker_to_tick_size,
                            ticker=ticker,
                            strategy_type=strategy_type,
                            order_type=order_type
                        )

                        # Re-determine exit prices after stop loss hits
                        compressions[compression_name] = determine_exit_prices_1(
                            df,
                            ma_name1=sig_ma,
                            ma_name2=con_ma,
                            rsi_column=rsi_col,
                            ticker_to_tick_size=ticker_to_tick_size,
                            ticker=ticker,
                            strategy_type=strategy_type,
                            order_type=order_type
                        )

                        # Update stop loss levels after stop loss hits
                        compressions[compression_name] = update_stop_loss_1(
                            df,
                            ma_name1=sig_ma,
                            ma_name2=con_ma,
                            rsi_column=rsi_col,
                            strategy_type=strategy_type,
                            order_type=order_type
                            
                        )

                        # Calculate profit/loss for each ticker's DataFrame
                        compressions[compression_name] = calculate_profit_loss_1(
                            df,
                            contract_multiplier=1,
                            ma_name1=sig_ma,
                            ma_name2=con_ma,
                            rsi_column=rsi_col,
                            strategy_type=strategy_type,
                            order_type=order_type
                        )

def generate_pnl_dataframe(compressed_sessions, ticker_to_tick_size, ticker_to_point_value, 
                           ma_name1='wma_5', ma_name2='sma_5', rsi_column='rsi_5', 
                           strategy_type='trend', order_type='market',
                           daily_stop_loss_dollars=-1000):
    """
    Generates a DataFrame containing detailed PnL, trade statistics, and max gain/loss for all tickers, 
    sessions, and compression factors, now including trend/reversion strategy type and MA/RSI periods.

    Parameters:
    - compressed_sessions: Dictionary of nested dictionaries containing trading data structured as:
      {ticker: {session: {compression_factor: DataFrame}}}
    - ticker_to_tick_size: Dictionary mapping tickers to tick sizes.
    - ticker_to_point_value: Dictionary mapping tickers to point values.
    - ma_name1, ma_name2: Moving average column names.
    - rsi_column: RSI column name.
    - strategy_type: "trend" or "reversion".

    Returns:
    - pd.DataFrame: A DataFrame with PnL, trade statistics, max gain/loss, session names, compression factors, 
      strategy type, and indicator periods.
    """
    def compute_stop_loss_metrics(dollar_pnl, dollar_max_loss, daily_stop_loss_dollars):
        hit = int(dollar_max_loss <= daily_stop_loss_dollars)
        cost = daily_stop_loss_dollars - dollar_pnl if dollar_pnl > daily_stop_loss_dollars and hit else 0.0
        gain = daily_stop_loss_dollars - dollar_pnl if dollar_pnl < daily_stop_loss_dollars and hit else 0.0
        return hit, cost, gain

    # Extract period lengths from indicator names (assumes format like "wma_5", "sma_10", "rsi_14")
    ma1_period = int(ma_name1.split('_')[-1])  # Extract last numeric part
    ma2_period = int(ma_name2.split('_')[-1])
    rsi_period = int(rsi_column.split('_')[-1])

    # Generate dynamic column names for PnL and trade metrics
    pnl_ma_col = f'pnl_{ma_name1}_{ma_name2}_{strategy_type}_{order_type}'
    pnl_rsi_col = f'pnl_{rsi_column}_{strategy_type}_{order_type}'
    cum_pnl_ma_col = f'cum_{pnl_ma_col}'
    cum_pnl_rsi_col = f'cum_{pnl_rsi_col}'
    cum_pnl_all_col = f'cum_pnl_all_{ma_name1}_{ma_name2}_{rsi_column}_{strategy_type}_{order_type}'
    ma_entry_price = f'entry_price_{ma_name1}_{ma_name2}_{strategy_type}_{order_type}'
    ma_exit_price = f'exit_price_{ma_name1}_{ma_name2}_{strategy_type}_{order_type}'
    rsi_entry_price = f'entry_price_{rsi_column}_{strategy_type}_{order_type}'
    rsi_exit_price = f'exit_price_{rsi_column}_{strategy_type}_{order_type}'
    ma_commission_col = f'commission_cost_{ma_name1}_{ma_name2}_{strategy_type}_{order_type}'
    rsi_commission_col = f'commission_cost_{rsi_column}_{strategy_type}_{order_type}'

    # Create a list to hold rows of the DataFrame
    pnl_rows = []

    # Iterate over tickers
    for ticker, sessions in compressed_sessions.items():
        tick_size = ticker_to_tick_size.get(ticker, "Unknown")
        point_value = ticker_to_point_value.get(ticker, 1)

        # Iterate over sessions
        for session_name, compressions in sessions.items():

            # Iterate over compression factors
            for compression_name, df in compressions.items():
                try:
                    # Extract compression factor (e.g., '5_minute_compression' → 5)
                    compression_factor = int(compression_name.split('_')[0])

                    # Calculate cumulative PnL and other statistics
                    ma_pnl = round(df[cum_pnl_ma_col].iloc[-1], 3)
                    rsi_pnl = round(df[cum_pnl_rsi_col].iloc[-1], 3)
                    total_pnl = round(df[cum_pnl_all_col].iloc[-1], 3)
                    close_price_diff = round(df["close"].iloc[-1] - df["close"].iloc[0], 3)
                    point_alpha = round(total_pnl - close_price_diff, 3)

                    # Retrieve total commission costs
                    ma_commission_total = round(df[ma_commission_col].iloc[-1], 3) if ma_commission_col in df else 0.0
                    rsi_commission_total = round(df[rsi_commission_col].iloc[-1], 3) if rsi_commission_col in df else 0.0
                    total_commission_cost = ma_commission_total + rsi_commission_total

                    # Calculate total dollar PnLs
                    # Dollar PnL adjusted for commissions (optional)
                    ma_dollar_pnl = ma_pnl * point_value
                    rsi_dollar_pnl = rsi_pnl * point_value
                    total_dollar_pnl = total_pnl * point_value
                    ma_dollar_pnl_sub_comms = ma_pnl * point_value - ma_commission_total
                    rsi_dollar_pnl_sub_comms = rsi_pnl * point_value - rsi_commission_total
                    total_dollar_pnl_sub_comms = total_pnl * point_value - total_commission_cost
                    close_price_dollar_diff = close_price_diff * point_value
                    dollar_alpha = point_alpha * point_value

                    # Count the number of trades for MA and RSI strategies
                    ma_trades = (df[ma_exit_price].notna().sum() + df[ma_entry_price].notna().sum()) / 2
                    rsi_trades = (df[rsi_exit_price].notna().sum() + df[rsi_entry_price].notna().sum()) / 2

                    # Calculate max gain and max loss for MA, RSI, and total strategies (point and dollar)
                    ma_max_gain = round(df[cum_pnl_ma_col].max(), 3)
                    ma_max_loss = round(df[cum_pnl_ma_col].min(), 3)
                    rsi_max_gain = round(df[cum_pnl_rsi_col].max(), 3)
                    rsi_max_loss = round(df[cum_pnl_rsi_col].min(), 3)
                    total_max_gain = round(df[cum_pnl_all_col].max(), 3)
                    total_max_loss = round(df[cum_pnl_all_col].min(), 3)

                    ma_max_dollar_gain = ma_max_gain * point_value
                    ma_max_dollar_loss = ma_max_loss * point_value
                    rsi_max_dollar_gain = rsi_max_gain * point_value
                    rsi_max_dollar_loss = rsi_max_loss * point_value
                    total_max_dollar_gain = total_max_gain * point_value
                    total_max_dollar_loss = total_max_loss * point_value

                    ma_gain_idx = df[cum_pnl_ma_col].idxmax()
                    ma_loss_idx = df[cum_pnl_ma_col].idxmin()
                    rsi_gain_idx = df[cum_pnl_rsi_col].idxmax()
                    rsi_loss_idx = df[cum_pnl_rsi_col].idxmin()
                    total_gain_idx = df[cum_pnl_all_col].idxmax()
                    total_loss_idx = df[cum_pnl_all_col].idxmin()

                    ma_max_dollar_gain_sub_comms = ma_max_dollar_gain - df[ma_commission_col].loc[ma_gain_idx]
                    ma_max_dollar_loss_sub_comms = ma_max_dollar_loss - df[ma_commission_col].loc[ma_loss_idx]
                    rsi_max_dollar_gain_sub_comms = rsi_max_dollar_gain - df[rsi_commission_col].loc[rsi_gain_idx]
                    rsi_max_dollar_loss_sub_comms = rsi_max_dollar_loss - df[rsi_commission_col].loc[rsi_loss_idx]
                    total_max_dollar_gain_sub_comms = total_max_dollar_gain - (
                        df[ma_commission_col].loc[total_gain_idx] + df[rsi_commission_col].loc[total_gain_idx]
                    )
                    total_max_dollar_loss_sub_comms = total_max_dollar_loss - (
                        df[ma_commission_col].loc[total_loss_idx] + df[rsi_commission_col].loc[total_loss_idx]
                    )

                    # Stop loss logic for total, MA, and RSI
                    stop_loss_hit_total, loss_prevention_cost_total, loss_prevention_gain_total = compute_stop_loss_metrics(
                        total_dollar_pnl, total_max_dollar_loss, daily_stop_loss_dollars)
                    stop_loss_hit_ma, loss_prevention_cost_ma, loss_prevention_gain_ma = compute_stop_loss_metrics(
                        ma_dollar_pnl, ma_max_dollar_loss, daily_stop_loss_dollars)
                    stop_loss_hit_rsi, loss_prevention_cost_rsi, loss_prevention_gain_rsi = compute_stop_loss_metrics(
                        rsi_dollar_pnl, rsi_max_dollar_loss, daily_stop_loss_dollars)
                    
                    # Compute the new stop-loss metrics for sub_comms
                    stop_loss_hit_total_sub_comms, loss_prevention_cost_total_sub_comms, loss_prevention_gain_total_sub_comms = compute_stop_loss_metrics(
                        total_dollar_pnl_sub_comms, total_max_dollar_loss_sub_comms, daily_stop_loss_dollars)
                    stop_loss_hit_ma_sub_comms, loss_prevention_cost_ma_sub_comms, loss_prevention_gain_ma_sub_comms = compute_stop_loss_metrics(
                        ma_dollar_pnl_sub_comms, ma_max_dollar_loss_sub_comms, daily_stop_loss_dollars)
                    stop_loss_hit_rsi_sub_comms, loss_prevention_cost_rsi_sub_comms, loss_prevention_gain_rsi_sub_comms = compute_stop_loss_metrics(
                        rsi_dollar_pnl_sub_comms, rsi_max_dollar_loss_sub_comms, daily_stop_loss_dollars)
                    
                    # Calculate adjusted total PnL after applying stop loss protection
                    total_dollar_pnl_stop_loss_adjusted = (total_dollar_pnl + loss_prevention_cost_total + loss_prevention_gain_total)
                    ma_dollar_pnl_stop_loss_adjusted = (ma_dollar_pnl + loss_prevention_cost_ma + loss_prevention_gain_ma)
                    rsi_dollar_pnl_stop_loss_adjusted = (rsi_dollar_pnl + loss_prevention_cost_rsi + loss_prevention_gain_rsi)
                    total_dollar_pnl_sub_comms_stop_loss_adjusted = (total_dollar_pnl_sub_comms + loss_prevention_cost_total_sub_comms + loss_prevention_gain_total_sub_comms)
                    ma_dollar_pnl_sub_comms_stop_loss_adjusted = (ma_dollar_pnl_sub_comms + loss_prevention_cost_ma_sub_comms + loss_prevention_gain_ma_sub_comms)
                    rsi_dollar_pnl_sub_comms_stop_loss_adjusted = (rsi_dollar_pnl_sub_comms + loss_prevention_cost_rsi_sub_comms + loss_prevention_gain_rsi_sub_comms)

                    # Append row with all calculated metrics
                    pnl_rows.append({
                        'ticker': ticker,
                        'session': session_name,
                        'compression_factor': compression_factor,

                        'ma_period1': ma1_period,
                        'ma_period2': ma2_period,
                        'rsi_period': rsi_period,

                        'strategy_type': strategy_type,
                        'order_type': order_type,

                        'ma_trades': ma_trades,
                        'rsi_trades': rsi_trades,

                        'total_point_pnl': total_pnl,
                        'ma_point_pnl': ma_pnl,
                        'rsi_point_pnl': rsi_pnl,

                        'total_dollar_pnl': total_dollar_pnl,
                        'ma_dollar_pnl': ma_dollar_pnl,
                        'rsi_dollar_pnl': rsi_dollar_pnl,

                        'ma_commission_cost': ma_commission_total,
                        'rsi_commission_cost': rsi_commission_total,
                        'total_commission_cost': total_commission_cost,

                        'total_dollar_pnl_sub_comms': total_dollar_pnl_sub_comms,
                        'ma_dollar_pnl_sub_comms': ma_dollar_pnl_sub_comms,
                        'rsi_dollar_pnl_sub_comms': rsi_dollar_pnl_sub_comms,

                        'ma_max_point_gain': ma_max_gain,
                        'ma_max_point_loss': ma_max_loss,
                        'rsi_max_point_gain': rsi_max_gain,
                        'rsi_max_point_loss': rsi_max_loss,
                        'total_max_point_gain': total_max_gain,
                        'total_max_point_loss': total_max_loss,

                        'ma_max_dollar_gain': ma_max_dollar_gain,
                        'ma_max_dollar_loss': ma_max_dollar_loss,
                        'rsi_max_dollar_gain': rsi_max_dollar_gain,
                        'rsi_max_dollar_loss': rsi_max_dollar_loss,
                        'total_max_dollar_gain': total_max_dollar_gain,
                        'total_max_dollar_loss': total_max_dollar_loss,

                        'ma_max_dollar_gain_sub_comms': ma_max_dollar_gain_sub_comms,
                        'ma_max_dollar_loss_sub_comms': ma_max_dollar_loss_sub_comms,
                        'rsi_max_dollar_gain_sub_comms': rsi_max_dollar_gain_sub_comms,
                        'rsi_max_dollar_loss_sub_comms': rsi_max_dollar_loss_sub_comms,
                        'total_max_dollar_gain_sub_comms': total_max_dollar_gain_sub_comms,
                        'total_max_dollar_loss_sub_comms': total_max_dollar_loss_sub_comms,

                        'close_price_diff': close_price_diff,
                        'point_alpha': point_alpha,
                        'close_price_dollar_diff': close_price_dollar_diff,
                        'dollar_alpha': dollar_alpha,
                        'tick_size': tick_size,

                        'daily_stop_loss_dollars': daily_stop_loss_dollars,

                        'session_stop_loss_hit_total': stop_loss_hit_total,
                        'session_stop_loss_hit_ma': stop_loss_hit_ma,
                        'session_stop_loss_hit_rsi': stop_loss_hit_rsi,
                        'loss_prevention_cost_total': loss_prevention_cost_total,
                        'loss_prevention_cost_ma': loss_prevention_cost_ma,
                        'loss_prevention_cost_rsi': loss_prevention_cost_rsi,                        
                        'loss_prevention_gain_total': loss_prevention_gain_total,
                        'loss_prevention_gain_ma': loss_prevention_gain_ma,
                        'loss_prevention_gain_rsi': loss_prevention_gain_rsi,

                        "session_stop_loss_hit_total_sub_comms": stop_loss_hit_total_sub_comms,
                        "session_stop_loss_hit_ma_sub_comms": stop_loss_hit_ma_sub_comms,
                        "session_stop_loss_hit_rsi_sub_comms": stop_loss_hit_rsi_sub_comms,
                        "loss_prevention_cost_total_sub_comms": loss_prevention_cost_total_sub_comms,
                        "loss_prevention_cost_ma_sub_comms": loss_prevention_cost_ma_sub_comms,
                        "loss_prevention_cost_rsi_sub_comms": loss_prevention_cost_rsi_sub_comms,
                        "loss_prevention_gain_total_sub_comms": loss_prevention_gain_total_sub_comms,
                        "loss_prevention_gain_ma_sub_comms": loss_prevention_gain_ma_sub_comms,
                        "loss_prevention_gain_rsi_sub_comms": loss_prevention_gain_rsi_sub_comms,

                        'total_dollar_pnl_stop_loss_adjusted': total_dollar_pnl_stop_loss_adjusted,
                        'ma_dollar_pnl_stop_loss_adjusted': ma_dollar_pnl_stop_loss_adjusted,
                        'rsi_dollar_pnl_stop_loss_adjusted': rsi_dollar_pnl_stop_loss_adjusted,
                        'total_dollar_pnl_sub_comms_stop_loss_adjusted': total_dollar_pnl_sub_comms_stop_loss_adjusted,
                        'ma_dollar_pnl_sub_comms_stop_loss_adjusted': ma_dollar_pnl_sub_comms_stop_loss_adjusted,
                        'rsi_dollar_pnl_sub_comms_stop_loss_adjusted': rsi_dollar_pnl_sub_comms_stop_loss_adjusted
                    })

                except Exception as e:
                    print(f"Error processing {ticker} - {session_name} - {compression_name}: {e}")

    # Convert the list of dictionaries into a DataFrame
    return pd.DataFrame(pnl_rows)

# Define your periods
ma_periods = [3, 5, 7, 9]
ma_combinations = [
    (f'wma_{wma}', f'sma_{sma}', f'rsi_{wma}')
    for wma in ma_periods
    for sma in ma_periods
    if wma <= sma
]
strategy_types = ["trend", "reversion"]
order_types = ["market", "limit"]

# Generate PnL DataFrame for each full combination
pnl_dataframes = [
    generate_pnl_dataframe(
        compressed_sessions,
        ticker_to_tick_size,
        ticker_to_point_value,
        ma_name1=sig_ma,
        ma_name2=con_ma,
        rsi_column=rsi_col,
        strategy_type=strategy,
        order_type=order_type
    )
    for (sig_ma, con_ma, rsi_col), strategy, order_type in product(ma_combinations, strategy_types, order_types)
]

# Combine into one big dataframe
final_pnl_df = pd.concat(pnl_dataframes, ignore_index=True)

def slice_final_pnl_df(df, ticker, compression_factor, ma_period1, ma_period2, rsi_period, strategy_type, order_type):
    return df[
        (df['ticker'] == ticker) &
        (df['compression_factor'] == compression_factor) &
        (df['ma_period1'] == ma_period1) &
        (df['ma_period2'] == ma_period2) &
        (df['rsi_period'] == rsi_period) &
        (df['strategy_type'] == strategy_type) &
        (df['order_type'] == order_type)
    ].reset_index(drop=True)

sliced_df = slice_final_pnl_df(
    final_pnl_df,
    ticker='/NQ',
    compression_factor=3,
    ma_period1=3,
    ma_period2=3,
    rsi_period=3,
    strategy_type='reversion',
    order_type='limit'
)
display(sliced_df)

def plot_trading_strategies_2(candles, 
                           ma_name1='wma_5', 
                           ma_name2='sma_5', 
                           rsi_column='rsi_5',  
                           figsize=(40, 20), 
                           font_size=10, 
                           ma_markersize=50, 
                           signal_markersize_y=400, 
                           signal_markersize_b=250,
                           strategy_type='trend',
                           order_type='market'
                           ):
    """
    Plots the minute_candles DataFrame with two selected moving averages and optional RSI.
    Also plots cumulative profit for MA and RSI strategies on a secondary axis.

    Parameters:
    - ma_name1 (str): The column name of the first moving average to plot.
    - ma_name2 (str): The column name of the second moving average to plot.
    - signal_column (str): The column name of the signal data (default is 'signal').
    - figsize (tuple): The size of the plot (width, height) in inches (default is (30, 20)).
    """

    try:
        # Clean the data to ensure numeric columns are valid
        columns_to_convert = ['open', 'high', 'low', 'close', 'volume', ma_name1, ma_name2, rsi_column] 
        candles[columns_to_convert] = candles[columns_to_convert].apply(pd.to_numeric, errors='coerce')

        # Generate dynamic column names for PnL and signals
        ma_entry_price = f'entry_price_{ma_name1}_{ma_name2}_{strategy_type}_{order_type}'
        ma_exit_price = f'exit_price_{ma_name1}_{ma_name2}_{strategy_type}_{order_type}'
        rsi_entry_price = f'entry_price_{rsi_column}_{strategy_type}_{order_type}'
        rsi_exit_price = f'exit_price_{rsi_column}_{strategy_type}_{order_type}'
        cum_pnl_ma_col = f'cum_pnl_{ma_name1}_{ma_name2}_{strategy_type}_{order_type}'
        cum_pnl_rsi_col = f'cum_pnl_{rsi_column}_{strategy_type}_{order_type}'
        cum_pnl_all_col = f'cum_pnl_all_{ma_name1}_{ma_name2}_{rsi_column}_{strategy_type}_{order_type}'
        stop_loss_ma = f'stop_loss_{ma_name1}_{ma_name2}_{strategy_type}_{order_type}'
        stop_loss_rsi = f'stop_loss_{rsi_column}_{strategy_type}_{order_type}'

        # Select the columns to plot
        plot_data = candles[['datetime', 'open', 'high', 'low', 'close', 'volume', 
                             ma_name1, ma_name2, rsi_column, 
                             ma_entry_price, ma_exit_price, rsi_entry_price, rsi_exit_price,
                             cum_pnl_ma_col, cum_pnl_rsi_col, cum_pnl_all_col,
                             stop_loss_ma, stop_loss_rsi]].copy() # here
        plot_data.set_index('datetime', inplace=True)

        # Create the additional plots for the moving averages and RSI, but only if they are warmed up
        add_plots = []

        # Check if the moving averages have enough valid data to plot
        if not candles[ma_name1].isnull().all() and not candles[ma_name2].isnull().all():
            add_plots.append(mpf.make_addplot(plot_data[ma_name1], color='yellow', type='scatter', marker='o', markersize=ma_markersize, label=f'{ma_name1}'))
            add_plots.append(mpf.make_addplot(plot_data[ma_name1], color='yellow', linestyle='-', width=0.75))
            add_plots.append(mpf.make_addplot(plot_data[ma_name2], color='purple', type='scatter', marker='o', markersize=ma_markersize, label=f'{ma_name2}'))
            add_plots.append(mpf.make_addplot(plot_data[ma_name2], color='purple', linestyle='-', width=0.75))
        else:
            print("Moving averages have not warmed up yet. Plotting without them.")

        # Check if the RSI has enough valid data to plot
        if not candles[rsi_column].isnull().all():
            add_plots.append(mpf.make_addplot(candles[rsi_column], panel=2, color='blue', type='scatter', marker='o', markersize=ma_markersize, label='RSI'))
            add_plots.append(mpf.make_addplot(candles[rsi_column], panel=2, color='blue', linestyle='-', width=0.75))
            add_plots.append(mpf.make_addplot(candles['trend_indicator'], panel=2, color='white', type='scatter', marker='o', markersize=ma_markersize, label='RSI'))
            add_plots.append(mpf.make_addplot(candles['trend_indicator'], panel=2, color='white', linestyle='-', width=0.75))
            add_plots.append(mpf.make_addplot(candles['hundred_line'], panel=2, color='red', linestyle=':', secondary_y=False))
            add_plots.append(mpf.make_addplot(candles['fifty_line'], panel=2, color='yellow', linestyle=':', secondary_y=False))
            add_plots.append(mpf.make_addplot(candles['zero_line'], panel=2, color='green', linestyle=':', secondary_y=False))
            add_plots.append(mpf.make_addplot(candles['trend_high_threshold'], panel=2, color='white', linestyle=':', secondary_y=False))
            add_plots.append(mpf.make_addplot(candles['trend_low_threshold'], panel=2, color='white', linestyle=':', secondary_y=False))
        else:
            print("RSI has not warmed up yet. Plotting without it.")

        # Add buy, sell, and neutral markers if signal_column exists. Eliminate the if else statement to revert to working order
        if ma_entry_price in candles.columns and ma_exit_price in candles.columns:
            add_plots.append(mpf.make_addplot(candles[ma_entry_price], type='scatter', marker='^', markersize=signal_markersize_y, color='yellow', panel=0, secondary_y=False))
            add_plots.append(mpf.make_addplot(candles[ma_exit_price], type='scatter', marker='o', markersize=signal_markersize_y, color='yellow', panel=0, secondary_y=False))
        else:
            print("Buy/Sell markers for MA strat have not warmed up yet. Plotting without them.")

        # Add buy, sell, and neutral markers for RSI strategy
        if rsi_entry_price in candles.columns and rsi_exit_price in candles.columns:
            add_plots.append(mpf.make_addplot(candles[rsi_entry_price], type='scatter', marker='^', markersize=signal_markersize_b, color='blue', panel=0, secondary_y=False))
            add_plots.append(mpf.make_addplot(candles[rsi_exit_price], type='scatter', marker='o', markersize=signal_markersize_b, color='blue', panel=0, secondary_y=False))
        else:
            print("Buy/Sell markers for RSI strat have not warmed up yet. Plotting without them.")

        # Add cumulative profit plots on a secondary y-axis with dynamic names
        add_plots.append(mpf.make_addplot(candles[cum_pnl_ma_col], panel=0, color='yellow', secondary_y=True, label=f'Cumulative PnL (MA: {ma_name1}_{ma_name2})', linestyle='-', width=1.25))
        add_plots.append(mpf.make_addplot(candles[cum_pnl_rsi_col], panel=0, color='blue', secondary_y=True, label=f'Cumulative PnL (RSI: {rsi_column})', linestyle='-', width=1.25))
        add_plots.append(mpf.make_addplot(candles[cum_pnl_all_col], panel=0, color='green', secondary_y=True, label=f'Cumulative PnL (Combined)', linestyle='-', width=1.25))

        # Add stop-loss markers (x) for both MA and RSI strategies
        # if 'stop_loss_ma' in candles.columns:
        add_plots.append(mpf.make_addplot(candles[stop_loss_ma], type='scatter', marker='x', markersize=100, color='yellow', panel=0, secondary_y=False))
        # else:
        #     print("There are no stop loss markers for MA strat")
        # if 'stop_loss_rsi' in candles.columns:
        add_plots.append(mpf.make_addplot(candles[stop_loss_rsi], type='scatter', marker='x', markersize=50, color='blue', panel=0, secondary_y=False))
        # else:
        #     print("There are no stop loss markers for RSI strat")

        # Add price action envelope as white lines
        if 'price_action_upper' in candles.columns and 'price_action_lower' in candles.columns:
            add_plots.append(mpf.make_addplot(candles['price_action_upper'], color='white', linestyle='-', width=0.5, label='Price Action Upper'))
            add_plots.append(mpf.make_addplot(candles['price_action_lower'], color='white', linestyle='-', width=0.5, label='Price Action Lower'))
            # add_plots.append(mpf.make_addplot(candles['ma_price_action_upper'], color='white', linestyle='-', width=0.5, label='Price Action Upper'))
            # add_plots.append(mpf.make_addplot(candles['ma_price_action_lower'], color='white', linestyle='-', width=0.5, label='Price Action Lower'))
        else:
            print("Price action envelope not calculating properly")

        # Create a custom style with a black background
        black_style = mpf.make_mpf_style(
            base_mpf_style='charles',  # Start with the 'charles' style and modify it
            facecolor='black',         # Set the background color to black
            gridcolor='black',          # Set the grid line color
            edgecolor='purple',          # Set the edge color for candles and boxes
            figcolor='black',          # Set the figure background color to black
            rc={'axes.labelcolor': 'yellow', 
                'xtick.color': 'yellow', 
                'ytick.color': 'yellow', 
                'axes.titlecolor': 'yellow',
                'font.size': font_size, 
                'axes.labelsize': font_size,
                'axes.titlesize': font_size,
                'xtick.labelsize': font_size,
                'ytick.labelsize': font_size,
                'legend.fontsize': font_size}  # Set tick and label colors to white
        )

        # Plot using mplfinance
        mpf.plot(plot_data, type='candle', style=black_style, 
                title='',
                ylabel='Price', 
                addplot=add_plots, 
                figsize=figsize,
                volume=True,
                panel_ratios=(8, 2),
                #  panel_ratios=(8, 2, 2),             
                tight_layout=True)
    except Exception as e:
        print(f"Something wrong in the plotting_moving_averages function: {e}")

def visualize_trades_2(candles, ticker_to_tick_size, ticker_to_point_value, ma_name1='wma_5', ma_name2='sma_5',
                        rsi_column='rsi_5', lower_slice=0, upper_slice=-1, compression_factor=1, session_key="", strategy_type="trend", order_type="market"):
    """
    Visualize trades and print summary statistics, including tick size for each ticker.

    Parameters:
    - candles (dict): Dictionary of DataFrames with candle data for each ticker.
    - ticker_to_tick_size (dict): Dictionary mapping tickers to their respective tick sizes.
    - ticker_to_point_value (dict): Dictionary mapping tickers to their respective point values.
    - ma_name1, ma_name2, rsi_column: Names of MA and RSI columns.
    - lower_slice, upper_slice: Range of rows to visualize.
    """
    # Generate dynamic column names for PnL and trade metrics
    pnl_ma_col = f'pnl_{ma_name1}_{ma_name2}_{strategy_type}_{order_type}'
    pnl_rsi_col = f'pnl_{rsi_column}_{strategy_type}_{order_type}'
    cum_pnl_ma_col = f'cum_{pnl_ma_col}'
    cum_pnl_rsi_col = f'cum_{pnl_rsi_col}'
    cum_pnl_all_col = f'cum_pnl_all_{ma_name1}_{ma_name2}_{rsi_column}_{strategy_type}_{order_type}'
    ma_entry_price = f'entry_price_{ma_name1}_{ma_name2}_{strategy_type}_{order_type}'
    ma_exit_price = f'exit_price_{ma_name1}_{ma_name2}_{strategy_type}_{order_type}'
    rsi_entry_price = f'entry_price_{rsi_column}_{strategy_type}_{order_type}'
    rsi_exit_price = f'exit_price_{rsi_column}_{strategy_type}_{order_type}'
    ma_commission_col = f'commission_cost_{ma_name1}_{ma_name2}_{strategy_type}_{order_type}'
    rsi_commission_col = f'commission_cost_{rsi_column}_{strategy_type}_{order_type}'

    # Variable to accumulate total dollar PnL
    total_dollar_pnl_sum = 0.0

    # Iterate through the candles dictionary
    for ticker, minute_candles_df in candles.items():
        # Create a copy of the DataFrame for the specified slice
        minute_candles_viz_1 = minute_candles_df[lower_slice:upper_slice].copy()
        tick_size = ticker_to_tick_size.get(ticker, "Unknown")  # Retrieve tick size or default to "Unknown"
        point_value = ticker_to_point_value.get(ticker, 1)  # Retrieve point value or default to 1

        try:
            # Plot moving averages
            plot_trading_strategies_2(
                minute_candles_viz_1,
                ma_name1=ma_name1, ma_name2=ma_name2, rsi_column=rsi_column,
                figsize=(40, 20), font_size=20,
                ma_markersize=50, signal_markersize_y=450, signal_markersize_b=300, strategy_type=strategy_type, order_type=order_type
            )
            
            # Calculate cumulative PnL and other statistics
            ma_pnl = round(minute_candles_df[cum_pnl_ma_col].iloc[-1], 3)
            rsi_pnl = round(minute_candles_df[cum_pnl_rsi_col].iloc[-1], 3)
            total_pnl = round(minute_candles_df[cum_pnl_all_col].iloc[-1], 3)
            close_price_diff = round(minute_candles_df["close"].iloc[-1] - minute_candles_df["close"].iloc[0], 3)
            point_alpha = round(total_pnl - close_price_diff, 3)

            # Retrieve total commission costs
            ma_commission_total = round(minute_candles_df[ma_commission_col].iloc[-1], 3) if ma_commission_col in minute_candles_df else 0.0
            rsi_commission_total = round(minute_candles_df[rsi_commission_col].iloc[-1], 3) if rsi_commission_col in minute_candles_df else 0.0
            total_commission_cost = ma_commission_total + rsi_commission_total

            # Calculate total dollar PnLs
            ma_dollar_pnl = (ma_pnl * point_value) - ma_commission_total
            rsi_dollar_pnl = (rsi_pnl * point_value) - rsi_commission_total
            total_dollar_pnl = (total_pnl * point_value) - total_commission_cost
            total_dollar_pnl_sum += total_dollar_pnl
            close_price_dollar_diff = close_price_diff * point_value
            dollar_alpha = (point_alpha * point_value) - total_commission_cost

            # Count the number of trades for MA and RSI strategies
            ma_trades = (minute_candles_df[ma_exit_price].notna().sum() + minute_candles_df[ma_entry_price].notna().sum())/2
            rsi_trades = (minute_candles_df[rsi_exit_price].notna().sum() + minute_candles_df[rsi_entry_price].notna().sum())/2
            total_trades = ma_trades + rsi_trades

            # Calculate max gain and max loss for MA, RSI, and total strategies
            ma_max_gain = round(minute_candles_df[cum_pnl_ma_col].max(), 3)
            ma_max_loss = round(minute_candles_df[cum_pnl_ma_col].min(), 3)
            rsi_max_gain = round(minute_candles_df[cum_pnl_rsi_col].max(), 3)
            rsi_max_loss = round(minute_candles_df[cum_pnl_rsi_col].min(), 3)
            total_max_gain = round(minute_candles_df[cum_pnl_all_col].max(), 3)
            total_max_loss = round(minute_candles_df[cum_pnl_all_col].min(), 3)
            ma_max_dollar_gain = (ma_max_gain * point_value) - ma_commission_total
            ma_max_dollar_loss = (ma_max_loss * point_value) - ma_commission_total
            rsi_max_dollar_gain = (rsi_max_gain * point_value) - rsi_commission_total
            rsi_max_dollar_loss = (rsi_max_loss * point_value) - rsi_commission_total
            total_max_dollar_gain = (total_max_gain * point_value) - total_commission_cost
            total_max_dollar_loss = (total_max_loss * point_value) - total_commission_cost

            # Print detailed statistics for the ticker
            print(f"{ticker}: {compression_factor}-Minute Compression Factor, Strategy Type: {strategy_type}, Order Type: {order_type}, Total PnL: {total_pnl:.2f}pt/${total_dollar_pnl:.2f}, {len(minute_candles_df)} rows, "
                f"Session: {session_key}, {ma_name1}, {ma_name2}, {rsi_column}, "                
                f"Total trades: {total_trades}, Total Commission Cost: ${total_commission_cost:.2f}, Total Max Gain: {total_max_gain:.2f}pt/${total_max_dollar_gain}, Total Max Loss: {total_max_loss:.2f}pt/${total_max_dollar_loss}, "
                f"MA PnL: {ma_pnl:.2f}pt/${ma_dollar_pnl:.2f},MA trades: {ma_trades}, MA Commission Cost: ${ma_commission_total:.2f}, MA Max Gain: {ma_max_gain:.2f}pt/${ma_max_dollar_gain}, MA Max Loss: {ma_max_loss:.2f}pt/${ma_max_dollar_loss}, "
                f"RSI PnL: {rsi_pnl:.2f}pt/${rsi_dollar_pnl:.2f}, RSI trades: {rsi_trades}, RSI Commission Cost: ${rsi_commission_total:.2f}, RSI Max Gain: {rsi_max_gain:.2f}pt/${rsi_max_dollar_gain}, RSI Max Loss: {rsi_max_loss:.2f}pt/${rsi_max_dollar_loss}, "
                f"Close Price Difference: {close_price_diff:.2f}pt/${close_price_dollar_diff:.2f}, Alpha: {point_alpha:.2f}pt/${dollar_alpha:.2f}, Tick Size: {tick_size}, "                
            )
        except Exception as e:
            # Handle any errors that occur during the plotting
            print(f"Error in visualize_trades_2 for {ticker}: {e}")

    # Return the total PnL for this ticker so it can be aggregated
    return total_dollar_pnl_sum

chosen_session_index = 50  # 0 for first session, 1 for second, etc.
chosen_compression_factor = 3  # Which compression factor to use
chosen_compression_key = f'{chosen_compression_factor}_minute_compression' # Build the compression string dynamically
upper_slice = -1  # Slice to the end of the DataFrame
lower_slice = 0  # Slice from the beginning of the DataFrame
_wma_rsi = '3'
_sma = '5'
wma = f'wma_{_wma_rsi}'
sma = f'sma_{_sma}'
rsi = f'rsi_{_wma_rsi}'

# Separate tracking for trend and reversion PnL
total_pnl_across_tickers_trend = 0.0
total_pnl_across_tickers_reversion = 0.0

for ticker, sessions in compressed_sessions.items():
    # Get sorted session keys to create a numerical map
    session_keys = list(sessions.keys())

    if chosen_session_index >= len(session_keys):
        print(f"Ticker {ticker} only has {len(session_keys)} sessions. Skipping...")
        continue

    # Get the actual session key from numerical index
    chosen_session = session_keys[chosen_session_index]

    # Check if the chosen compression exists within this session
    if chosen_compression_key not in sessions[chosen_session]:
        print(f"Ticker {ticker}, session {chosen_session} does not have compression {chosen_compression_key}. Skipping...")
        continue

    # Grab the DataFrame for the chosen session and compression
    df = sessions[chosen_session][chosen_compression_key]

    for strategy_type in ["trend", "reversion"]: # Ensure both strategies are visualized sequentially
        # print(f"\nVisualizing {strategy_type.upper()} strategy for {ticker} - Session: {chosen_session}")

        for order_type in ["market", "limit"]: # Ensure both order types are visualized sequentially

            ticker_pnl = visualize_trades_2(
                candles={ticker: df},
                ticker_to_tick_size=ticker_to_tick_size,
                ticker_to_point_value=ticker_to_point_value,
                ma_name1=wma,
                ma_name2=sma,
                rsi_column=rsi,
                lower_slice=lower_slice,
                upper_slice=upper_slice,
                compression_factor=chosen_compression_factor,
                session_key=chosen_session, # Pass the session key
                strategy_type=strategy_type,
                order_type=order_type
            )

            # Store PnL separately for each strategy
            if strategy_type == "trend":
                total_pnl_across_tickers_trend += ticker_pnl
            else:
                total_pnl_across_tickers_reversion += ticker_pnl

    # Print final aggregated PnL for each strategy type
    print(f"\nTotal Dollar PnL Across All Tickers for Trend Strategy (Session Index {chosen_session_index}): {total_pnl_across_tickers_trend:.2f}")
    print(f"Total Dollar PnL Across All Tickers for Reversion Strategy (Session Index {chosen_session_index}): {total_pnl_across_tickers_reversion:.2f}")

# Group and aggregate
grouped = final_pnl_df.groupby([
    'ticker', 
    'session', # comment out here to exclude session
    'compression_factor', 
    'strategy_type', 
    'order_type', 
    'ma_period1', 
    'ma_period2', 
    'rsi_period'
])

# Aggregate total, MA, and RSI PnL (sum and mean)
agg_df = grouped[['total_dollar_pnl', 'total_dollar_pnl_stop_loss_adjusted', 'total_dollar_pnl_sub_comms_stop_loss_adjusted', 
                  'ma_dollar_pnl', 'ma_dollar_pnl_stop_loss_adjusted', 'ma_dollar_pnl_sub_comms_stop_loss_adjusted',
                  'rsi_dollar_pnl', 'rsi_dollar_pnl_stop_loss_adjusted', 'rsi_dollar_pnl_sub_comms_stop_loss_adjusted',
                  ]].agg(['sum', 'mean']).reset_index()

# 

# Flatten MultiIndex columns
agg_df.columns = [
    'ticker', 
    'session', # comment out here to exclude session
    'compression_factor', 'strategy_type', 'order_type', 
    'ma_period1', 'ma_period2', 'rsi_period', 
    'total_pnl_sum', 'total_pnl_mean', 'total_dollar_pnl_stop_loss_adjusted_sum', 'total_dollar_pnl_stop_loss_adjusted_mean', 'total_dollar_pnl_sub_comms_stop_loss_adjusted_sum', 'total_dollar_pnl_sub_comms_stop_loss_adjusted_mean',
    'ma_pnl_sum', 'ma_pnl_mean', 'ma_dollar_pnl_stop_loss_adjusted_sum', 'ma_dollar_pnl_stop_loss_adjusted_mean', 'ma_dollar_pnl_sub_comms_stop_loss_adjusted_sum', 'ma_dollar_pnl_sub_comms_stop_loss_adjusted_mean',
    'rsi_pnl_sum', 'rsi_pnl_mean', 'rsi_dollar_pnl_stop_loss_adjusted_sum', 'rsi_dollar_pnl_stop_loss_adjusted_mean', 'rsi_dollar_pnl_sub_comms_stop_loss_adjusted_sum', 'rsi_dollar_pnl_sub_comms_stop_loss_adjusted_mean'
]

# Top 10 best-performing parameter combinations
top_combinations = agg_df.sort_values('total_dollar_pnl_sub_comms_stop_loss_adjusted_sum', ascending=False)
print("If sums and means are printing the same, you are including session in the groupby. To exclude session, comment out the designated lines in the `grouped` variable and where `agg_df.columns` is created.")
display(top_combinations[:60])  # Show top 60

def plot_pnl_distributions(
    final_pnl_df,
    strategy_type="trend",
    order_type="market",
    ma_period1=None,
    ma_period2=None,
    rsi_period=None,
    ma_dollar_pnl = "ma_dollar_pnl",
    total_dollar_pnl = "total_dollar_pnl",
    rsi_dollar_pnl = "rsi_dollar_pnl",
):
    """
    Plots the PnL distributions for a single ticker, strategy type, and order type.
    Assumes the input DataFrame is already filtered to one ticker.
    """

    # Filter for the selected MA/RSI period combination if provided
    if ma_period1 is not None and ma_period2 is not None and rsi_period is not None:
        final_pnl_df = final_pnl_df[
            (final_pnl_df["ma_period1"] == ma_period1) &
            (final_pnl_df["ma_period2"] == ma_period2) &
            (final_pnl_df["rsi_period"] == rsi_period)
        ]

    if final_pnl_df.empty:
        print("The DataFrame is empty. No data to visualize.")
        return

    if "compression_factor" not in final_pnl_df.columns:
        print("Missing Compression_Factor column.")
        return

    ticker = final_pnl_df["ticker"].iloc[0]  # For title display

    # Extract unique compression factors
    compression_factors = sorted(final_pnl_df["compression_factor"].unique())

    # Prepare data for violin plots
    ma_pnl_data = [final_pnl_df[final_pnl_df["compression_factor"] == cf][ma_dollar_pnl].values for cf in compression_factors]
    total_pnl_data = [final_pnl_df[final_pnl_df["compression_factor"] == cf][total_dollar_pnl].values for cf in compression_factors]
    rsi_pnl_data = [final_pnl_df[final_pnl_df["compression_factor"] == cf][rsi_dollar_pnl].values for cf in compression_factors]

    plt.figure(figsize=(20, 12))

    # Position offsets
    ma_positions = [cf - 0.5 for cf in compression_factors]
    total_positions = compression_factors
    rsi_positions = [cf + 0.5 for cf in compression_factors]

    def plot_extra_stats(data, positions, color):
        for i, pos in enumerate(positions):
            if len(data[i]) == 0:
                continue
            mean_val = np.mean(data[i])
            q25, q75 = np.percentile(data[i], [25, 75])
            p5, p95 = np.percentile(data[i], [5, 95])
            plt.scatter(pos, mean_val, color='black', s=80, zorder=3)
            plt.plot([pos, pos], [q25, q75], color=color, linewidth=4)
            plt.plot([pos, pos], [p5, p95], color=color, linewidth=1, linestyle='--')

    # Plot MA
    violin_ma = plt.violinplot(ma_pnl_data, positions=ma_positions, showmedians=True)
    for vp in violin_ma['bodies']:
        vp.set_facecolor('yellow')
        vp.set_edgecolor('black')
        vp.set_alpha(0.5)
    violin_ma['cmedians'].set_color('yellow')
    for part in ['cmins', 'cmaxes', 'cbars']:
        violin_ma[part].set_color('yellow')
    plot_extra_stats(ma_pnl_data, ma_positions, 'yellow')

    # Plot Total
    violin_total = plt.violinplot(total_pnl_data, positions=total_positions, showmedians=True)
    for vp in violin_total['bodies']:
        vp.set_facecolor('green')
        vp.set_edgecolor('black')
        vp.set_alpha(0.5)
    violin_total['cmedians'].set_color('green')
    for part in ['cmins', 'cmaxes', 'cbars']:
        violin_total[part].set_color('green')
    plot_extra_stats(total_pnl_data, total_positions, 'green')

    # Plot RSI
    violin_rsi = plt.violinplot(rsi_pnl_data, positions=rsi_positions, showmedians=True)
    for vp in violin_rsi['bodies']:
        vp.set_facecolor('blue')
        vp.set_edgecolor('black')
        vp.set_alpha(0.5)
    violin_rsi['cmedians'].set_color('blue')
    for part in ['cmins', 'cmaxes', 'cbars']:
        violin_rsi[part].set_color('blue')
    plot_extra_stats(rsi_pnl_data, rsi_positions, 'blue')

    plt.axhline(0, color='yellow', linestyle='--', linewidth=1)
    plt.xlabel("Compression Factor (Minutes)")
    plt.ylabel("Dollar PnL")
    plt.title(f"PnL Distributions vs Compression Factor for {ticker} ({strategy_type}, {order_type})")
    plt.xticks(compression_factors, labels=[str(cf) for cf in compression_factors])
    plt.show()

    # Summary Stats
    print(f"Ticker: {ticker} - Aggregate PnL Metrics by Compression Factor ({strategy_type}, {order_type}):\n")
    for cf in compression_factors:
        cf_ma_pnl = final_pnl_df[final_pnl_df["compression_factor"] == cf][ma_dollar_pnl]
        cf_total_pnl = final_pnl_df[final_pnl_df["compression_factor"] == cf][total_dollar_pnl]
        cf_rsi_pnl = final_pnl_df[final_pnl_df["compression_factor"] == cf][rsi_dollar_pnl]
        print(f"  Compression Factor {cf}:")
        print(f"    MA PnL    -> Sum: {cf_ma_pnl.sum():.2f}, Mean: {cf_ma_pnl.mean():.2f}")
        print(f"    Total PnL -> Sum: {cf_total_pnl.sum():.2f}, Mean: {cf_total_pnl.mean():.2f}")
        print(f"    RSI PnL   -> Sum: {cf_rsi_pnl.sum():.2f}, Mean: {cf_rsi_pnl.mean():.2f}")
        print("-" * 50)
    print("\n" + "=" * 60 + "\n")

def plot_all_pnl_distributions(final_pnl_df, 
                               ma_period1=None, 
                               ma_period2=None, 
                               rsi_period=None,
                               ma_dollar_pnl='ma_dollar_pnl',
                               total_dollar_pnl='total_dollar_pnl',
                               rsi_dollar_pnl='rsi_dollar_pnl'):
    """
    Plots PnL distributions for each ticker, strategy, order type, and selected MA/RSI period combination.
    """
    if final_pnl_df.empty:
        print("The DataFrame is empty. Nothing to plot.")
        return

    required_columns = {"ticker", "strategy_type", "order_type", "compression_factor"}
    if not required_columns.issubset(final_pnl_df.columns):
        print(f"Missing one or more required columns: {required_columns}")
        return

    unique_tickers = final_pnl_df["ticker"].unique()

    for ticker in unique_tickers:
        ticker_df = final_pnl_df[final_pnl_df["ticker"] == ticker]

        for strategy_type in ["trend", "reversion"]:
            for order_type in ["market", "limit"]:
                subset = ticker_df[
                    (ticker_df["strategy_type"] == strategy_type) &
                    (ticker_df["order_type"] == order_type) &
                    (ticker_df["ma_period1"] == ma_period1) &
                    (ticker_df["ma_period2"] == ma_period2) &
                    (ticker_df["rsi_period"] == rsi_period)
                ]
                if not subset.empty:
                    plot_pnl_distributions(
                        subset,
                        strategy_type=strategy_type,
                        order_type=order_type,
                        ma_period1=ma_period1,
                        ma_period2=ma_period2,
                        rsi_period=rsi_period,
                        ma_dollar_pnl=ma_dollar_pnl,
                        total_dollar_pnl=total_dollar_pnl,
                        rsi_dollar_pnl=rsi_dollar_pnl
                    )

# Trade logic after implementing market vs limit order iteration and before iterating through directional bias
###############################################################################################################################







































