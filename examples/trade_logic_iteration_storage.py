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
    # Initialize signal and position state columns
    candles[f'signal_{ma_name1}_{ma_name2}'] = 0
    candles[f'signal_{rsi_column}'] = 0
    candles['ma_position_open'] = False
    candles['rsi_position_open'] = False

    # Moving average signal and position state generation
    ma_position_open = False
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
    candles[f'signal_{ma_name1}_{ma_name2}'] = ma_signals
    candles['ma_position_open'] = ma_positions

    # RSI signal and position state generation
    rsi_position_open = False
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
    candles[f'signal_{rsi_column}'] = rsi_signals
    candles['rsi_position_open'] = rsi_positions

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
    
    # Update position open columns
    candles['ma_position_open'] = candles[ma_signal_column] == 1
    candles['rsi_position_open'] = candles[rsi_signal_column] == 1
    
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
    # Initialize entry price columns
    candles['ma_entry_price'] = None
    candles['rsi_entry_price'] = None

    # Get tick size
    tick_size = ticker_to_tick_size.get(ticker, 0) if ticker_to_tick_size else 0

    # Moving Average Strategy
    ma_signals = candles[f'signal_{ma_name1}_{ma_name2}']
    ma_close_prices = candles['close']
    ma_entry_mask = (ma_signals == 1) & (ma_signals.shift(1) != 1)
    candles.loc[ma_entry_mask, 'ma_entry_price'] = ma_close_prices[ma_entry_mask] + tick_size

    # RSI Strategy
    rsi_signals = candles[f'signal_{rsi_column}']
    rsi_entry_mask = (rsi_signals == 1) & (rsi_signals.shift(1) != 1)
    candles.loc[rsi_entry_mask, 'rsi_entry_price'] = ma_close_prices[rsi_entry_mask] + tick_size

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
    # Initialize exit price columns
    candles['ma_exit_price'] = None
    candles['rsi_exit_price'] = None

    # Get tick size
    tick_size = ticker_to_tick_size.get(ticker, 0) if ticker_to_tick_size else 0

    # Moving Average Strategy
    ma_signals = candles[f'signal_{ma_name1}_{ma_name2}']
    ma_close_prices = candles['close']
    ma_exit_mask = (ma_signals == 0) & (ma_signals.shift(1) == 1)
    candles.loc[ma_exit_mask, 'ma_exit_price'] = ma_close_prices[ma_exit_mask] - tick_size

    # RSI Strategy
    rsi_signals = candles[f'signal_{rsi_column}']
    rsi_exit_mask = (rsi_signals == 0) & (rsi_signals.shift(1) == 1)
    candles.loc[rsi_exit_mask, 'rsi_exit_price'] = ma_close_prices[rsi_exit_mask] - tick_size

    return candles

def calculate_stop_losses(candles):
    """ Vectorized without iterrows
    Calculate stop loss levels for MA and RSI strategies and ensure they persist while positions are open.

    Parameters:
    - candles (pd.DataFrame): The DataFrame containing candle data.

    Returns:
    - candles (pd.DataFrame): The DataFrame with stop loss columns updated.
    """
    # Initialize stop loss columns
    candles['stop_loss_ma'] = None
    candles['stop_loss_rsi'] = None
    # candles['ma_position_open'] = False
    # candles['rsi_position_open'] = False

    # Moving Average Stop Loss
    ma_entry_mask = candles['ma_entry_price'].notnull()
    ma_exit_mask = candles['ma_exit_price'].notnull() 
    
    # Set stop loss where positions open
    candles.loc[ma_entry_mask, 'stop_loss_ma'] = candles['ma_entry_price'] - candles['candle_span_max']
    # candles.loc[ma_entry_mask, 'ma_position_open'] = True

    # Reset stop loss and close position where positions close
    candles.loc[ma_exit_mask, 'stop_loss_ma'] = None
    # candles.loc[ma_exit_mask, 'ma_position_open'] = False

    # RSI Stop Loss
    rsi_entry_mask = candles['rsi_entry_price'].notnull()
    rsi_exit_mask = candles['rsi_exit_price'].notnull()
    
    # Set stop loss where positions open
    candles.loc[rsi_entry_mask, 'stop_loss_rsi'] = candles['rsi_entry_price'] - candles['candle_span_max']
    # candles.loc[rsi_entry_mask, 'rsi_position_open'] = True

    # Reset stop loss and close position where positions close
    candles.loc[rsi_exit_mask, 'stop_loss_rsi'] = None
    # candles.loc[rsi_exit_mask, 'rsi_position_open'] = False

    # Forward-fill stop loss and position open status for both strategies
    candles['stop_loss_ma'] = candles['stop_loss_ma'].ffill()
    candles['stop_loss_rsi'] = candles['stop_loss_rsi'].ffill()
    # candles['ma_position_open'] = candles['ma_position_open'].ffill().fillna(False)
    # candles['rsi_position_open'] = candles['rsi_position_open'].ffill().fillna(False)

    return candles

def track_stop_loss_hits(candles, ma_name1='wma_5', ma_name2='sma_5', rsi_column='rsi_5', ticker_to_tick_size=None, ticker=None):
    """ Vectorized without iterrows
    Track whether stop losses have been hit for MA and RSI strategies, and update position open status and exit prices.

    Parameters:
    - candles (pd.DataFrame): The DataFrame containing candle data.
    - ma_name1 (str): Column name for the first moving average.
    - ma_name2 (str): Column name for the second moving average.
    - rsi_column (str): Column name for the RSI indicator.
    - ticker_to_tick_size (dict): Mapping of tickers to their tick sizes.
    - ticker (str): The ticker for which the tick size applies.

    Returns:
    - candles (pd.DataFrame): The DataFrame with stop loss hit flags, adjusted signals, and updated exit prices.
    """
    # Initialize stop loss hit columns
    candles['ma_stop_loss_hit'] = False
    candles['rsi_stop_loss_hit'] = False

    # Get tick size
    tick_size = ticker_to_tick_size.get(ticker, 0) if ticker_to_tick_size else 0

    # Ensure stop loss values are numerical (convert None to NaN)
    candles['stop_loss_ma'] = candles['stop_loss_ma'].fillna(float('inf'))
    candles['stop_loss_rsi'] = candles['stop_loss_rsi'].fillna(float('inf'))

    # Moving Average Stop Loss Hit Logic
    ma_stop_loss_hit = (candles['stop_loss_ma'].notnull()) & (candles['close'] <= candles['stop_loss_ma']) & candles['ma_position_open']
    candles.loc[ma_stop_loss_hit, 'ma_stop_loss_hit'] = True
    # candles.loc[ma_stop_loss_hit, 'ma_exit_price'] = candles['close'] - tick_size
    # candles.loc[ma_stop_loss_hit, f'signal_{ma_name1}_{ma_name2}'] = 0
    # candles.loc[ma_stop_loss_hit, 'ma_position_open'] = False
    # candles.loc[ma_stop_loss_hit, 'stop_loss_ma'] = None

    # RSI Stop Loss Hit Logic
    rsi_stop_loss_hit = (candles['stop_loss_rsi'].notnull()) & (candles['close'] <= candles['stop_loss_rsi']) & candles['rsi_position_open']
    candles.loc[rsi_stop_loss_hit, 'rsi_stop_loss_hit'] = True
    # candles.loc[rsi_stop_loss_hit, 'rsi_exit_price'] = candles['close'] - tick_size
    # candles.loc[rsi_stop_loss_hit, f'signal_{rsi_column}'] = 0
    # candles.loc[rsi_stop_loss_hit, 'rsi_position_open'] = False
    # candles.loc[rsi_stop_loss_hit, 'stop_loss_rsi'] = None

    return candles

def adjust_signals_for_stop_loss(candles, ma_name1='wma_5', ma_name2='sma_5', rsi_column='rsi_5'):
    """
    Adjust MA signals to 0 where stop loss has been hit.

    Parameters:
    - candles (pd.DataFrame): The DataFrame containing candle data.
    - ma_name1 (str): Column name for the first moving average.
    - ma_name2 (str): Column name for the second moving average.
    - rsi_column (str): Column name for the RSI indicator.

    Returns:
    - candles (pd.DataFrame): The DataFrame with adjusted MA signals.
    """
    # Rewrite signal to 0 where stop loss has been hit
    candles.loc[candles['ma_stop_loss_hit'], f'signal_{ma_name1}_{ma_name2}'] = 0
    candles.loc[candles['rsi_stop_loss_hit'], f'signal_{rsi_column}']
    return candles

def update_stop_loss(candles, ma_name1='wma_5', ma_name2='sma_5', rsi_column='rsi_5'):
    """
    Set 'stop_loss_ma' to NaN where the signal column is 0.

    Parameters:
    - candles (pd.DataFrame): The DataFrame containing candle data.
    - ma_name1 (str): Column name for the first moving average.
    - ma_name2 (str): Column name for the second moving average.
    - rsi_column (str): Column name for the RSI indicator.

    Returns:
    - pd.DataFrame: The updated DataFrame with modified 'stop_loss_ma' values.
    """
    # Set 'stop_loss_ma' to NaN where the MA signal is 0
    signal_column_ma = f'signal_{ma_name1}_{ma_name2}'
    candles.loc[candles[signal_column_ma] == 0, 'stop_loss_ma'] = float('nan')

    # Set 'stop_loss_rsi' to NaN where the RSI signal is 0
    signal_column_rsi = f'signal_{rsi_column}'
    candles.loc[candles[signal_column_rsi] == 0, 'stop_loss_rsi'] = float('nan')

    return candles

def calculate_profit_loss_1(candles, multiplier=1, ma_name1='wma_5', ma_name2='sma_5', rsi_column='rsi_5'):

    """ Vectorized without iterrows
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
    cum_pnl_all_col = f'cum_pnl_all_{ma_name1}_{ma_name2}_{rsi_column}'

    # Initialize PnL columns
    candles[pnl_ma_col] = 0.0
    candles[pnl_rsi_col] = 0.0

    # Moving Average Strategy PnL Calculation
    ma_entry_indices = candles.index[candles['ma_entry_price'].notnull()]
    ma_exit_indices = candles.index[candles['ma_exit_price'].notnull()]

    # Pair up entry and exit prices
    valid_pairs = min(len(ma_entry_indices), len(ma_exit_indices))
    ma_entry_prices = candles.loc[ma_entry_indices[:valid_pairs], 'ma_entry_price'].values
    ma_exit_prices = candles.loc[ma_exit_indices[:valid_pairs], 'ma_exit_price'].values

    # Calculate PnL for MA strategy
    ma_pnl = (ma_exit_prices - ma_entry_prices) * multiplier
    candles.loc[ma_exit_indices[:valid_pairs], pnl_ma_col] = ma_pnl

    # RSI Strategy PnL Calculation
    rsi_entry_indices = candles.index[candles['rsi_entry_price'].notnull()]
    rsi_exit_indices = candles.index[candles['rsi_exit_price'].notnull()]

    # Pair up entry and exit prices
    valid_pairs = min(len(rsi_entry_indices), len(rsi_exit_indices))
    rsi_entry_prices = candles.loc[rsi_entry_indices[:valid_pairs], 'rsi_entry_price'].values
    rsi_exit_prices = candles.loc[rsi_exit_indices[:valid_pairs], 'rsi_exit_price'].values

    # Calculate PnL for RSI strategy
    rsi_pnl = (rsi_exit_prices - rsi_entry_prices) * multiplier
    candles.loc[rsi_exit_indices[:valid_pairs], pnl_rsi_col] = rsi_pnl

    # Calculate cumulative PnL for both strategies
    candles[cum_pnl_ma_col] = candles[pnl_ma_col].cumsum()
    candles[cum_pnl_rsi_col] = candles[pnl_rsi_col].cumsum()

    # Calculate combined cumulative PnL
    candles[cum_pnl_all_col] = candles[cum_pnl_ma_col] + candles[cum_pnl_rsi_col]

    return candles

for ticker, df in minute_candles_1.items():  # Generate state 2 signals
    for sig_ma, con_ma, rsi_col in ma_combinations:
        minute_candles_1[ticker] = generate_trading_signals(
            df,
            ma_name1=sig_ma,
            ma_name2=con_ma,
            rsi_column=rsi_col
        )
        
for ticker, df in minute_candles_1.items():  # Update position_open columns to be 1:1 verbal boolean with the signal
    for sig_ma, con_ma, rsi_col in ma_combinations:
        minute_candles_1[ticker] = update_position_open(
            df,
            ma_name1=sig_ma,
            ma_name2=con_ma,
            rsi_column=rsi_col
        )

for ticker, df in minute_candles_1.items():  # Determine entry prices for each ticker
    for sig_ma, con_ma, rsi_col in ma_combinations:
        minute_candles_1[ticker] = determine_entry_prices(
            df,
            ma_name1=sig_ma,
            ma_name2=con_ma,
            rsi_column=rsi_col,
            ticker_to_tick_size=ticker_to_tick_size,
            ticker=ticker
        )

for ticker, df in minute_candles_1.items():  # Determine exit prices for each ticker
    for sig_ma, con_ma, rsi_col in ma_combinations:
        minute_candles_1[ticker] = determine_exit_prices(
            df,
            ma_name1=sig_ma,
            ma_name2=con_ma,
            rsi_column=rsi_col,
            ticker_to_tick_size=ticker_to_tick_size,
            ticker=ticker
        )

for ticker, df in minute_candles_1.items():  # Stop loss calculation
    for sig_ma, con_ma, rsi_col in ma_combinations:
        minute_candles_1[ticker] = calculate_stop_losses(
            df,
            ma_name1=sig_ma,
            ma_name2=con_ma,
            rsi_column=rsi_col
        )

for ticker, df in minute_candles_1.items():  # Track stop loss hits
    for sig_ma, con_ma, rsi_col in ma_combinations:
        minute_candles_1[ticker] = track_stop_loss_hits(
            df,
            ma_name1=sig_ma,
            ma_name2=con_ma,
            rsi_column=rsi_col,
            ticker_to_tick_size=ticker_to_tick_size,
            ticker=ticker
        )

for ticker, df in minute_candles_1.items():  # Adjust signals from stop loss hits
    for sig_ma, con_ma, rsi_col in ma_combinations:
        minute_candles_1[ticker] = adjust_signals_for_stop_loss(
            df,
            ma_name1=sig_ma,
            ma_name2=con_ma,
            rsi_column=rsi_col
        )

for ticker, df in minute_candles_1.items():  # Re-update position_open column to be 1:1 verbal boolean with the signal after stop loss hits
    for sig_ma, con_ma, rsi_col in ma_combinations:
        minute_candles_1[ticker] = update_position_open(
            df,
            ma_name1=sig_ma,
            ma_name2=con_ma,
            rsi_column=rsi_col
        )
    
for ticker, df in minute_candles_1.items():  # Re-determine entry prices for each ticker after stop loss hits
    for sig_ma, con_ma, rsi_col in ma_combinations:
        minute_candles_1[ticker] = determine_entry_prices(
            df,
            ma_name1=sig_ma,
            ma_name2=con_ma,
            rsi_column=rsi_col,
            ticker_to_tick_size=ticker_to_tick_size,
            ticker=ticker
        )

for ticker, df in minute_candles_1.items():  # Re-determine exit prices for each ticker after stop loss hits
    for sig_ma, con_ma, rsi_col in ma_combinations:
        minute_candles_1[ticker] = determine_exit_prices(
            df,
            ma_name1=sig_ma,
            ma_name2=con_ma,
            rsi_column=rsi_col,
            ticker_to_tick_size=ticker_to_tick_size,
            ticker=ticker
        )

for ticker, df in minute_candles_1.items():  # Update stop loss levels after stop loss hits
    for sig_ma, con_ma, rsi_col in ma_combinations:
        minute_candles_1[ticker] = update_stop_loss(
            df,
            ma_name1=sig_ma,
            ma_name2=con_ma,
            rsi_column=rsi_col
        )
    
for ticker, df in minute_candles_1.items():  # Calculate profit/loss for each ticker's DataFrame
    for sig_ma, con_ma, rsi_col in ma_combinations:
        minute_candles_1[ticker] = calculate_profit_loss_1(
            df,
            multiplier=1,
            ma_name1=sig_ma,
            ma_name2=con_ma,
            rsi_column=rsi_col
        )

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
