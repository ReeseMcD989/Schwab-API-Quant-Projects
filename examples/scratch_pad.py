

# Make daily or 30 min optional in fetch_incubation_data_by_category and get_schwab_data

# Find out where 30 min incubation data is being transformed to daily data
# Go through whole notebook and make sure all cells have a summary markdown
# Ensure all functions have docstrings
# Continue comprehensive code review and cleanup redundant or unnecessary code
# Figure out how to code TrendHeat in Tradovate

# Modify slice_last_year_from functions to be able to take more than one year
# Implement parallel processing in the fetch_incubation_data_by_category function if possible
# Reverse the ranking process so that we sort by best performers first and fit the ranking metrics to that-
    # Keep experimenting with different models and features to improve the ranking accuracy
# Split data into analysis, walk-forward, and "incubation" periods
    # This will involve removing evaluation related code from the notebook entirely and building in new code


# pandas, NumPy, scikit-learn, matplotlib
# Use the repl
















# Config Flags
get_ticker_list_from_schwab = False
use_local_data_schwab = True
build_results_schwab = True
fresh_start_schwab = True
output_stub_schwab = pd.to_datetime(summary_candles_df["last_candle"].max()).strftime("%Y%m%d_")

display(summary_candles_df.sort_values("liquidity_proxy", ascending=False)[["ticker", "total_hours", "total_days", "total_weeks"]][0:1])
max_weeks = summary_candles_df.sort_values("liquidity_proxy", ascending=False).iloc[0]["total_weeks"]
max_days = summary_candles_df.sort_values("liquidity_proxy", ascending=False).iloc[0]["total_days"]
max_hours = summary_candles_df.sort_values("liquidity_proxy", ascending=False).iloc[0]["total_hours"]

display(max_weeks)

# fresh_start logic weeks/older
if build_results_schwab:    

    overall_finish_schwab              = pd.to_datetime(summary_candles_df["last_candle"].max())         # Most recent timestamp available in the dataset

    num_periods_schwab                 = 1                                                                # Number of walk-forward cycles to simulate
    cycle_length_weeks_schwab         = round(max_weeks / num_periods_schwab, ndigits=0)                                   # Total weeks per analysis + evaluation cycle
    cycle_length_days_schwab          = round(max_days / num_periods_schwab, ndigits=0)                                    # Total days per analysis + evaluation cycle
    analysis_ratio_schwab             = 1                                                              # Portion of each cycle allocated to analysis
    evaluation_ratio_schwab           = 1 - analysis_ratio_schwab                                        # Remaining portion allocated to evaluation

    analysis_weeks_schwab             = cycle_length_weeks_schwab * analysis_ratio_schwab               # Total weeks in each analysis window
    evaluation_weeks_schwab           = cycle_length_weeks_schwab * evaluation_ratio_schwab             # Total weeks in each evaluation window

    days_in_week_schwab               = 7                                                                # Used to convert weeks to days
    analysis_period_schwab            = timedelta(days=analysis_weeks_schwab*days_in_week_schwab)     # Total duration of an analysis window, expressed in days
    evaluation_period_schwab          = timedelta(days=evaluation_weeks_schwab*days_in_week_schwab)   # Total duration of an evaluation window, expressed in days
    
    end_of_last_analysis_period_schwab = overall_finish_schwab - evaluation_period_schwab                   # End date of final analysis window
    overall_start_schwab = end_of_last_analysis_period_schwab - num_periods_schwab * analysis_period_schwab  # Earliest date included in analysis
    expected_min_analysis_days_schwab = 1 #(analysis_weeks_schwab + evaluation_weeks_schwab)*days_in_week_schwab           # Heuristic: expected number of days required for a valid analysis

    if fresh_start_schwab:                                                                              # If a full reset is requested...
        for file in WALK_FORWARD_DIR_SCHWAB.glob("*"):                                                  # Iterate over all files in the walk-forward directory
            file.unlink()                                                                                # Delete each file (clears previous results)
    else:                                                                                                 # Otherwise, resume from where you left off
        existing_files = [f.stem.replace(output_stub_schwab, "")                                          # Get list of tickers that already have result files
                        for f in WALK_FORWARD_DIR_SCHWAB.glob("*.txt")]                                   # Only look for .txt files matching previous outputs
        tickers_to_test_schwab = [t for t in tickers_to_test_schwab if t not in existing_files]           # Remove tickers that already have results from the test list

    print("SCHWAB Overall finish date:", overall_finish_schwab)
    print("SCHWAB Number of periods:", num_periods_schwab)
    print("SCHWAB Analysis plus evaluation total weeks:", cycle_length_weeks_schwab)
    print("SCHWAB Analysis ratio:", analysis_ratio_schwab)
    print("SCHWAB Evaluation ratio:", evaluation_ratio_schwab)
    print("SCHWAB Analysis weeks:", analysis_weeks_schwab)
    print("SCHWAB Evaluation weeks:", evaluation_weeks_schwab)
    print("SCHWAB Analysis period:", analysis_period_schwab)
    print("SCHWAB Evaluation period:", evaluation_period_schwab)
    print("SCHWAB End of last analysis period:", end_of_last_analysis_period_schwab)
    print("SCHWAB Overall start date:", overall_start_schwab)
    print("SCHWAB Expected min analysis days:", expected_min_analysis_days_schwab)

def prepare_analysis_structure(ticker, end_of_last_analysis_period, analysis_period, evaluation_period, num_periods):
    """
    Build a rolling schedule of analysis/evaluation windows for one ticker.

    Args:
        ticker (str): Ticker symbol to assign to every generated row.
        end_of_last_analysis_period (datetime-like): Anchor date marking the end of the most recent
            analysis window. Earlier cycles are generated by stepping backward from this date.
        analysis_period (datetime.timedelta): Length of each analysis window.
        evaluation_period (datetime.timedelta): Length of the evaluation window that follows each
            analysis window (evaluation starts the day after analysis ends).
        num_periods (int): Number of rolling (analysis → evaluation) cycles to generate.

    Returns:
        pd.DataFrame: A table with `num_periods` rows and these columns initialized:
            - ticker (str)
            - analysis_period_start, analysis_period_end (Timestamp)
            - evaluation_period_start, evaluation_period_end (Timestamp)
            - analysis_buy, analysis_sell (float; 0.0)
            - analysis_return (float; 0.0)
            - analysis_trades (int; 0)
            - analysis_eval_metric (float; 0.0)
            - evaluation_return (float; 0.0)
            - evaluation_trades (int; 0)
            - evaluation_data_good (bool; False)
    """
    analysis_period_starts = [                                                                  # Generate list of period start dates
        end_of_last_analysis_period - (analysis_period + evaluation_period) * i                 # Each start is offset by i * total cycle length
        for i in range(1, num_periods + 1)                                                      # For the last `num_periods` analysis windows
    ]

    df = pd.DataFrame({                                                                         # Create base DataFrame for the analysis structure
        "ticker": ticker,                                                                       # Set the ticker label
        "analysis_period_start": pd.to_datetime(analysis_period_starts),                        # Assign start dates for each analysis period
    })

    df["analysis_period_end"] = df["analysis_period_start"] + analysis_period                   # Calculate the end of each analysis period
    df["analysis_buy"] = 0.0                                                                    # Initialize buy threshold column
    df["analysis_sell"] = 0.0                                                                   # Initialize sell threshold column
    df["analysis_return"] = 0.0                                                                 # Initialize return column for analysis period
    df["analysis_trades"] = 0                                                                   # Initialize number of trades in analysis period
    df["analysis_eval_metric"] = 0.0                                                            # Initialize penalized evaluation metric column
    df["evaluation_period_start"] = df["analysis_period_end"] + timedelta(days=1)               # Evaluation starts the day after analysis ends
    df["evaluation_period_end"] = df["evaluation_period_start"] + evaluation_period             # Evaluation end is offset from its start
    df["evaluation_return"] = 0.0                                                               # Initialize evaluation return column
    df["evaluation_trades"] = 0                                                                 # Initialize evaluation trade count
    df["evaluation_data_good"] = False                                                          # Flag whether evaluation data exists

    return df                                                                                   # Return the prepared DataFrame

# 1. Build the base results structure
results_structure_dict_schwab = {}
num_tickers_schwab = len(tickers_to_test_schwab)

# Prepare the analysis structure for each ticker
for i, ticker in enumerate(tickers_to_test_schwab, start=1):
    print(f"({i}/{num_tickers_schwab}) Preparing analysis structure for {ticker}")
    results_structure_dict_schwab[ticker] = prepare_analysis_structure(
        ticker,
        end_of_last_analysis_period_schwab,
        analysis_period_schwab,
        evaluation_period_schwab,
        num_periods_schwab
    )

def get_trades(data, upper_bound, lower_bound, time_start):
    """
    Generate buy/sell signals from daily OHLC using fixed bounds.

    Scans the series once (NumPy-backed loop) starting at `time_start`, enters long
    when low <= `lower_bound`, exits when high >= `upper_bound`, holds at most one
    position, and force-closes any open position on the final bar at that bar’s
    midpoint.

    Args:
        data (pd.DataFrame): Daily market data with columns:
            - 'date_time' (datetime-like, ascending)
            - 'high' (float)
            - 'low'  (float)
            Extra columns are ignored.
        upper_bound (float): Price level that triggers a sell.
        lower_bound (float): Price level that triggers a buy.
        time_start (datetime-like): Ignore rows before this timestamp.

    Returns:
        pd.DataFrame: Executed trades in chronological order with columns:
            - 'date' (Timestamp): Execution timestamp.
            - 'type' (str): 'buy' or 'sell'.
            - 'daily_high' (float): High of the execution day.
            - 'daily_low' (float): Low of the execution day.
            - 'trade_price' (float): Bound price or final-bar midpoint.
    """
    dt = pd.to_datetime(data["date_time"]).to_numpy()
    hi = data["high"].to_numpy(copy=False)
    lo = data["low"].to_numpy(copy=False)

    # start from the first row on/after time_start
    time_start = np.datetime64(pd.to_datetime(time_start))
    i0 = np.searchsorted(dt, time_start, side="left")

    state = 0  # 0 = flat, 1 = long
    trades = []
    for i in range(i0, len(dt)):
        if state == 0:
            if lo[i] <= lower_bound:
                trades.append({"date": pd.Timestamp(dt[i]), "type": "buy",
                               "daily_high": float(hi[i]), "daily_low": float(lo[i]),
                               "trade_price": float(lower_bound)})
                state = 1
        else:
            if hi[i] >= upper_bound:
                trades.append({"date": pd.Timestamp(dt[i]), "type": "sell",
                               "daily_high": float(hi[i]), "daily_low": float(lo[i]),
                               "trade_price": float(upper_bound)})
                state = 0

    if state == 1:
        # force close on last bar at mid
        trades.append({"date": pd.Timestamp(dt[-1]), "type": "sell",
                       "daily_high": float(hi[-1]), "daily_low": float(lo[-1]),
                       "trade_price": float(0.5 * (hi[-1] + lo[-1]))})

    return pd.DataFrame(trades)

def get_returns(data, upper_bound, lower_bound, time_start, starting_cash=10000, trades=None):
    """
    Compute total and annualized returns from buy/sell round-trips.

    Uses precomputed `trades` if supplied; otherwise generates trades from `data`
    and the given bounds. Positions are all-in on buys and all-out on sells.
    Total growth is the product of (sell / buy) over completed pairs; annualization
    uses the span from `time_start` to data['date_time'].max().

    Args:
        data (pd.DataFrame): Market data with 'date_time'; if `trades` is None,
            must also include 'high' and 'low' for trade generation.
        upper_bound (float): Sell threshold (used only when `trades` is None).
        lower_bound (float): Buy threshold  (used only when `trades` is None).
        time_start (datetime-like): Start of the return horizon (used for
            annualization and trade generation when needed).
        starting_cash (float, optional): Initial notional. Default: 10_000.
        trades (pd.DataFrame, optional): Executions with columns
            ['date','type','daily_high','daily_low','trade_price'].
            If provided, bounds are ignored.

    Returns:
        dict: {
            'total_return' (float|None): Final/initial − 1 over completed pairs,
                or None if no complete round-trip exists.
            'annualized_return' (float|None): Growth ** (1/years) − 1 over the
                [time_start, max(date_time)] horizon, or None if years ≤ 0.
            'num_trades' (int): Number of completed buy→sell pairs used in the calc.
        }
    """
    if trades is None:
        trades = get_trades(data, upper_bound, lower_bound, time_start)

    if trades.empty:
        return {"total_return": None, "annualized_return": None, "num_trades": 0}

    buys  = trades.loc[trades["type"] == "buy",  "trade_price"].to_numpy()
    sells = trades.loc[trades["type"] == "sell", "trade_price"].to_numpy()
    n = min(len(buys), len(sells))
    if n == 0:
        return {"total_return": None, "annualized_return": None, "num_trades": 0}

    growth = (sells[:n] / buys[:n]).prod()
    final_cash = starting_cash * growth

    last_day  = pd.to_datetime(data["date_time"]).max()
    time_start = pd.to_datetime(time_start)
    years = (last_day - time_start).days / 365.25
    if years <= 0:
        return {"total_return": None, "annualized_return": None, "num_trades": n}

    total_return = final_cash / starting_cash - 1.0
    annualized   = growth ** (1.0 / years) - 1.0
    return {"total_return": total_return, "annualized_return": annualized, "num_trades": n}

def analyze_ticker_data(data, grid_size=20, num_pse=1.5):
    """
    Grid-search optimal buy/sell thresholds with a pseudo-SE penalty.

    Evaluates a grid of lower (buy) and upper (sell) bounds and selects the
    pair that maximizes a penalized objective:
        return_lb = annualized_return − num_pse * (|annualized_return| / sqrt(trades))

    Args:
        data (pd.DataFrame): Historical OHLC data with columns:
            'low', 'high', and 'date_time' (ascending time expected).
        grid_size (int, optional): Number of evenly spaced candidates for each bound
            (produces grid_size × grid_size combinations before filtering). Default: 20.
        num_pse (float, optional): Penalty multiplier applied to the pseudo standard error
            term |annualized_return| / sqrt(trades). Default: 1.5.

    Returns:
        pd.DataFrame: Single-row DataFrame describing the best configuration with columns:
            - 'lb' (float): Selected lower bound (buy threshold).
            - 'ub' (float): Selected upper bound (sell threshold).
            - 'spread' (float): ub − lb.
            - 'return' (float): Annualized return for the selected pair.
            - 'trades' (int): Number of completed buy→sell pairs.
            - 'pseudo_se' (float): |return| / sqrt(trades) for the selection.
            - 'return_lb' (float): Penalized objective used for selection.
            - 'time_start' (Timestamp): Start timestamp used for evaluation/annualization.
    """
    lb_start = data['low'].quantile(0.01)
    lb_end = data['low'].quantile(0.75)
    ub_start = data['high'].quantile(0.10)
    ub_end = data['high'].quantile(0.99)
    time_start = data['date_time'].min()

    lb_values = np.linspace(lb_start, lb_end, grid_size)
    ub_values = np.linspace(ub_start, ub_end, grid_size)

    experiments = []

    for lb in lb_values:
        for ub in ub_values:
            if lb >= ub:
                continue

            result = get_returns(data, upper_bound=ub, lower_bound=lb, time_start=time_start)
            num_trades = result.get("num_trades", 0)

            if num_trades == 0:
                continue  # Skip configurations with no trades

            annualized_return = result.get("annualized_return") or 0.0
            spread = ub - lb
            pseudo_se = abs(annualized_return) / np.sqrt(num_trades)
            return_lb = annualized_return - num_pse * pseudo_se

            experiments.append({
                "lb": lb,
                "ub": ub,
                "spread": spread,
                "return": annualized_return,
                "trades": num_trades,
                "pseudo_se": pseudo_se,
                "return_lb": return_lb,
                "time_start": time_start
            })

    df = pd.DataFrame(experiments)

    if df.empty:
        raise ValueError("No valid parameter combinations found.")

    max_return_lb = df["return_lb"].max()
    best = df[df["return_lb"] == max_return_lb]
    best = best.sort_values(["trades", "spread"], ascending=[False, True]).tail(1)

    return best.reset_index(drop=True)

def run_analysis_loop(ticker_results, daily_data, expected_min_analysis_days):
    """
    Optimize thresholds and evaluate performance per window for a single ticker.

    For each row in `ticker_results`, slices `daily_data` into analysis/evaluation
    windows, optimizes buy/sell bounds on the analysis window, reuses generated
    trades to compute average trade duration and returns, repeats on the evaluation
    window with the optimized bounds, and writes results back into `ticker_results`.

    Args:
        ticker_results (pd.DataFrame): One row per analysis/evaluation window with
            at least ['analysis_period_start', 'analysis_period_end',
                    'evaluation_period_start', 'evaluation_period_end'].
            Updated in place with thresholds and metrics.
        daily_data (pd.DataFrame): OHLC data in ascending time with columns
            ['date_time', 'high', 'low'] (additional columns ignored).
        expected_min_analysis_days (int): Minimum number of rows required to run
            optimization for an analysis window; shorter windows are skipped.

    Returns:
        pd.DataFrame: The same `ticker_results` with these columns populated/updated:
            - analysis_buy (float), analysis_sell (float)
            - analysis_return (float, annualized), analysis_trades (int)
            - analysis_eval_metric (float)
            - analysis_avg_trade_duration (float, days)
            - evaluation_data_good (bool)
            - evaluation_trades (int), evaluation_return (float, annualized)
            - evaluation_avg_trade_duration (float, days)
    """
    for row in ticker_results.itertuples():  # includes index as row.Index
        idx = row.Index

        analysis_data = daily_data[(daily_data["date_time"] >= row.analysis_period_start) &
                                   (daily_data["date_time"] <= row.analysis_period_end)]
        if len(analysis_data) < expected_min_analysis_days:
            continue

        evaluation_data = daily_data[(daily_data["date_time"] >= row.evaluation_period_start) &
                                     (daily_data["date_time"] <= row.evaluation_period_end)]

        if not analysis_data.empty:
            period_results = analyze_ticker_data(analysis_data)
            lb = period_results["lb"].iat[0]
            ub = period_results["ub"].iat[0]

            # --- compute trades ONCE and reuse ---
            analysis_trades = get_trades(analysis_data, ub, lb, row.analysis_period_start)

            # avg trade duration (vectorized: sell - buy)
            buy_dates  = analysis_trades.loc[analysis_trades.type == "buy",  "date"].to_numpy("datetime64[D]")
            sell_dates = analysis_trades.loc[analysis_trades.type == "sell", "date"].to_numpy("datetime64[D]")
            n = min(len(buy_dates), len(sell_dates))
            if n:
                durations_days = (sell_dates[:n] - buy_dates[:n]).astype("timedelta64[D]").astype(float)
                ticker_results.at[idx, "analysis_avg_trade_duration"] = float(np.mean(durations_days))
            else:
                ticker_results.at[idx, "analysis_avg_trade_duration"] = np.nan

            # returns (reuse trades)
            ret = get_returns(analysis_data, ub, lb, row.analysis_period_start, trades=analysis_trades)

            ticker_results.at[idx, "analysis_buy"]   = lb
            ticker_results.at[idx, "analysis_sell"]  = ub
            ticker_results.at[idx, "analysis_return"] = ret["annualized_return"]
            ticker_results.at[idx, "analysis_trades"] = ret["num_trades"]
            ticker_results.at[idx, "analysis_eval_metric"] = period_results["return_lb"].iat[0]
        else:
            ticker_results.at[idx, "analysis_return"] = np.nan

        if not evaluation_data.empty:
            ticker_results.at[idx, "evaluation_data_good"] = True

            # --- compute eval trades ONCE and reuse ---
            eval_trades = get_trades(evaluation_data, ticker_results.at[idx, "analysis_sell"],
                                     ticker_results.at[idx, "analysis_buy"], row.evaluation_period_start)

            # avg trade duration (vectorized)
            buy_dates  = eval_trades.loc[eval_trades.type == "buy",  "date"].to_numpy("datetime64[D]")
            sell_dates = eval_trades.loc[eval_trades.type == "sell", "date"].to_numpy("datetime64[D]")
            n = min(len(buy_dates), len(sell_dates))
            if n:
                durations_days = (sell_dates[:n] - buy_dates[:n]).astype("timedelta64[D]").astype(float)
                ticker_results.at[idx, "evaluation_avg_trade_duration"] = float(np.mean(durations_days))
            else:
                ticker_results.at[idx, "evaluation_avg_trade_duration"] = np.nan

            eval_results = get_returns(evaluation_data,
                                       upper_bound=ticker_results.at[idx, "analysis_sell"],
                                       lower_bound=ticker_results.at[idx, "analysis_buy"],
                                       time_start=row.evaluation_period_start,
                                       trades=eval_trades)
            ticker_results.at[idx, "evaluation_trades"] = eval_results["num_trades"]
            ticker_results.at[idx, "evaluation_return"] = eval_results["annualized_return"]

    return ticker_results

print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
final_results_dict_schwab = {}
num_tickers_schwab = len(tickers_to_test_schwab)

# 2. Run analysis and evaluation loop
for i, ticker in enumerate(tickers_to_test_schwab, start=1):
    print(f"({i}/{num_tickers_schwab}) Running analysis loop for {ticker} from Schwab data")
    final_results_dict_schwab[ticker] = run_analysis_loop(
        results_structure_dict_schwab[ticker],
        candles_dict_schwab[ticker],
        expected_min_analysis_days_schwab
    )

# Convert dict items to a list to enable index-based access
final_results_list_schwab = list(final_results_dict_schwab.items())

# Example: display the DataFrame at index 0
ticker, df = final_results_list_schwab[0]
print(f"Ticker: {ticker}")
display(df)
display(final_results_dict_schwab)

# Collect all analysis_avg_trade_duration values across tickers
all_durations = []
for ticker, df in final_results_dict_schwab.items():
    if "analysis_avg_trade_duration" in df.columns:
        vals = df["analysis_avg_trade_duration"].dropna().tolist()
        all_durations.extend(vals)

if all_durations:
    all_durations = np.asarray(all_durations, dtype=float)

    avg_duration = float(np.mean(all_durations))
    med_duration = float(np.median(all_durations))

    print(f"Mean analysis_avg_trade_duration across all tickers:   {avg_duration:.2f} days")
    print(f"Median analysis_avg_trade_duration across all tickers: {med_duration:.2f} days")

    # --- Visualization ---
    plt.figure(figsize=(10, 6))
    plt.hist(all_durations, bins=50, alpha=0.7, edgecolor="black")
    plt.axvline(avg_duration, color="red", linestyle="--", linewidth=2, label=f"Mean = {avg_duration:.2f} days")
    plt.axvline(med_duration, color="blue", linestyle="-.", linewidth=2, label=f"Median = {med_duration:.2f} days")

    plt.title("Distribution of Analysis Avg Trade Duration Across All Tickers")
    plt.xlabel("Trade Duration (days)")
    plt.ylabel("Frequency")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()
else:
    print("No valid analysis_avg_trade_duration values found across tickers.")






























def list_available_swing_analysis_dates(base_dir="swing_trade_charts"):
    """
    List available swing-analysis end dates by scanning a base directory.

    Searches `base_dir` for folders named like
    `swing_analysis_period_ending_YYYY-MM-DD`, collects the YYYY-MM-DD parts,
    sorts them in reverse chronological order, and prints each date on its own line.
    If the directory doesn’t exist or no matching folders are found, prints a warning.

    Args:
        base_dir (str): Root directory to scan for dated analysis folders.
            Defaults to "swing_trade_charts".

    Returns:
        None. Prints a sorted list of detected dates (or a warning message).
    """
    base_path = Path(base_dir)
    if not base_path.exists():
        print(f"⚠️ Directory not found: {base_path}")
        return

    date_pattern = re.compile(r"swing_analysis_period_ending_(\d{4}-\d{2}-\d{2})")
    available_dates = []

    for folder in base_path.iterdir():
        if folder.is_dir():
            match = date_pattern.match(folder.name)
            if match:
                available_dates.append(match.group(1))

    if not available_dates:
        print("No valid swing analysis folders found.")
    else:
        for date_str in sorted(available_dates, reverse=True):
            print(date_str)

list_available_swing_analysis_dates(base_dir="swing_trade_charts")

def list_tickers_by_swing_strength(date_str, base_dir="swing_trade_charts"):
    """
    Build per-category ticker lists from saved chart images for a given analysis date.

    Scans:
        {base_dir}/swing_analysis_period_ending_{date_str}/prediction_charts_{date_str}/
    for subfolders named like "{category}_{date_str}" (e.g., "very_high_2025-01-10"), then
    parses each PNG filename to extract the ticker symbol and aggregates a unique, sorted
    list per category. As a side effect, also registers each list as a global variable
    named "{category}_tickers".

    Args:
        date_str (str): Analysis-period end date to target, formatted "YYYY-MM-DD".
        base_dir (str): Root directory containing swing-analysis outputs. Defaults to
            "swing_trade_charts".

    Returns:
        dict: Mapping from category name (str) → list[str] of unique, sorted tickers
            discovered under that category’s folder. Returns an empty dict if the
            expected directory is missing.
    """
    base_path = Path(base_dir) / f"swing_analysis_period_ending_{date_str}" / f"prediction_charts_{date_str}"
    if not base_path.exists():
        print(f"⚠️ Folder not found: {base_path}")
        return {}

    tickers_by_category = {}

    # Dynamically get all swing probability folders
    for subfolder in base_path.iterdir():
        if subfolder.is_dir() and subfolder.name.endswith(f"_{date_str}"):
            category = subfolder.name.replace(f"_{date_str}", "")
            tickers = []

            for file in subfolder.glob("*.png"):
                try:
                    parts = file.name.split("_")
                    ticker = parts[1]
                    tickers.append(ticker)
                except IndexError:
                    print(f"⚠️ Unexpected file format: {file.name}")

            tickers_by_category[category] = sorted(set(tickers))
            globals()[f"{category}_tickers"] = tickers_by_category[category]  # Register as global variable

    return tickers_by_category

display(chosen_end_date)
display(candles_df['date_time'].max().strftime('%Y-%m-%d'))
print(chosen_end_date == candles_df['date_time'].max().strftime('%Y-%m-%d'))

track_profit_from = chosen_end_date #PARAMETER#

print(f"Tracking profit from: {track_profit_from}")

tickers_by_swing_probability = list_tickers_by_swing_strength(date_str=track_profit_from)

for category, tickers in tickers_by_swing_probability.items():
    print(f"\n🟩 {category.upper()} Tickers for {track_profit_from}:")
    print(tickers)

display(tickers_by_swing_probability)

def fetch_incubation_data_by_category(track_from_date, tickers_by_category, base_dir="swing_trade_charts"):
    """
    Fetch and save 30-minute OHLCV “incubation” data for tickers by swing category.

    For each category in `tickers_by_category`, queries candles from `track_from_date`
    through now, cleans zero-OHLC rows, and writes one CSV per ticker under:
    {swing_trade_charts}/swing_analysis_period_ending_{track_from_date}/
        incubation_data_{track_from_date}/{category}/{ticker}.csv

    If available, the ticker’s `exchange` is inferred from prior analysis output at:
    .../top_price_data_{track_from_date}/{category}_{track_from_date}/{ticker}.csv.

    Args:
        track_from_date (str): Analysis period end date to anchor the pull window,
            formatted "YYYY-MM-DD". Used for both API start time and output paths.
        tickers_by_category (dict[str, list[str]]): Mapping of swing categories
            (e.g., "very_high") to lists of ticker symbols to fetch.
        base_dir (str): Root directory for swing outputs. Subfolders are created
            under this path. Defaults to "swing_trade_charts".

    Returns:
        None. Writes cleaned CSV files to disk and prints progress/messages.
    """
    start = pd.to_datetime(track_from_date)
    end = datetime.now()

    for category, tickers in tickers_by_category.items():
        output_folder = (
            Path(base_dir)
            / f"swing_analysis_period_ending_{track_from_date}"
            / f"incubation_data_{track_from_date}"
            / category
        )
        output_folder.mkdir(parents=True, exist_ok=True)

        for ticker in tickers:
            try:
                # Infer exchange from stored analysis data
                exchange = None
                analysis_path = (
                    Path(base_dir)
                    / f"swing_analysis_period_ending_{track_from_date}"
                    / f"top_price_data_{track_from_date}"
                    / f"{category}_{track_from_date}"
                    / f"{ticker}.csv"
                )
                if analysis_path.exists():
                    df_existing = pd.read_csv(analysis_path)
                    exchange = df_existing["exchange"].iloc[0]

                response = client.price_history(
                    symbol=ticker,
                    frequencyType="minute",
                    frequency=30,
                    startDate=start,
                    endDate=end,
                    needExtendedHoursData=False,
                )
                data = response.json()

                if "candles" in data and data["candles"]:
                    df = pd.DataFrame(data["candles"])
                    df["ticker"] = ticker
                    df["exchange"] = exchange
                    df["date_time"] = pd.to_datetime(df["datetime"], unit="ms")
                    df = df.sort_values("date_time").reset_index(drop=True)

                    # Remove zero OHLC rows
                    df = df[
                        (df["open"] != 0) &
                        (df["high"] != 0) &
                        (df["low"] != 0) &
                        (df["close"] != 0)
                    ].copy()

                    if df.empty:
                        print(f"⚠️ All-zero rows removed for {ticker}, resulting in empty DataFrame.")
                        continue

                    # Save as CSV
                    output_path = output_folder / f"{ticker}.csv"
                    df.to_csv(output_path, index=False)
                    print(f"✅ {ticker} saved in {category}")
                else:
                    print(f"⚠️ No data returned for {ticker}")
            except Exception as e:
                print(f"❌ Error fetching {ticker}: {e}")

# Load the tickers from prediction chart folders
tickers_by_swing_probability = list_tickers_by_swing_strength(date_str=track_profit_from)

# Fetch 30-minute OHLCV incubation data and save as CSVs by category
fetch_incubation_data_by_category(track_from_date=track_profit_from, tickers_by_category=tickers_by_swing_probability)

def calculate_incubation_price_change(track_from_date, base_dir="swing_trade_charts"):
    """
    Compute post-analysis price changes by category using latest closes.

    For each category under
    {swing_trade_charts}/swing_analysis_period_ending_{track_from_date}/incubation_data_{track_from_date}/,
    compares the most recent close in the incubation file to the most recent close in the
    corresponding top price file at
    .../top_price_data_{track_from_date}/{category}_{track_from_date}/{ticker}.csv,
    prints per-ticker details, and returns percent changes.

    Args:
        track_from_date (str): Analysis period end date in "YYYY-MM-DD" format. Used to locate
            both incubation_data_{track_from_date} and top_price_data_{track_from_date}.
        base_dir (str): Root directory containing swing analysis outputs. Defaults to
            "swing_trade_charts".

    Returns:
        dict[str, dict[str, float]]: Nested mapping of
            {category: {ticker: percent_change}}, where percent_change is
            100 * (incubation_close - top_price_close) / top_price_close
            for each ticker found in the category’s incubation folder.
    """
    results = {}

    incubation_root = Path(base_dir) / f"swing_analysis_period_ending_{track_from_date}" / f"incubation_data_{track_from_date}"

    for category_folder in incubation_root.iterdir():
        if not category_folder.is_dir():
            continue

        category = category_folder.name
        top_price_folder = (
            Path(base_dir)
            / f"swing_analysis_period_ending_{track_from_date}"
            / f"top_price_data_{track_from_date}"
            / f"{category}_{track_from_date}"
        )

        if not top_price_folder.exists():
            print(f"⚠️ Top price folder not found for {category}")
            continue

        ticker_changes = {}
        pct_changes = []

        for file in category_folder.glob("*.csv"):
            ticker = file.stem
            incubation_df = pd.read_csv(file)

            top_price_file = top_price_folder / f"{ticker}.csv"
            if not top_price_file.exists():
                print(f"⚠️ Top price data not found for {ticker} in {category}")
                continue

            top_price_df = pd.read_csv(top_price_file)

            try:
                incubation_close = incubation_df.sort_values("date_time")["close"].iloc[-1]
                top_price_close = top_price_df.sort_values("date_time")["close"].iloc[-1]
                pct_change = ((incubation_close - top_price_close) / top_price_close) * 100

                ticker_changes[ticker] = pct_change
                pct_changes.append(pct_change)

                # print(
                #     f"{category.upper()} - {ticker}: {pct_change:+.2f}% "
                #     f"(Analaysis Last Close Price: {top_price_close:.2f}, Incubation Close: {incubation_close:.2f})"
                # )

            except Exception as e:
                print(f"❌ Error processing {ticker} in {category}: {e}")

        results[category] = ticker_changes

        if pct_changes:
            avg_change = sum(pct_changes) / len(pct_changes)
            print(f"\n📊 {category.upper()} AVERAGE CHANGE: {avg_change:+.2f}%\n")

    return results

# --- Build a mergeable table: profit_rank (by ticker at track date) + pct change ---
def build_rank_vs_change_df(final_results_dict, price_change_results, track_from_date):
    rows = []
    track_ts = pd.to_datetime(track_from_date)

    # Flatten {category: {ticker: pct_change}} into rows and merge with ranks
    for category, tick_map in price_change_results.items():
        for ticker, pct_change in tick_map.items():
            df = final_results_dict.get(ticker)
            if df is None or df.empty:
                continue

            # Get the row for this analysis period end date
            df = df.copy()
            df["analysis_period_end"] = pd.to_datetime(df["analysis_period_end"], errors="coerce")
            at_date = df[df["analysis_period_end"] == track_ts]

            # Fallback: if not found (timezone or rounding), pick the row with min rank
            if at_date.empty:
                at_date = df.loc[[df["profit_rank"].idxmin()]]

            # Skip rows whose swing_probability is 'zero'
            if "swing_probability" in at_date.columns:
                sp = str(at_date["swing_probability"].iloc[0]).lower()
                if sp == "zero":
                    continue

            rank_val = at_date["profit_rank"].iloc[0]
            rows.append({"ticker": ticker, "category": category, "profit_rank": rank_val, "pct_change": pct_change})

    out = pd.DataFrame(rows)
    # Keep only finite numeric pairs
    return out[np.isfinite(out["profit_rank"]) & np.isfinite(out["pct_change"])]

rank_change_df = build_rank_vs_change_df(final_results_dict_schwab, price_change_results, track_profit_from)
print(f"Built rank/change table with {len(rank_change_df)} rows")

# --- Helper: remove outliers by percentile trimming on a column ---
def remove_outliers(df, col="pct_change", lo=0.0, hi=0.99):
    if df.empty or col not in df:
        return df
    q_lo, q_hi = df[col].quantile([lo, hi])
    trimmed = df[(df[col] >= q_lo) & (df[col] <= q_hi)].copy()
    print(f"Outlier trim ({int(lo*100)}–{int(hi*100)} pct): kept {len(trimmed)}/{len(df)} rows")
    return trimmed

# --- Plot: profit_rank (x) vs percent change (y) ---
def plot_rank_vs_price_change(df, title=None):
    if df.empty:
        print("No data to plot.")
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(df["profit_rank"], df["pct_change"], alpha=0.8)

    # Zero line and labels
    ax.axhline(0, linewidth=1, alpha=0.6)
    ax.set_xlabel("Profit Rank (1 = best)")
    ax.set_ylabel("% Price Change (Incubation vs Baseline)")
    ax.set_title(title or "Profit Rank vs Subsequent % Price Change")

    # Optional: simple least-squares trend line
    try:
        x = df["profit_rank"].to_numpy(dtype=float)
        y = df["pct_change"].to_numpy(dtype=float)
        if len(x) >= 2:
            m, b = np.polyfit(x, y, 1)
            xs = np.linspace(x.min(), x.max(), 200)
            ax.plot(xs, m * xs + b, linewidth=2, alpha=0.9, label="OLS trend")
            ax.legend()
    except Exception:
        pass

    plt.tight_layout()
    plt.show()

rank_change_df = build_rank_vs_change_df(final_results_dict_schwab, price_change_results, track_profit_from)

# Remove outliers for visibility (e.g., keep 2nd–98th percentile of % changes)
rank_change_vis = remove_outliers(rank_change_df, col="pct_change", lo=0.02, hi=0.98)

plot_rank_vs_price_change(rank_change_vis, title=f"Rank vs % Change (track date = {track_profit_from})")

if not rank_change_vis.empty:
    pearson = rank_change_vis[["profit_rank", "pct_change"]].corr(method="pearson").iloc[0, 1]
    spearman = rank_change_vis[["profit_rank", "pct_change"]].corr(method="spearman").iloc[0, 1]
    print(f"Pearson corr (trimmed):  {pearson:+.3f}")
    print(f"Spearman corr (trimmed): {spearman:+.3f}")

def plot_incubation_pct_change_curves(track_from_date, base_dir="swing_trade_charts", save=False, daily_only=False):
    """
    Plot incubation %-change curves per category, with a category-average overlay.

    For each swing-strength category under
    {swing_trade_charts}/swing_analysis_period_ending_{track_from_date}/incubation_data_{track_from_date}/,
    loads each ticker’s incubation series, computes percent change vs that ticker’s
    baseline (its most recent close in
    .../top_price_data_{track_from_date}/{category}_{track_from_date}/{ticker}.csv),
    plots all ticker curves together, and overlays a time-aligned average line.

    Args:
        track_from_date (str): Analysis period end date in "YYYY-MM-DD" used to resolve
            incubation_data_{track_from_date} and top_price_data_{track_from_date} folders.
        base_dir (str): Root directory that contains swing outputs. Default "swing_trade_charts".
        save (bool): If True, saves each category plot as a PNG under
            {base_dir}/swing_analysis_period_ending_{track_from_date}/incubation_charts_{track_from_date}/.
        daily_only (bool): If True, down-samples incubation data to the last close per day;
            if False, uses every 30-minute close.

    Returns:
        None. Displays one plot per category (and optionally saves it). 
    """
    base = Path(base_dir) / f"swing_analysis_period_ending_{track_from_date}"
    incub_root = base / f"incubation_data_{track_from_date}"
    top_root   = base / f"top_price_data_{track_from_date}"

    if not incub_root.exists():
        print(f"⚠️ Incubation root not found: {incub_root}")
        return
    if not top_root.exists():
        print(f"⚠️ Top price root not found: {top_root}")
        return

    for category_folder in incub_root.iterdir():
        if not category_folder.is_dir():
            continue

        category = category_folder.name
        baseline_folder = top_root / f"{category}_{track_from_date}"
        if not baseline_folder.exists():
            print(f"⚠️ Baseline folder not found for category '{category}': {baseline_folder}")
            continue

        plt.figure(figsize=(14, 8))
        any_series = False
        avg_df = pd.DataFrame()  # to store all tickers' % change series

        for file in category_folder.glob("*.csv"):
            ticker = file.stem

            try:
                inc = pd.read_csv(file)
            except Exception as e:
                print(f"❌ Failed to read incubation CSV for {ticker} in {category}: {e}")
                continue

            baseline_csv = baseline_folder / f"{ticker}.csv"
            if not baseline_csv.exists():
                print(f"⚠️ Missing baseline for {ticker} in {category}: {baseline_csv}")
                continue

            try:
                base_df = pd.read_csv(baseline_csv)
            except Exception as e:
                print(f"❌ Failed to read baseline CSV for {ticker}: {e}")
                continue

            for df in (inc, base_df):
                if "date_time" not in df.columns:
                    print(f"⚠️ 'date_time' missing in {file if df is inc else baseline_csv}")
                    continue
                df["date_time"] = pd.to_datetime(df["date_time"], errors="coerce")

            try:
                base_close = (
                    base_df.sort_values("date_time")["close"]
                    .dropna()
                    .iloc[-1]
                )
            except Exception:
                print(f"⚠️ Could not compute baseline close for {ticker}")
                continue

            if {"open","high","low","close"}.issubset(inc.columns):
                inc = inc[~((inc["open"]==0) & (inc["high"]==0) & (inc["low"]==0) & (inc["close"]==0))]

            inc = inc.sort_values("date_time")

            if daily_only:
                # Keep only the last close per day
                inc = inc.groupby(inc["date_time"].dt.date).tail(1)

            if "close" not in inc.columns or inc["close"].dropna().empty:
                print(f"⚠️ No valid close series for {ticker} in {category}")
                continue

            try:
                pct_series = (inc["close"] - base_close) / base_close * 100.0
                plt.plot(inc["date_time"], pct_series, label=ticker, linewidth=1)

                # Store for average calculation
                temp_df = pd.DataFrame({"date_time": inc["date_time"], ticker: pct_series})
                if avg_df.empty:
                    avg_df = temp_df
                else:
                    avg_df = pd.merge(avg_df, temp_df, on="date_time", how="outer")

                any_series = True
            except Exception as e:
                print(f"❌ Error computing % series for {ticker}: {e}")
                continue

        if not any_series:
            plt.close()
            print(f"ℹ️ No valid series to plot for category '{category}'.")
            continue

        # Compute running average line
        if not avg_df.empty:
            avg_df = avg_df.sort_values("date_time").set_index("date_time")
            avg_df["average_pct_change"] = avg_df.mean(axis=1, skipna=True)
            plt.plot(avg_df.index, avg_df["average_pct_change"], color="black", linewidth=2.5, label="Category Avg", alpha=0.8)

        plt.axhline(0, linewidth=1, alpha=0.6)
        freq_label = "Daily Close" if daily_only else "30m Close"
        plt.title(f"{category.upper()} — % Change since {track_from_date} ({freq_label})")
        plt.ylabel("% from baseline close")
        plt.xlabel("Date/Time")
        plt.tight_layout()
        plt.legend(ncol=2, fontsize=9)

        if save:
            out_dir = base / f"incubation_charts_{track_from_date}"
            out_dir.mkdir(parents=True, exist_ok=True)
            suffix = "_daily" if daily_only else "_30m"
            out_path = out_dir / f"{category}_{track_from_date}_pct_change{suffix}.png"
            plt.savefig(out_path, dpi=200, bbox_inches="tight")
            print(f"💾 Saved: {out_path}")

        plt.show()

plot_incubation_pct_change_curves(track_from_date=track_profit_from, base_dir="swing_trade_charts", save=True, daily_only=True)





