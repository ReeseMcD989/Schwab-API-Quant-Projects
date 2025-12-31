
# Make daily or 30 min optional in fetch_incubation_data_by_category and get_schwab_data
# Find out where 30 min incubation data is being transformed to daily data
# Go through whole notebook and make sure all cells have a summary markdown
# Ensure all functions have docstrings
# Continue comprehensive code review and cleanup redundant or unnecessary code
# Figure out how to code TrendHeat in Tradovate






# 0.5% trailing stop loss on tickers that pass their sell thresholds

# Investigate if smaller expected returns result in faster hits (plot pct_change against number of days it took to hit)






# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$

# Run weighting grid search on many windows to develop distributions of weights for each scoring metric

# Implement parallel processing in the fetch_incubation_data_by_category function if possible

# Reverse the ranking process so that we sort by best performers first and fit the ranking metrics to that-
    # Keep experimenting with different models and features to improve the ranking accuracy
        # Models to try: Brute-Force Grid Search, Gradient-Free Optimizer (Powell / Nelder-Mead), Linear Regression (OLS), 
        # Rank Regression (Optimize Spearman/Pearson corr), Lasso or Ridge Regression, Tree-Based Regression (RandomForest / GBM)
        # Continue to work new metrics into other models

# Continue through code to calculate incubation columns similar to evaluation columns
    # Calculate all metrics for only analysis period, some metrics for only evaluation and incubation periods

# Isolate striken tickers in the trimmed data and investigate relationship between rank/probability and strike binary and trade length

# Clean up add_high_weight_metrics and add_relative_strength_metrics. Unabbreviate all the local variables so this reads nicely

# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$













# Python, pandas, NumPy, scikit-learn, matplotlib
# Use the repl



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

    # df["analysis_period_end"] = df["analysis_period_start"] + analysis_period                   # Calculate the end of each analysis period
    # df["analysis_buy"] = 0.0                                                                    # Initialize buy threshold column
    # df["analysis_sell"] = 0.0                                                                   # Initialize sell threshold column
    # df["analysis_return"] = 0.0                                                                 # Initialize return column for analysis period
    # df["analysis_trades"] = 0                                                                   # Initialize number of trades in analysis period
    # df["analysis_eval_metric"] = 0.0                                                            # Initialize penalized evaluation metric column
    # df["evaluation_period_start"] = df["analysis_period_end"] + timedelta(days=1)               # Evaluation starts the day after analysis ends
    # df["evaluation_period_end"] = df["evaluation_period_start"] + evaluation_period             # Evaluation end is offset from its start
    # df["evaluation_return"] = 0.0                                                               # Initialize evaluation return column
    # df["evaluation_trades"] = 0                                                                 # Initialize evaluation trade count
    # df["evaluation_data_good"] = False                                                          # Flag whether evaluation data exists

    return df                                                                                   # Return the prepared DataFrame




























# Split data into analysis(possibly "development" or "calibration"), "walk-forward", and "incubation" periods
    # This will involve removing evaluation related code from the notebook entirely and building in new code
    # And replacing later walk forward data with code that splits the data the first time it's called


















################################################################################################################################################
# Futures_Trading_Performance

# Questions to answer:
    # trade_direction and turn_out interaction
    # success rate by time of day
    # put pnl_usd, trade_direction_ and turn_out against indicators
    # Calculate AI suggested metrics using trade data against ohlc data
        # Metrics include:
            # Adverse Excursion vs Outcome (MAE / MFE)
            # 

#############################################################################################################################################




# Get used to abbreviated variable/object names





















