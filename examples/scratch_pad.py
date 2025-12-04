
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



































# Split data into analysis(possibly "development" or "calibration"), "walk-forward", and "incubation" periods
    # This will involve removing evaluation related code from the notebook entirely and building in new code
    # And replacing later walk forward data with code that splits the data the first time it's called





























