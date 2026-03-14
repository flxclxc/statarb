import pandas as pd
import numpy as np
from typing import Tuple, Optional
from itertools import combinations
from statsmodels.tsa.stattools import coint
import statsmodels.api as sm
from tqdm import tqdm

def ou_half_life(spread: pd.Series) -> float:
    """
    Calculate the Ornstein-Uhlenbeck half-life of a spread.
    
    Half-life is the expected time for the spread to revert to its mean.
    Lower values are better for mean-reversion strategies (faster mean reversion).
    
    Args:
        spread: The spread series
        
    Returns:
        Half-life in periods (typically days if daily data is used)
    """
    spread = spread.dropna()
    
    if len(spread) < 2:
        return np.nan
    
    try:
        lagged = spread.shift(1).dropna()
        delta = spread.diff().dropna()
        
        # Align indices
        lagged = lagged.loc[delta.index]
        
        # OLS regression: delta = lambda * lagged + constant
        X = sm.add_constant(lagged)
        model = sm.OLS(delta, X).fit()
        
        lam = model.params[0].item()
        
        # Avoid division by zero or log of non-positive numbers
        if lam >= 0 or lam == 0:
            return np.inf
        
        half_life = -np.log(2) / lam
        return half_life
    except Exception as e:
        print(f"Error calculating OU half-life: {e}")
        return np.nan

def test_cointegration(
    price_series1: pd.Series, 
    price_series2: pd.Series, 
    cutoff: float = 0.01,
    min_t_stat: float = 3.5
) -> Tuple[float, float, bool]:
    """
    Test cointegration between two price series using Engle-Granger test.
    
    Args:
        price_series1: First price series
        price_series2: Second price series
        cutoff: Significance level for cointegration test (default 0.01 for stricter)
        min_t_stat: Minimum absolute t-statistic (more negative = more cointegrated)
        
    Returns:
        Tuple of (t-statistic, p-value, is_cointegrated)
    """
    # Remove NaN values
    valid_data = pd.DataFrame({
        'series1': price_series1,
        'series2': price_series2
    }).dropna()
    
    if len(valid_data) < 2:
        return np.nan, 1.0, False
    
    # Engle-Granger cointegration test
    t_stat, p_value, _ = coint(valid_data['series1'].values, valid_data['series2'].values)
    
    # Both conditions must be met: low p-value AND large negative t-stat
    is_cointegrated = (p_value < cutoff) and (abs(t_stat) > min_t_stat)
    
    return t_stat, p_value, is_cointegrated


def calculate_hurst_exponent(series: pd.Series, lags_range: range = range(10, 100)) -> float:
    """
    Calculate the Hurst exponent to measure mean reversion.
    
    A Hurst exponent < 0.5 indicates mean reversion (good for pairs trading).
    Values close to 0.5 indicate random walk.
    Values > 0.5 indicate trending behavior.
    
    Args:
        series: Price series (typically the spread)
        lags_range: Range of lags to test
        
    Returns:
        Hurst exponent
    """
    lags = np.array(lags_range)
    tau = []
    
    for lag in lags:
        # Calculate price differences
        diff = np.diff(series.dropna(), lag)
        # Calculate variance
        tau.append(np.sqrt(np.mean(diff ** 2)))
    
    # Fit log-log regression
    poly = np.polyfit(np.log(lags), np.log(tau), 1)
    hurst = poly[0] * 2
    
    return hurst


def test_spread_stationarity(spread: pd.Series, cutoff: float = 0.05) -> Tuple[float, bool]:
    """
    Test if the spread is stationary using ADF test.
    
    Args:
        spread: The spread series
        cutoff: Significance level
        
    Returns:
        Tuple of (p-value, is_stationary)
    """
    from statsmodels.tsa.stattools import adfuller
    
    valid_spread = spread.dropna()
    if len(valid_spread) < 10:
        return 1.0, False
    
    try:
        adf_result = adfuller(valid_spread, autolag='AIC')
        p_value = adf_result[1]
        is_stationary = p_value < cutoff
        return p_value, is_stationary
    except Exception:
        return 1.0, False


def find_cointegrated_pairs(
    close_prices: pd.DataFrame,
    cutoff: float = 0.01,
    min_t_stat: float = 3.5,
    min_correlation: float = 0.3,
    require_adf: bool = True,
    max_pairs: Optional[int] = None
) -> pd.DataFrame:
    """
    Find all cointegrated pairs in the price data and flag them.
    
    Args:
        close_prices: DataFrame with close prices (columns are tickers)
        cutoff: Significance level for cointegration test (default 0.01)
        min_t_stat: Minimum absolute t-statistic (default 3.5)
        min_correlation: Minimum absolute correlation between pairs
        require_adf: Require spread to pass ADF stationarity test
        max_pairs: Limit results to top N pairs (None = all)
        
    Returns:
        DataFrame with columns: ticker1, ticker2, t_stat, p_value, correlation,
                               half_life, cointegrated
        Sorted by half_life (fastest mean reversion first)
        All pairs are included with cointegrated flag set appropriately
    """
    results = []
    tickers = close_prices.columns.tolist()
    total_pairs = len(list(combinations(tickers, 2)))
    
    print(f"Testing {total_pairs} stock pairs for cointegration...")
    print(f"Filters: p-value < {cutoff}, |t-stat| > {min_t_stat}, ")
    print(f"         |correlation| > {min_correlation}")
    print()
    
    for i, (ticker1, ticker2) in tqdm(enumerate(combinations(tickers, 2)), total=total_pairs, desc="Testing pairs"):
        if (i + 1) % 10000 == 0:
            print(f"  Progress: {i + 1}/{total_pairs} pairs tested")
        
        # Get series
        series1 = close_prices[ticker1]
        series2 = close_prices[ticker2]
        
        # Check: Minimum data points
        valid_data = pd.DataFrame({
            'series1': series1,
            'series2': series2
        }).dropna()
        
        if len(valid_data) < 50:
            continue
        
        # Calculate correlation
        corr = valid_data['series1'].corr(valid_data['series2'])
        
        # Cointegration test
        t_stat, p_value, passes_coint_test = test_cointegration(
            series1,
            series2,
            cutoff=cutoff,
            min_t_stat=min_t_stat
        )
        
        # Check correlation filter
        passes_corr_filter = abs(corr) > min_correlation
        
        # Calculate spread for OU half-life
        # Use simple spread: series1 - beta * series2
        X = valid_data['series2'].values.reshape(-1, 1)
        y = valid_data['series1'].values
        from sklearn.linear_model import LinearRegression
        model = LinearRegression().fit(X, y)
        beta = model.coef_[0]
        spread = valid_data['series1'] - beta * valid_data['series2']
        
        # Calculate OU half-life (lower is better - faster mean reversion)
        half_life = ou_half_life(spread)
        
        # Overall cointegration flag: all filters must pass
        is_cointegrated = (
            passes_coint_test and 
            passes_corr_filter
        )
        
        results.append({
            'ticker1': ticker1,
            'ticker2': ticker2,
            't_stat': t_stat,
            'p_value': p_value,
            'correlation': corr,
            'half_life': half_life,
            'cointegrated': is_cointegrated
        })
    
    # Convert to DataFrame and sort by half_life (fastest mean reversion first)
    if not results:
        print("No pairs found!")
        return pd.DataFrame()
    
    results_df = pd.DataFrame(results)
    
    # Remove pairs with infinite or NaN half-life before sorting
    results_df = results_df[np.isfinite(results_df['half_life'])].copy()
    
    # Sort by half_life (ascending = fastest mean reversion first)
    results_df = results_df.sort_values('half_life').reset_index(drop=True)
    results_df['rank'] = range(1, len(results_df) + 1)
    
    if max_pairs:
        results_df = results_df.head(max_pairs)
    
    num_cointegrated = results_df['cointegrated'].sum()
    total_tested = len(results_df)
    
    print(f"\n✓ Analysis complete!")
    print(f"  Total pairs tested: {total_tested}")
    print(f"  Cointegrated pairs (passing all filters): {num_cointegrated}")
    print(f"  Pass rate: {num_cointegrated / total_tested * 100:.2f}%")
    print(f"\n  Half-life statistics (days):")
    print(f"    Mean: {results_df['half_life'].mean():.2f}")
    print(f"    Median: {results_df['half_life'].median():.2f}")
    print(f"    Min: {results_df['half_life'].min():.2f}")
    print(f"    Max: {results_df['half_life'].max():.2f}")
    
    return results_df


def calculate_beta(price_series1: pd.Series, price_series2: pd.Series) -> float:
    """
    Calculate the beta (hedge ratio) between two price series using linear regression.
    
    Args:
        price_series1: First price series (dependent variable)
        price_series2: Second price series (independent variable)
        
    Returns:
        Beta (hedge ratio) for the pair
    """
    valid_data = pd.DataFrame({
        'series1': price_series1,
        'series2': price_series2
    }).dropna()
    
    if len(valid_data) < 2:
        return np.nan
    
    # Perform linear regression to find beta
    X = valid_data['series2'].values.reshape(-1, 1)
    y = valid_data['series1'].values
    
    from sklearn.linear_model import LinearRegression
    model = LinearRegression().fit(X, y)
    
    return model.coef_[0]


def calculate_rolling_spread(
    price_series1: pd.Series,
    price_series2: pd.Series,
    window: int = 60,
    min_periods: Optional[int] = None
) -> pd.Series:
    """
    Calculate the stationary spread between two price series on a rolling basis.
    
    The spread is calculated as: series1 - beta * series2
    where beta is computed on a rolling window using linear regression.
    
    Args:
        price_series1: First price series (long position)
        price_series2: Second price series (short position, hedging)
        window: Rolling window size (default 60 days)
        min_periods: Minimum periods for rolling calculation (default = window)
        
    Returns:
        pd.Series: Rolling stationary spread
    """
    if min_periods is None:
        min_periods = window
    
    # Align the indices
    aligned_data = pd.DataFrame({
        'series1': price_series1,
        'series2': price_series2
    }).dropna()
    
    # if len(aligned_data) < window:
        # raise ValueError(f"Not enough data: {len(aligned_data)} < window size {window}")
    
    spreads = []
    rolling_betas = []
    
    # Calculate rolling betas using a rolling window
    for i in range(len(aligned_data)):
        # if i < window - 1:
        #     spreads.append(np.nan)
        #     rolling_betas.append(np.nan)
        # else:
            # Get the window of data
            window_data = aligned_data.iloc[max(0,i - window):i]
            
            # Calculate beta for this window
            X = window_data['series2'].values.reshape(-1, 1)
            y = window_data['series1'].values
            
            try:
                from sklearn.linear_model import LinearRegression
                model = LinearRegression().fit(X, y)
                beta = model.coef_[0]
                rolling_betas.append(beta)
                
                # Calculate spread as: series1 - beta * series2
                current_series1 = aligned_data['series1'].iloc[i]
                current_series2 = aligned_data['series2'].iloc[i]
                spread = current_series1 - beta * current_series2
                spreads.append(spread)
            except Exception as e:
                print(f"Error at index {i}: {e}")
                spreads.append(np.nan)
                rolling_betas.append(np.nan)
    
    # Create output series with original index
    spread_series = pd.Series(spreads, index=aligned_data.index)
    
    return spread_series, pd.Series(rolling_betas, index=aligned_data.index)


def calculate_zscore(spread_series: pd.Series, window: int = 60) -> pd.Series:
    """
    Calculate z-score of the spread using rolling mean and standard deviation.
    
    Args:
        spread_series: The spread series
        window: Rolling window size for mean/std calculation
        
    Returns:
        pd.Series: Z-score of the spread
    """
    rolling_mean = spread_series.rolling(window=window, min_periods=1, closed='left').mean()
    rolling_std = spread_series.rolling(window=window, min_periods=1, closed='left').std()
    
    # Avoid division by zero
    zscore = (spread_series - rolling_mean) / (rolling_std + 1e-8)
    
    return zscore
