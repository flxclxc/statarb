# Statistical Arbitrage Pairs Trading Strategy

A complete end-to-end pairs trading strategy pipeline that identifies cointegrated stock pairs, optimizes trading parameters via grid search, and backtests on out-of-sample data.

## Overview

This project implements a statistical arbitrage strategy using pairs trading. The strategy identifies pairs of stocks that move together (cointegrated), then trades on mean reversion when their spread deviates from historical norms.

### Key Features

- **Cointegration Analysis**: Identifies statistically related stock pairs from S&P 500
- **Per-Pair Grid Search**: Optimizes trading parameters individually for each pair using parallel processing
- **Out-of-Sample Validation**: Trains on historical data, backtests on completely separate period
- **Organized Results**: Timestamped result folders with CSV exports and visualizations
- **Unified Pipeline**: Single `StrategyPipeline` class handles all steps without nested pipeline classes
- **Comprehensive Visualizations**: Rolling spreads, z-scores, equity curves, and trade markers

## Pipeline Architecture

The strategy runs as a single unified pipeline with 7 sequential steps:

```
┌─────────────────────────────────────┐
│  Step 1: Load Data                  │
│  (S&P 500 stock prices)             │
└────────────────┬────────────────────┘
                 ↓
┌─────────────────────────────────────┐
│  Step 1.5: Train/Test Split         │
│  (40% train, 60% out-of-sample)     │
└────────────────┬────────────────────┘
                 ↓
┌─────────────────────────────────────┐
│  Step 2: Cointegration Analysis     │
│  (Training data only)               │
└────────────────┬────────────────────┘
                 ↓
┌─────────────────────────────────────┐
│  Step 2.5: Grid Search Optimization │
│  (Per-pair parameter tuning)        │
└────────────────┬────────────────────┘
                 ↓
┌─────────────────────────────────────┐
│  Steps 3-7: Backtest Pipeline       │
│  (Out-of-sample data)               │
│  • Step 3: Rolling Spread Analysis  │
│  • Step 4: Z-Score Signals          │
│  • Step 5: Backtest Strategy        │
│  • Step 6: Trade Details & Viz      │
│  • Step 7: Performance Analysis     │
└────────────────┬────────────────────┘
                 ↓
┌─────────────────────────────────────┐
│  Combined Results Analysis          │
│  (Ensemble visualization)           │
└─────────────────────────────────────┘
```

## Installation

### Requirements
- Python 3.8+
- macOS / Linux / Windows

### Setup

```bash
# Clone the repository
git clone <repo-url>
cd statarb

# Install dependencies
pip install pandas numpy scikit-learn matplotlib pyyaml yfinance statsmodels joblib

# Run the pipeline
python main.py
```

## Quick Start

```bash
python main.py
```

This will:
1. Download/load S&P 500 price data
2. Identify cointegrated pairs on training data
3. Run grid search to optimize parameters for each pair
4. Backtest the strategy on out-of-sample data
5. Generate visualizations and CSV reports
6. Save everything to `results/YYYYMMDD_HHMMSS/`

## Configuration

All strategy parameters are defined in `config.yml`. Modify this file to customize the strategy:

### Data Configuration
```yaml
data:
  min_trading_days: 1500      # Minimum historical data points
  num_stocks: 50              # Number of stocks to analyze
```

### Train/Test Split
```yaml
train_test_split:
  train_fraction: 0.4         # 40% training, 60% out-of-sample
```

### Cointegration Analysis
```yaml
cointegration:
  p_value_cutoff: 0.01        # Stricter = fewer pairs identified
  min_t_stat: 3.5             # Minimum t-statistic strength
  min_correlation: 0.3        # Minimum correlation threshold
  require_adf: true           # Require ADF test confirmation
```

### Grid Search Parameters
```yaml
pipeline:
  # Ranges to search over (for each pair independently)
  grid_search_spread_windows: [30, 50, 70]          # Days
  grid_search_zscore_windows: [30, 50, 70]          # Days
  grid_search_entry_thresholds: [1.0, 1.5, 2.0, 2.5]  # Z-score
  grid_search_exit_thresholds: [0.0, 0.5, 1.0]        # Z-score
  
  # Position management
  position_size: 0.5          # 50% of capital per trade
  initial_capital: 100000.0   # $100k starting capital
  transaction_cost: 0.0       # 0% transaction cost (set to 0.001 for 0.1%)
  
  # Output directories
  output_dir: plots           # For visualizations
  results_dir: results        # For CSV results
```

### Backtest Configuration
```yaml
backtest:
  num_pairs: 10               # Top 10 pairs to backtest
  generate_combined_plots: true
```

## Output Structure

Results are organized into timestamped folders for easy tracking:

```
results/
├── 20260314_190947/                          # Timestamp: YYYYMMDD_HHMMSS
│   ├── cointegrated_pairs.csv                # All pairs passing filters
│   ├── grid_search_results.csv               # All parameter combinations tested
│   ├── optimal_parameters_by_pair.csv        # Best parameters per pair
│   └── backtest_summary.csv                  # Final performance metrics

plots/
├── PAIR1_PAIR2_rolling_spread_analysis.png   # Spread + beta + prices
├── PAIR1_PAIR2_zscore_signals.png            # Z-score with trading zones
├── PAIR1_PAIR2_backtest_results.png          # Equity curve + trade markers
└── combined_equity_curve.png                 # All pairs + ensemble
```

## Strategy Logic

### How It Works

1. **Identify Pairs**: Find stocks that move together (cointegrated)
2. **Calculate Spread**: Compute stationary spread using rolling regression
3. **Generate Signals**: Normalize spread to z-score, trade on extremes
4. **Execute Trades**: Long when spread is low, short when high
5. **Measure Performance**: Track returns, Sharpe ratio, drawdown, win rate

### Position Entry/Exit

**LONG Position** (when spread is unusually **low**):
```
Entry:  z-score < -entry_threshold  (e.g., -2.0)
Exit:   z-score > -exit_threshold   (e.g., -0.5)
Action: Short stock1 / Long stock2
Profit: If spread mean-reverts upward
```

**SHORT Position** (when spread is unusually **high**):
```
Entry:  z-score > entry_threshold   (e.g., +2.0)
Exit:   z-score < exit_threshold    (e.g., +0.5)
Action: Long stock1 / Short stock2
Profit: If spread mean-reverts downward
```

### Spread Calculation

The spread is computed via rolling linear regression:
```
spread = price1 - (beta * price2)
```

Where `beta` is the slope from regressing price1 on price2, computed in a rolling window. This makes the spread **stationary** and suitable for mean reversion trading.

### Z-Score Signals

Trading signals are normalized using z-score:
```
z-score = (spread - mean(spread)) / std(spread)
```

Benefits:
- Consistent thresholds across pairs
- Handles pairs with different spreads/volatilities
- Quantifies deviation from mean in standard deviations

## Grid Search Optimization

### Per-Pair Parameter Tuning

Each pair receives **individual optimization** during training:

1. **Generate Grid**: 3 × 3 × 4 × 3 = **108 combinations** per pair
   - spread_window: [30, 50, 70] days
   - zscore_window: [30, 50, 70] days
   - entry_threshold: [1.0, 1.5, 2.0, 2.5]
   - exit_threshold: [0.0, 0.5, 1.0]

2. **Backtest Each**: Test all 108 combinations on training data
   - Parallel processing for speed (`n_jobs=-1`)
   - Uses Sharpe ratio as optimization metric

3. **Select Best**: Choose parameters with highest Sharpe ratio
   - Example: Pair V-MA might get [spread=50, zscore=70, entry=2.0, exit=0.5]
   - Example: Pair KO-PEP might get [spread=30, zscore=50, entry=1.5, exit=0.0]

4. **Backtest OOS**: Use optimal parameters on out-of-sample data

### Why Per-Pair Optimization?

Different stock pairs have different dynamics:
- **Fast mean-reverters**: Benefit from shorter windows
- **Slow mean-reverters**: Benefit from longer windows
- **Volatile pairs**: Need higher entry thresholds
- **Stable pairs**: Can use lower thresholds

Global optimization would compromise some pairs to benefit others. Per-pair optimization allows each pair to trade with its ideal rules.

## Key Files

| File | Purpose |
|------|---------|
| **main.py** | Single `StrategyPipeline` class (all 7 steps) |
| **config.yml** | All configurable parameters |
| **stats.py** | Cointegration analysis & statistics |
| **backtest.py** | `SimpleBacktester` class for trading simulation |
| **grid_search.py** | `GridSearchOptimizer` for parameter optimization |
| **data.py** | Data fetching utilities (yfinance) |

## Performance Metrics

The strategy reports the following metrics:

| Metric | Definition | Better? |
|--------|-----------|---------|
| **Total Return** | Final equity / initial capital | Higher |
| **Annual Return** | Return annualized to 252 trading days | Higher |
| **Sharpe Ratio** | Excess return / volatility | Higher |
| **Max Drawdown** | Largest peak-to-trough decline | Higher (less negative) |
| **Win Rate** | % of trades that are profitable | Higher |
| **Num Trades** | Total trades executed | Context-dependent |
| **Profit Factor** | Avg win / Avg loss | Higher |

## Example Results

```
================================================================================
COMBINED RESULTS SUMMARY
================================================================================

Total pairs backtested: 10

DETAILED PERFORMANCE TABLE
ticker1 ticker2 total_return sharpe_ratio max_drawdown num_trades win_rate
     KO     PEP        8.24%         1.32       -12.5%        45     62.2%
      V      MA       12.15%         1.87        -8.3%        38     65.8%
    XOM     CVX       -2.31%         0.45       -15.2%        52     48.1%
    JPM     BAC        5.67%         0.89       -10.1%        41     58.5%
     HD     LOW        9.11%         1.41        -9.8%        47     60.7%

AGGREGATE STATISTICS

Average Metrics:
  Total Return: 5.12%
  Sharpe Ratio: 1.24
  Max Drawdown: -11.8%
  Win Rate: 57.3%
  Avg Trades: 42

Best Performers:
  Highest Return: V-MA (12.15%)
  Highest Sharpe: V-MA (1.87)
  Most Trades: XOM-CVX (52 trades)
```

## Visualizations

### Rolling Spread Analysis
Shows:
- Price series for both stocks (dual y-axis)
- Rolling hedge ratio (beta) over time
- Rolling spread with ±1 std dev bands

### Z-Score Signals
Shows:
- Z-score with entry/exit thresholds
- Buy/sell regions highlighted
- Days where signals would trigger

### Backtest Results
Shows:
- Equity curve with profit/loss regions
- Actual trades marked:
  - 🟢 Green triangles: Entry points
  - 🟢 Green circles: Exit from winning trades
  - 🔴 Red triangles: Entry points
  - 🔴 Red circles: Exit from losing trades

### Combined Equity Curve
Shows:
- All individual pair equity curves
- Ensemble (equal-weight average)
- Initial capital baseline

## Limitations & Considerations

### Statistical
- **Look-Ahead Bias**: Parameters optimized on training data
- **Overfitting**: Grid search may fit noise rather than signal
- **Regime Changes**: Cointegration can break down during crises
- **Survivorship Bias**: Only includes current S&P 500 stocks

### Practical
- **Transaction Costs**: Slippage and commissions not fully modeled
- **Gap Risk**: Assumes continuous trading, ignores overnight gaps
- **Liquidity**: Small-cap pairs may be hard to trade
- **Execution**: Uses daily close prices, not realistic fills
- **Leverage**: No explicit position sizing constraints

### Data
- **Delisting**: Strategy only uses current stocks (survivorship bias)
- **Corporate Actions**: Splits/dividends may create false signals
- **Missing Data**: Stocks with gaps dropped from analysis

## Future Enhancements

- [ ] **Dynamic Position Sizing**: Scale positions based on volatility
- [ ] **Stop-Loss Orders**: Limit downside on individual trades
- [ ] **Entry Signal Confirmation**: Wait for multiple signals before trading
- [ ] **Portfolio-Level Risk Management**: Max portfolio drawdown limits
- [ ] **Machine Learning**: Use ML for pair selection
- [ ] **Real-Time Execution**: Connect to broker API
- [ ] **Transaction Cost Modeling**: More realistic commissions/slippage
- [ ] **Walk-Forward Analysis**: Retrain parameters periodically

## Troubleshooting

### No cointegrated pairs found
- **Cause**: Too strict cointegration filters
- **Solution**: Loosen `p_value_cutoff` or `min_t_stat` in config.yml

### Grid search is slow
- **Cause**: Parallel processing not enabled or testing too many combinations
- **Solution**: Reduce parameter ranges or ensure `n_jobs=-1` in grid_search.py

### Overfitting concerns
- **Cause**: Best parameters on training data underperform on test data
- **Solution**: Use larger `train_fraction` or add regularization

### Poor backtesting results
- **Cause**: Mean reversion hypothesis doesn't hold during test period
- **Solution**: Try different train/test periods or adjust strategy parameters

## References

### Academic Papers
- **Gatev, E., Goetzmann, W. N., & Rouwenhorst, K. G. (2006)**
  "Pairs trading: Performance of a relative-value arbitrage rule"
  *Journal of Finance*

- **Vidyamurthy, G. (2004)**
  "Pairs Trading: Quantitative Methods and Analysis"
  *Wiley Finance*

### Resources
- [Cointegration on Wikipedia](https://en.wikipedia.org/wiki/Cointegration)
- [Mean Reversion Strategies](https://en.wikipedia.org/wiki/Mean_reversion)
- [Statistical Arbitrage](https://en.wikipedia.org/wiki/Statistical_arbitrage)

## License

MIT License - See LICENSE file for details

## Disclaimer

This project is for **educational purposes only**. Past performance does not guarantee future results. Use of this strategy involves substantial risk of loss. Always test thoroughly and use appropriate position sizing before trading with real capital.

## Author

Statistical Arbitrage Research Project  
Last Updated: March 2026
