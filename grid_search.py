"""
Grid search optimizer for finding optimal trading strategy parameters.
Tests different combinations of entry_threshold, exit_threshold, spread_window,
and zscore_window on training data.
"""

import pandas as pd
import numpy as np
from itertools import product
from typing import Dict, List, Optional
import warnings
from joblib import Parallel, delayed

# Suppress sklearn warnings about empty arrays
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')
warnings.filterwarnings('ignore', message='.*Found array with 0 sample.*')
warnings.filterwarnings('ignore', message='.*LinearRegression.*')

from stats import calculate_rolling_spread, calculate_zscore
from backtest import SimpleBacktester, BacktestResult


class GridSearchOptimizer:
    """Grid search optimizer for trading strategy parameters"""
    
    def __init__(self,
                 initial_capital: float = 100000.0,
                 transaction_cost: float = 0.001,
                 position_size: float = 0.5,
                 n_jobs: int = 1):
        """
        Initialize the grid search optimizer
        
        Args:
            initial_capital: Starting capital for backtests
            transaction_cost: Transaction cost as fraction
            position_size: Position size for each trade
            n_jobs: Number of parallel jobs (-1 = all cores)
        """
        self.initial_capital = initial_capital
        self.transaction_cost = transaction_cost
        self.position_size = position_size
        self.n_jobs = n_jobs
    
    def grid_search(self,
                    price_data: pd.DataFrame,
                    pairs: pd.DataFrame,
                    param_grid: Dict,
                    metric: str = 'sharpe_ratio') -> pd.DataFrame:
        """
        Perform grid search over parameter combinations
        
        Args:
            price_data: DataFrame with price data indexed by date
            pairs: DataFrame with columns ['ticker1', 'ticker2'] for pairs to test
            param_grid: Dict of parameter names to lists of values
            metric: Metric to optimize ('sharpe_ratio', 'total_return', 'win_rate')
            
        Returns:
            DataFrame with all results sorted by metric (best first)
        """
        print(f"\n{'='*80}")
        print("GRID SEARCH - PARAMETER OPTIMIZATION")
        print(f"{'='*80}")
        print(f"Metric to optimize: {metric}")
        print(f"Parallel jobs: {self.n_jobs}")
        
        # Generate all parameter combinations
        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())
        combinations = list(product(*param_values))
        
        print(f"Total parameter combinations: {len(combinations)}")
        print(f"Pairs to test: {len(pairs)}")
        print(f"Total evaluations: {len(combinations) * len(pairs)}\n")
        
        # Create list of jobs
        jobs = []
        for combo_idx, combo in enumerate(combinations):
            params = dict(zip(param_names, combo))
            for pair_idx, (_, pair) in enumerate(pairs.iterrows()):
                ticker1 = pair['ticker1']
                ticker2 = pair['ticker2']
                jobs.append((params, ticker1, ticker2, price_data, combo_idx, len(combinations), pair_idx, len(pairs)))
        
        # Run parallel evaluation
        if self.n_jobs == 1:
            results = []
            for i, job in enumerate(jobs):
                result = self._evaluate_params(*job)
                if result is not None:
                    results.append(result)
                if (i + 1) % 10 == 0:
                    print(f"  Progress: {i + 1}/{len(jobs)} evaluations completed")
        else:
            results = Parallel(n_jobs=self.n_jobs, verbose=1)(
                delayed(self._evaluate_params)(*job) for job in jobs
            )
            results = [r for r in results if r is not None]
        
        # Convert to DataFrame and sort
        results_df = pd.DataFrame(results)
        
        if metric == 'sharpe_ratio':
            # Higher is better
            results_df = results_df.sort_values('sharpe_ratio', ascending=False).reset_index(drop=True)
        elif metric == 'total_return':
            # Higher is better
            results_df = results_df.sort_values('total_return', ascending=False).reset_index(drop=True)
        elif metric == 'win_rate':
            # Higher is better
            results_df = results_df.sort_values('win_rate', ascending=False).reset_index(drop=True)
        
        results_df['rank'] = range(1, len(results_df) + 1)
        
        return results_df
    
    def grid_search_single_pair(self,
                               price_data: pd.DataFrame,
                               ticker1: str,
                               ticker2: str,
                               param_grid: Dict,
                               metric: str = 'sharpe_ratio') -> pd.DataFrame:
        """
        Perform grid search for a single pair over parameter combinations
        
        Args:
            price_data: DataFrame with 2 columns [ticker1, ticker2]
            ticker1: First ticker symbol
            ticker2: Second ticker symbol
            param_grid: Dict of parameter names to lists of values
            metric: Metric to optimize ('sharpe_ratio', 'total_return', 'win_rate')
            
        Returns:
            DataFrame with all results sorted by metric (best first)
        """
        # Generate all parameter combinations
        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())
        combinations = list(product(*param_values))
        
        print(f"Testing {len(combinations)} parameter combinations for {ticker1}_{ticker2}...")
        
        # Evaluate each parameter combination
        results = []
        for combo_idx, combo in enumerate(combinations):
            params = dict(zip(param_names, combo))
            
            result = self._evaluate_params(
                params=params,
                ticker1=ticker1,
                ticker2=ticker2,
                price_data=price_data,
                combo_idx=combo_idx,
                total_combos=len(combinations),
                pair_idx=0,
                total_pairs=1
            )
            
            if result is not None:
                results.append(result)
            
            # Progress indicator
            if (combo_idx + 1) % max(1, len(combinations) // 10) == 0:
                print(f"  Progress: {combo_idx + 1}/{len(combinations)}")
        
        # Convert to DataFrame and sort
        results_df = pd.DataFrame(results)
        
        if len(results_df) == 0:
            print(f"  ⚠️  No valid parameter combinations found!")
            return pd.DataFrame()
        
        if metric == 'sharpe_ratio':
            results_df = results_df.sort_values('sharpe_ratio', ascending=False).reset_index(drop=True)
        elif metric == 'total_return':
            results_df = results_df.sort_values('total_return', ascending=False).reset_index(drop=True)
        elif metric == 'win_rate':
            results_df = results_df.sort_values('win_rate', ascending=False).reset_index(drop=True)
        
        results_df['rank'] = range(1, len(results_df) + 1)
        
        # Remove ticker columns for consistency
        if 'ticker1' in results_df.columns:
            results_df = results_df.drop(['ticker1', 'ticker2'], axis=1)
        
        return results_df
    
    def _evaluate_params(self,
                        params: Dict,
                        ticker1: str,
                        ticker2: str,
                        price_data: pd.DataFrame,
                        combo_idx: int,
                        total_combos: int,
                        pair_idx: int,
                        total_pairs: int) -> Optional[Dict]:
        """
        Evaluate a single parameter combination on a single pair
        
        Args:
            params: Parameter dict with spread_window, zscore_window, etc.
            ticker1: First ticker symbol
            ticker2: Second ticker symbol
            price_data: Price data DataFrame
            combo_idx: Current combination index
            total_combos: Total combinations
            pair_idx: Current pair index
            total_pairs: Total pairs
            
        Returns:
            Dict with results or None if evaluation failed
        """
        try:
            # Extract parameters
            spread_window = params['spread_window']
            zscore_window = params['zscore_window']
            entry_threshold = params['entry_threshold']
            exit_threshold = params['exit_threshold']
            
            # Get price series
            if ticker1 not in price_data.columns or ticker2 not in price_data.columns:
                return None
            
            series1 = price_data[ticker1]
            series2 = price_data[ticker2]
            
            # Align and check data
            valid_data = pd.DataFrame({
                'series1': series1,
                'series2': series2
            }).dropna()
            
            if len(valid_data) < spread_window + zscore_window:
                return None
            
            # Calculate rolling spread
            spread_series, _ = calculate_rolling_spread(
                series1, series2, window=spread_window
            )
            
            # Calculate z-score
            zscore = calculate_zscore(spread_series, window=zscore_window)
            
            # Run backtest
            backtester = SimpleBacktester(
                initial_capital=self.initial_capital,
                transaction_cost=self.transaction_cost
            )
            
            backtest_result = backtester.backtest_pairs_strategy(
                spread_data=spread_series,
                signals=zscore,
                entry_threshold=entry_threshold,
                exit_threshold=exit_threshold,
                position_size=self.position_size
            )
            
            # Collect results
            result = {
                'spread_window': spread_window,
                'zscore_window': zscore_window,
                'entry_threshold': entry_threshold,
                'exit_threshold': exit_threshold,
                'ticker1': ticker1,
                'ticker2': ticker2,
                'total_return': backtest_result.total_return,
                'annual_return': backtest_result.annual_return,
                'sharpe_ratio': backtest_result.sharpe_ratio,
                'max_drawdown': backtest_result.max_drawdown,
                'win_rate': backtest_result.win_rate,
                'num_trades': len(backtest_result.trades),
            }
            
            return result
            
        except Exception as e:
            # Silently skip failed evaluations
            return None
    
    def _evaluate_pair_with_params(self,
                                  ticker1: str,
                                  ticker2: str,
                                  price_data: pd.DataFrame,
                                  spread_window: int,
                                  zscore_window: int,
                                  entry_threshold: float,
                                  exit_threshold: float) -> Optional[BacktestResult]:
        """
        Evaluate a single pair with specific parameters
        
        Args:
            ticker1: First ticker symbol
            ticker2: Second ticker symbol
            price_data: Price data DataFrame
            spread_window: Window for rolling spread calculation
            zscore_window: Window for z-score calculation
            entry_threshold: Entry threshold for signals
            exit_threshold: Exit threshold for signals
            
        Returns:
            BacktestResult or None if evaluation failed
        """
        try:
            # Get price series
            if ticker1 not in price_data.columns or ticker2 not in price_data.columns:
                return None
            
            series1 = price_data[ticker1]
            series2 = price_data[ticker2]
            
            # Align and check data
            valid_data = pd.DataFrame({
                'series1': series1,
                'series2': series2
            }).dropna()
            
            if len(valid_data) < spread_window + zscore_window:
                return None
            
            # Calculate rolling spread
            spread_series, _ = calculate_rolling_spread(
                series1, series2, window=spread_window
            )
            
            # Calculate z-score
            zscore = calculate_zscore(spread_series, window=zscore_window)
            
            # Run backtest
            backtester = SimpleBacktester(
                initial_capital=self.initial_capital,
                transaction_cost=self.transaction_cost
            )
            
            backtest_result = backtester.backtest_pairs_strategy(
                spread_data=spread_series,
                signals=zscore,
                entry_threshold=entry_threshold,
                exit_threshold=exit_threshold,
                position_size=self.position_size
            )
            
            return backtest_result
            
        except Exception as e:
            return None
