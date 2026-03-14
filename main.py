"""
Statistical Arbitrage Strategy - Unified Pipeline
Complete end-to-end pairs trading strategy pipeline with:
1. Data loading & train/test split
2. Cointegration analysis (training data only)
3. Grid search for optimal parameters (per-pair, training data)
4. Rolling spread calculation (out-of-sample)
5. Z-score signal generation (out-of-sample)
6. Backtest simulation (out-of-sample)
7. Combined results analysis
"""

import os
import yaml
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from datetime import datetime

from data import fetch_and_save_sp500_data
from stats import find_cointegrated_pairs, calculate_rolling_spread, calculate_zscore
from grid_search import GridSearchOptimizer
from backtest import SimpleBacktester, BacktestResult


class StrategyPipeline:
    """Unified pipeline for complete pairs trading strategy (Steps 1-7)"""
    
    def __init__(self, config_file: str = 'config.yml'):
        """
        Initialize the unified strategy pipeline
        
        Args:
            config_file: Path to YAML configuration file
        """
        self.config = self._load_config(config_file)
        self._setup_directories()
    
    def _load_config(self, config_file: str) -> Dict:
        """Load configuration from YAML file"""
        print("\n" + "="*80)
        print("LOADING CONFIGURATION FROM config.yml")
        print("="*80)
        
        try:
            with open(config_file, 'r') as f:
                config_dict = yaml.safe_load(f)
            print("✓ Configuration loaded successfully")
            return config_dict
        except FileNotFoundError:
            raise FileNotFoundError(f"{config_file} not found! Please create it in the project root.")
        except yaml.YAMLError as e:
            raise ValueError(f"Error parsing {config_file}: {e}")
    
    def _setup_directories(self):
        """Create necessary output directories"""
        results_base = self.config['pipeline'].get('results_dir', 'results')
        self.results_dir = os.path.join(results_base, datetime.now().strftime("%Y%m%d_%H%M%S"))
        self.plots_dir = self.config['pipeline'].get('output_dir', 'plots')
        self.data_dir = self.config['pipeline'].get('data_dir', 'data')
        
        for dir_path in [self.results_dir, self.plots_dir, self.data_dir]:
            if not os.path.exists(dir_path):
                os.makedirs(dir_path, exist_ok=True)
        
        print(f"✓ Results directory: {self.results_dir}")
        print(f"✓ Plots directory: {self.plots_dir}")
    
    def run(self):
        """Execute the complete pipeline"""
        try:
            df = self._step_1_load_data()
            train_df, oos_df = self._step_1_5_train_test_split(df)
            cointegrated_pairs = self._step_2_cointegration_analysis(train_df)
            optimal_params_by_pair = self._step_2_5_grid_search(train_df, cointegrated_pairs)
            backtest_results = self._steps_3_7_backtest_pipeline(oos_df, cointegrated_pairs, optimal_params_by_pair)
            self._combined_results_analysis(backtest_results)
            
            print("\n" + "="*80)
            print("PIPELINE EXECUTION COMPLETE!")
            print("="*80)
            
        except Exception as e:
            print(f"\n❌ ERROR: {str(e)}")
            import traceback
            traceback.print_exc()
            raise
    
    # ========================================================================
    # STEP 1: Load Data
    # ========================================================================
    def _step_1_load_data(self) -> pd.DataFrame:
        """Step 1: Load S&P 500 price data"""
        print("\n" + "="*80)
        print("STEP 1: LOADING S&P 500 PRICE DATA")
        print("="*80)
        
        data_config = self.config['data']
        
        if not os.path.exists('sp500_close_prices.csv'):
            print("Downloading S&P 500 data...")
            df = fetch_and_save_sp500_data(output_path=self.data_dir + '/sp500_close_prices.csv')
        else:
            print("S&P 500 data already exists. Loading from cache...")
            df = pd.read_csv('sp500_close_prices.csv', index_col=0, parse_dates=True)
        
        tick_counts_by_ticker = (1 - df.isna()).sum()
        df = df[tick_counts_by_ticker[tick_counts_by_ticker >= data_config['min_trading_days']].index]
        
        tickers = [
            "KO", "PEP", "V", "MA", "XOM", "CVX", "JPM", "BAC", "WFC",
            "HD", "LOW", "TMO", "DHR", "ADBE", "CRM", "SBUX", "MCD",
            "DAL", "UAL", "LUV", "BKNG", "EXPE", "CMCSA", "DIS",
            "DUK", "SO", "NEE", "LIN", "APD", "SHW", "PPG",
            "AMAT", "LRCX", "NVDA", "AMD", "F", "GM"
        ]
        df = df[tickers]
        
        print(f"Loaded data shape: {df.shape}")
        print(f"Date range: {df.index.min()} to {df.index.max()}")
        
        return df
    
    # ========================================================================
    # STEP 1.5: Train/Test Split
    # ========================================================================
    def _step_1_5_train_test_split(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Step 1.5: Split data into training and out-of-sample periods"""
        print("\n" + "="*80)
        print("STEP 1.5: TRAIN/TEST SPLIT")
        print("="*80)
        
        train_test_config = self.config['train_test_split']
        total_days = len(df)
        train_days = int(total_days * train_test_config['train_fraction'])
        
        train_df = df.iloc[:train_days]
        oos_df = df.iloc[train_days:]
        
        print(f"\nTraining period:")
        print(f"  Start: {train_df.index.min()}")
        print(f"  End: {train_df.index.max()}")
        print(f"  Days: {len(train_df)}")
        
        print(f"\nOut-of-Sample period:")
        print(f"  Start: {oos_df.index.min()}")
        print(f"  End: {oos_df.index.max()}")
        print(f"  Days: {len(oos_df)}")
        
        return train_df, oos_df
    
    # ========================================================================
    # STEP 2: Cointegration Analysis
    # ========================================================================
    def _step_2_cointegration_analysis(self, train_df: pd.DataFrame) -> pd.DataFrame:
        """Step 2: Cointegration analysis on training data only"""
        print("\n" + "="*80)
        print("STEP 2: COINTEGRATION ANALYSIS (Training Data Only)")
        print("="*80)
        
        cointegration_config = self.config['cointegration']
        
        print("Finding cointegrated pairs on training data...")
        print(f"Sorting by: Ornstein-Uhlenbeck Half-Life (lower = faster mean reversion)")
        
        cointegrated_pairs = find_cointegrated_pairs(
            train_df,
            cutoff=cointegration_config['p_value_cutoff'],
            min_t_stat=cointegration_config['min_t_stat'],
            min_correlation=cointegration_config['min_correlation'],
            require_adf=cointegration_config['require_adf'],
            max_pairs=None
        )
        
        cointegrated_pairs = cointegrated_pairs[cointegrated_pairs['cointegrated'] == True]
        
        cointegrated_pairs.to_csv(
            os.path.join(self.results_dir, 'cointegrated_pairs.csv'), 
            index=False
        )
        
        print(f"\nTop 20 Cointegrated Pairs:")
        display_df = cointegrated_pairs.head(20).copy()
        display_df['half_life'] = display_df['half_life'].apply(lambda x: f"{x:.1f} days" if np.isfinite(x) else "inf")
        display_df['p_value'] = display_df['p_value'].apply(lambda x: f"{x:.6f}")
        display_df['correlation'] = display_df['correlation'].apply(lambda x: f"{x:.4f}")
        print(display_df[['rank', 'ticker1', 'ticker2', 'half_life', 'p_value', 'correlation']].to_string(index=False))
        print(f"\nTotal pairs found: {len(cointegrated_pairs)}")
        
        return cointegrated_pairs
    
    # ========================================================================
    # STEP 2.5: Grid Search for Optimal Parameters
    # ========================================================================
    def _step_2_5_grid_search(self, train_df: pd.DataFrame, cointegrated_pairs: pd.DataFrame) -> Dict:
        """Step 2.5: Grid search for optimal parameters (per-pair)"""
        print("\n" + "="*80)
        print("STEP 2.5: GRID SEARCH - PER-PAIR PARAMETER OPTIMIZATION (Training Data)")
        print("="*80)
        
        pipeline_config = self.config['pipeline']
        backtest_config = self.config['backtest']
        
        num_pairs_to_optimize = min(backtest_config['num_pairs'], len(cointegrated_pairs))
        grid_search_pairs = cointegrated_pairs.head(num_pairs_to_optimize)
        
        optimal_params_by_pair = {}
        
        if len(grid_search_pairs) == 0:
            print("\n⚠️  Not enough cointegrated pairs for grid search.")
            return optimal_params_by_pair
        
        param_grid = {
            'spread_window': pipeline_config.get('grid_search_spread_windows', [30, 50, 70]),
            'zscore_window': pipeline_config.get('grid_search_zscore_windows', [30, 50, 70]),
            'entry_threshold': pipeline_config.get('grid_search_entry_thresholds', [1.0, 1.5, 2.0, 2.5]),
            'exit_threshold': pipeline_config.get('grid_search_exit_thresholds', [0.0, 0.5, 1.0]),
        }
        
        print(f"\nParameter Grid (per pair):")
        for param, values in param_grid.items():
            print(f"  {param}: {values}")
        
        total_combinations = np.prod([len(v) for v in param_grid.values()])
        print(f"\nTotal combinations per pair: {total_combinations}")
        print(f"Pairs to optimize: {len(grid_search_pairs)}")
        
        optimizer = GridSearchOptimizer(
            initial_capital=pipeline_config['initial_capital'],
            transaction_cost=pipeline_config['transaction_cost'],
            position_size=pipeline_config['position_size'],
            n_jobs=-1
        )
        
        all_grid_results = []
        
        for rank, (idx, pair) in enumerate(grid_search_pairs.iterrows(), 1):
            ticker1 = pair['ticker1']
            ticker2 = pair['ticker2']
            pair_name = f"{ticker1}_{ticker2}"
            
            print(f"\n{'='*80}")
            print(f"PAIR {rank}/{len(grid_search_pairs)}: {pair_name}")
            print(f"{'='*80}")
            
            if ticker1 not in train_df.columns or ticker2 not in train_df.columns:
                print(f"⚠️  SKIPPING: One or both tickers not in training data")
                continue
            
            pair_df = train_df[[ticker1, ticker2]].dropna()
            
            if len(pair_df) < 100:
                print(f"⚠️  SKIPPING: Insufficient training data ({len(pair_df)} days)")
                continue
            
            print(f"Training data: {len(pair_df)} days")
            
            single_pair_df = pd.DataFrame({
                ticker1: pair_df[ticker1],
                ticker2: pair_df[ticker2]
            })
            
            print(f"\nRunning grid search for {pair_name}...")
            pair_grid_results = optimizer.grid_search_single_pair(
                single_pair_df,
                ticker1,
                ticker2,
                param_grid,
                metric='sharpe_ratio'
            )
            
            if len(pair_grid_results) > 0:
                best_result = pair_grid_results.iloc[0]
                
                best_params = {
                    'spread_window': int(best_result['spread_window']),
                    'zscore_window': int(best_result['zscore_window']),
                    'entry_threshold': float(best_result['entry_threshold']),
                    'exit_threshold': float(best_result['exit_threshold']),
                }
                
                optimal_params_by_pair[pair_name] = best_params
                
                print(f"\n✓ BEST PARAMETERS FOR {pair_name}:")
                print(f"  Spread Window: {best_params['spread_window']} days")
                print(f"  Z-Score Window: {best_params['zscore_window']} days")
                print(f"  Entry Threshold: ±{best_params['entry_threshold']:.2f}")
                print(f"  Exit Threshold: ±{best_params['exit_threshold']:.2f}")
                print(f"  Expected Sharpe Ratio: {best_result['sharpe_ratio']:.3f}")
                print(f"  Expected Return: {best_result['total_return']:.2%}")
                
                pair_grid_results['pair_name'] = pair_name
                pair_grid_results['ticker1'] = ticker1
                pair_grid_results['ticker2'] = ticker2
                all_grid_results.append(pair_grid_results)
        
        if all_grid_results:
            combined_grid_results = pd.concat(all_grid_results, ignore_index=True)
            combined_grid_results.to_csv(
                os.path.join(self.results_dir, 'grid_search_results.csv'),
                index=False
            )
            
            summary_data = []
            for pair_name, params in optimal_params_by_pair.items():
                summary_data.append({
                    'pair': pair_name,
                    'spread_window': params['spread_window'],
                    'zscore_window': params['zscore_window'],
                    'entry_threshold': f"{params['entry_threshold']:.2f}",
                    'exit_threshold': f"{params['exit_threshold']:.2f}"
                })
            
            summary_df = pd.DataFrame(summary_data)
            summary_df.to_csv(
                os.path.join(self.results_dir, 'optimal_parameters_by_pair.csv'),
                index=False
            )
            
            print(f"\n✓ Grid search results saved to {self.results_dir}")
        
        return optimal_params_by_pair
    
    # ========================================================================
    # STEPS 3-7: Backtest Pipeline for Each Pair
    # ========================================================================
    def _steps_3_7_backtest_pipeline(self, oos_df: pd.DataFrame, cointegrated_pairs: pd.DataFrame, 
                                      optimal_params_by_pair: Dict) -> Tuple[List[Dict], List]:
        """Steps 3-7: Run backtest pipeline for each pair on out-of-sample data"""
        print("\n" + "="*80)
        print("STEPS 3-7: BACKTEST PIPELINE (Out-of-Sample Data)")
        print("="*80)
        
        pipeline_config = self.config['pipeline']
        backtest_config = self.config['backtest']
        
        good_pairs = cointegrated_pairs.head(backtest_config['num_pairs'])
        
        if len(good_pairs) == 0:
            raise Exception("No cointegrated pairs found for backtesting!")
        
        print(f"\nRunning backtests for {len(good_pairs)} pairs with optimized parameters...\n")
        
        all_results = []
        backtest_summary = []
        
        for rank, (idx, pair) in enumerate(good_pairs.iterrows(), 1):
            ticker1 = pair['ticker1']
            ticker2 = pair['ticker2']
            pair_name = f"{ticker1}_{ticker2}"
            
            print(f"\n{'='*80}")
            print(f"PAIR {rank}/{len(good_pairs)}: {ticker1} vs {ticker2}")
            print(f"{'='*80}")
            
            if ticker1 not in oos_df.columns or ticker2 not in oos_df.columns:
                print(f"⚠️  SKIPPING: Tickers not in out-of-sample data")
                continue
            
            oos_series1 = oos_df[ticker1].dropna()
            oos_series2 = oos_df[ticker2].dropna()
            
            common_dates = oos_series1.index.intersection(oos_series2.index)
            oos_series1 = oos_series1[common_dates]
            oos_series2 = oos_series2[common_dates]
            
            if len(oos_series1) == 0:
                print(f"⚠️  SKIPPING: No common dates in out-of-sample data")
                continue
            
            print(f"\nBacktest Period (Out-of-Sample):")
            print(f"  Start: {oos_series1.index.min()}")
            print(f"  End: {oos_series1.index.max()}")
            print(f"  Days: {len(oos_series1)}")
            
            # Get optimal parameters or use defaults
            if optimal_params_by_pair and pair_name in optimal_params_by_pair:
                pair_params = optimal_params_by_pair[pair_name]
                print(f"\n✓ Using optimized parameters from grid search")
            else:
                pair_params = {
                    'spread_window': pipeline_config['spread_window'],
                    'zscore_window': pipeline_config['zscore_window'],
                    'entry_threshold': pipeline_config['entry_threshold'],
                    'exit_threshold': pipeline_config['exit_threshold'],
                }
                print(f"\n⚠️  Using default parameters")
            
            print(f"  Spread Window: {pair_params['spread_window']} days")
            print(f"  Z-Score Window: {pair_params['zscore_window']} days")
            print(f"  Entry Threshold: ±{pair_params['entry_threshold']:.2f}")
            print(f"  Exit Threshold: ±{pair_params['exit_threshold']:.2f}")
            
            try:
                # Steps 3-7 for this pair
                result = self._run_pair_backtest(
                    oos_series1, oos_series2, ticker1, ticker2,
                    pair_name, pair['p_value'],
                    pair_params, pipeline_config
                )
                
                all_results.append(result)
                
                backtest_summary.append({
                    'rank': rank,
                    'ticker1': ticker1,
                    'ticker2': ticker2,
                    'pair_name': pair_name,
                    'p_value': pair['p_value'],
                    'correlation': pair['correlation'],
                    'spread_window': pair_params['spread_window'],
                    'zscore_window': pair_params['zscore_window'],
                    'entry_threshold': pair_params['entry_threshold'],
                    'exit_threshold': pair_params['exit_threshold'],
                    'total_return': result['backtest_result'].total_return,
                    'annual_return': result['backtest_result'].annual_return,
                    'sharpe_ratio': result['backtest_result'].sharpe_ratio,
                    'max_drawdown': result['backtest_result'].max_drawdown,
                    'num_trades': len(result['backtest_result'].trades),
                    'win_rate': result['backtest_result'].win_rate,
                    'final_equity': result['backtest_result'].equity_curve.iloc[-1]
                })
                
            except Exception as e:
                print(f"⚠️  ERROR processing pair: {str(e)}")
                continue
        
        return backtest_summary, all_results
    
    def _run_pair_backtest(self, series1: pd.Series, series2: pd.Series,
                          ticker1: str, ticker2: str, pair_name: str, p_value: float,
                          pair_params: Dict, pipeline_config: Dict) -> Dict:
        """
        Run steps 3-7 for a single pair
        
        Returns:
            Dict with backtest_result and spread/zscore data
        """
        # Step 3: Rolling Spread Analysis
        print("\n" + "="*80)
        print("STEP 3: ROLLING SPREAD ANALYSIS")
        print("="*80)
        
        print(f"\nAnalyzing rolling spread for {ticker1} vs {ticker2}")
        print(f"Cointegration p-value: {p_value:.6f}")
        
        spread_series, rolling_betas = calculate_rolling_spread(
            series1, series2, window=pair_params['spread_window']
        )
        
        print(f"\nRolling spread statistics:")
        print(f"  Mean: {spread_series.mean():.4f}")
        print(f"  Std Dev: {spread_series.std():.4f}")
        print(f"  Min: {spread_series.min():.4f}")
        print(f"  Max: {spread_series.max():.4f}")
        
        print(f"\nRolling beta statistics:")
        print(f"  Mean: {rolling_betas.mean():.4f}")
        print(f"  Std Dev: {rolling_betas.std():.4f}")
        print(f"  Min: {rolling_betas.min():.4f}")
        print(f"  Max: {rolling_betas.max():.4f}")
        
        self._plot_rolling_spread(series1, series2, spread_series, rolling_betas, ticker1, ticker2, pair_name)
        
        # Step 4: Z-Score Signal Analysis
        print("\n" + "="*80)
        print("STEP 4: Z-SCORE SIGNAL ANALYSIS")
        print("="*80)
        
        zscore = calculate_zscore(spread_series, window=pair_params['zscore_window'])
        
        print(f"\nZ-score statistics:")
        print(f"  Mean: {zscore.mean():.6f}")
        print(f"  Std Dev: {zscore.std():.4f}")
        print(f"  Min: {zscore.min():.4f}")
        print(f"  Max: {zscore.max():.4f}")
        
        entry_signals = (zscore.abs() > pair_params['entry_threshold']).sum()
        extreme_signals = (zscore.abs() > (pair_params['entry_threshold'] + 1)).sum()
        print(f"\nTrading Signals (based on z-score thresholds):")
        print(f"  |z-score| > {pair_params['entry_threshold']}: {entry_signals} days")
        print(f"  |z-score| > {pair_params['entry_threshold'] + 1}: {extreme_signals} days")
        
        self._plot_zscore_signals(zscore, ticker1, ticker2, pair_name, pair_params)
        
        # Step 5: Backtest Trading Strategy
        print("\n" + "="*80)
        print("STEP 5: BACKTEST TRADING STRATEGY")
        print("="*80)
        
        backtester = SimpleBacktester(
            initial_capital=pipeline_config['initial_capital'],
            transaction_cost=pipeline_config['transaction_cost']
        )
        
        print(f"\nRunning backtest with:")
        print(f"  Initial Capital: ${backtester.initial_capital:,.2f}")
        print(f"  Entry Threshold: ±{pair_params['entry_threshold']} (z-score)")
        print(f"  Exit Threshold: ±{pair_params['exit_threshold']} (z-score)")
        print(f"  Position Size: {pipeline_config['position_size']:.0%}")
        print(f"  Transaction Cost: {pipeline_config['transaction_cost']:.2%}")
        
        backtest_result = backtester.backtest_pairs_strategy(
            spread_data=spread_series,
            signals=zscore,
            entry_threshold=pair_params['entry_threshold'],
            exit_threshold=pair_params['exit_threshold'],
            position_size=pipeline_config['position_size']
        )
        
        backtester.print_results(backtest_result)
        
        # Step 6: Trade Details and Visualization
        print("\n" + "="*80)
        print("STEP 6: TRADE EXECUTION DETAILS")
        print("="*80)
        
        if backtest_result.trades:
            trades_df = pd.DataFrame(backtest_result.trades)
            trades_df['duration'] = (trades_df['exit_date'] - trades_df['entry_date']).dt.days
            print("\n=== TRADES EXECUTED ===")
            print(trades_df.to_string())
            print(f"\nTotal Trades: {len(backtest_result.trades)}")
        else:
            print("No trades were executed.")
        
        self._plot_backtest_results(spread_series, zscore, backtest_result, ticker1, ticker2, pair_name, pair_params)
        
        # Step 7: Detailed Performance Analysis
        print("\n" + "="*80)
        print("STEP 7: DETAILED PERFORMANCE ANALYSIS")
        print("="*80)
        
        initial_equity = pipeline_config['initial_capital']
        final_equity = backtest_result.equity_curve.iloc[-1]
        
        print(f"\nStarting Equity: ${initial_equity:,.2f}")
        print(f"Ending Equity: ${final_equity:,.2f}")
        print(f"Total P&L: ${final_equity - initial_equity:,.2f}")
        print(f"Total Return: {backtest_result.total_return:.2%}")
        print(f"Annual Return: {backtest_result.annual_return:.2%}")
        
        print(f"\nRisk Metrics:")
        print(f"  Sharpe Ratio: {backtest_result.sharpe_ratio:.2f}")
        print(f"  Max Drawdown: {backtest_result.max_drawdown:.2%}")
        
        print(f"\nTrade Statistics:")
        print(f"  Total Trades: {len(backtest_result.trades)}")
        print(f"  Win Rate: {backtest_result.win_rate:.2%}")
        
        if backtest_result.trades:
            winning_trades = sum(1 for t in backtest_result.trades if t['pnl'] > 0)
            losing_trades = len(backtest_result.trades) - winning_trades
            total_pnl = sum(t['pnl'] for t in backtest_result.trades)
            avg_win = sum(t['pnl'] for t in backtest_result.trades if t['pnl'] > 0) / winning_trades if winning_trades > 0 else 0
            avg_loss = sum(t['pnl'] for t in backtest_result.trades if t['pnl'] <= 0) / losing_trades if losing_trades > 0 else 0
            
            print(f"  Winning Trades: {winning_trades}")
            print(f"  Losing Trades: {losing_trades}")
            print(f"  Total PnL: ${total_pnl:,.2f}")
            print(f"  Avg Win: ${avg_win:,.2f}")
            print(f"  Avg Loss: ${avg_loss:,.2f}")
            if avg_loss != 0:
                print(f"  Profit Factor: {abs(avg_win / avg_loss):.2f}")
        
        print("\n" + "="*80)
        print("PAIR BACKTEST COMPLETE!")
        print("="*80 + "\n")
        
        return {
            'ticker1': ticker1,
            'ticker2': ticker2,
            'backtest_result': backtest_result,
            'spread_series': spread_series,
            'zscore': zscore,
            'rolling_betas': rolling_betas
        }
    
    # ========================================================================
    # Plotting Methods (Steps 3-7)
    # ========================================================================
    def _plot_rolling_spread(self, series1: pd.Series, series2: pd.Series, 
                            spread_series: pd.Series, rolling_betas: pd.Series, 
                            ticker1: str, ticker2: str, pair_name: str) -> None:
        """Plot rolling spread analysis"""
        print("\nGenerating rolling spread visualization...")
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 10))
        
        ax1_twin = ax1.twinx()
        ax1.plot(series1, label=ticker1, color='blue', alpha=0.7)
        ax1_twin.plot(series2, label=ticker2, color='red', alpha=0.7)
        ax1.set_ylabel(f'{ticker1} Price', color='blue')
        ax1_twin.set_ylabel(f'{ticker2} Price', color='red')
        ax1.set_title(f'Price Series: {ticker1} vs {ticker2}')
        ax1.legend(loc='upper left')
        ax1_twin.legend(loc='upper right')
        
        ax2.plot(rolling_betas, label='Rolling Beta (60-day)', color='green', linewidth=1.5)
        ax2.axhline(rolling_betas.mean(), color='black', linestyle='--', 
                    label=f'Mean Beta: {rolling_betas.mean():.4f}')
        ax2.set_ylabel('Beta (Hedge Ratio)')
        ax2.set_title('Time-Varying Hedge Ratio (Rolling Beta)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        ax3.plot(spread_series, label='Rolling Spread', color='purple', linewidth=1.5)
        ax3.axhline(spread_series.mean(), color='black', linestyle='--', label='Mean')
        ax3.axhline(spread_series.mean() + spread_series.std(), color='red', linestyle='--', 
                    alpha=0.5, label='+1 Std Dev')
        ax3.axhline(spread_series.mean() - spread_series.std(), color='green', linestyle='--', 
                    alpha=0.5, label='-1 Std Dev')
        ax3.fill_between(spread_series.index, spread_series.mean() - spread_series.std(), 
                          spread_series.mean() + spread_series.std(), alpha=0.1, color='gray')
        ax3.set_ylabel('Spread')
        ax3.set_xlabel('Date')
        ax3.set_title('Rolling Stationary Spread')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_dir, f'{pair_name}_rolling_spread_analysis.png'), 
                   dpi=100, bbox_inches='tight')
        plt.close()
    
    def _plot_zscore_signals(self, zscore: pd.Series, ticker1: str, ticker2: str, 
                            pair_name: str, pair_params: Dict) -> None:
        """Plot z-score signals"""
        print("\nGenerating z-score visualization...")
        fig, ax = plt.subplots(figsize=(14, 6))
        
        ax.plot(zscore, label='Z-score', color='blue', linewidth=1.5)
        ax.axhline(0, color='black', linestyle='-', alpha=0.3, linewidth=0.5)
        ax.axhline(pair_params['entry_threshold'], color='red', linestyle='--', alpha=0.5, 
                  label=f"Entry threshold (±{pair_params['entry_threshold']})")
        ax.axhline(-pair_params['entry_threshold'], color='red', linestyle='--', alpha=0.5)
        ax.axhline(pair_params['entry_threshold'] + 1, color='darkred', linestyle='--', alpha=0.5, 
                  label=f"Extreme threshold (±{pair_params['entry_threshold'] + 1})")
        ax.axhline(-(pair_params['entry_threshold'] + 1), color='darkred', linestyle='--', alpha=0.5)
        
        ax.fill_between(zscore.index, pair_params['entry_threshold'], 
                       pair_params['entry_threshold'] + 1, alpha=0.1, color='green', 
                       label='Sell region')
        ax.fill_between(zscore.index, -pair_params['entry_threshold'], 
                       -(pair_params['entry_threshold'] + 1), alpha=0.1, color='red', 
                       label='Buy region')
        
        ax.set_ylabel('Z-Score')
        ax.set_xlabel('Date')
        ax.set_title(f'Z-Score of Rolling Spread: {ticker1} vs {ticker2}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_dir, f'{pair_name}_zscore_signals.png'), 
                   dpi=100, bbox_inches='tight')
        plt.close()
    
    def _plot_backtest_results(self, spread_series: pd.Series, zscore: pd.Series,
                               backtest_result: BacktestResult, 
                               ticker1: str, ticker2: str, pair_name: str, pair_params: Dict) -> None:
        """Plot backtest results with detailed trade markers"""
        print("\nGenerating backtest results visualization...")
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10))
        
        # ...existing code...
        initial_capital = 100000.0
        ax1.plot(backtest_result.equity_curve, label='Equity Curve', color='blue', linewidth=2)
        ax1.axhline(initial_capital, color='black', linestyle='--', alpha=0.5, 
                   label='Initial Capital')
        ax1.fill_between(backtest_result.equity_curve.index, initial_capital, 
                         backtest_result.equity_curve.values,
                         where=(backtest_result.equity_curve.values >= initial_capital), 
                         alpha=0.1, color='green', label='Profit')
        ax1.fill_between(backtest_result.equity_curve.index, initial_capital, 
                         backtest_result.equity_curve.values,
                         where=(backtest_result.equity_curve.values < initial_capital), 
                         alpha=0.1, color='red', label='Loss')
        
        for trade in backtest_result.trades:
            entry_idx = backtest_result.equity_curve.index.get_loc(trade['entry_date'])
            exit_idx = backtest_result.equity_curve.index.get_loc(trade['exit_date'])
            
            entry_equity = backtest_result.equity_curve.iloc[entry_idx]
            exit_equity = backtest_result.equity_curve.iloc[exit_idx]
            
            if trade['pnl'] > 0:
                ax1.scatter(trade['entry_date'], entry_equity, color='darkgreen', marker='^', 
                           s=150, zorder=5, edgecolors='black', linewidth=1.5)
                ax1.scatter(trade['exit_date'], exit_equity, color='lime', marker='o', 
                           s=150, zorder=5, edgecolors='darkgreen', linewidth=1.5)
            else:
                ax1.scatter(trade['entry_date'], entry_equity, color='darkred', marker='^', 
                           s=150, zorder=5, edgecolors='black', linewidth=1.5)
                ax1.scatter(trade['exit_date'], exit_equity, color='salmon', marker='o', 
                           s=150, zorder=5, edgecolors='darkred', linewidth=1.5)
        
        ax1.set_ylabel('Equity ($)', fontsize=11)
        ax1.set_title('Backtest Equity Curve with Trade Markers', fontsize=12, fontweight='bold')
        ax1.legend(loc='best', fontsize=9)
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Spread with trade markers
        mean_spread = spread_series.mean()
        std_spread = spread_series.std()
        entry_upper = mean_spread + pair_params['entry_threshold'] * std_spread
        entry_lower = mean_spread - pair_params['entry_threshold'] * std_spread
        exit_upper = mean_spread + pair_params['exit_threshold'] * std_spread
        exit_lower = mean_spread - pair_params['exit_threshold'] * std_spread
        
        ax2.plot(spread_series, label='Spread', color='blue', linewidth=1.5)
        ax2.axhline(mean_spread, color='black', linestyle='-', alpha=0.3, label='Mean')
        ax2.axhline(entry_upper, color='red', linestyle='--', alpha=0.5, 
                   label=f"Entry Threshold (±{pair_params['entry_threshold']}σ)")
        ax2.axhline(entry_lower, color='red', linestyle='--', alpha=0.5)
        ax2.axhline(exit_upper, color='orange', linestyle=':', alpha=0.5, 
                   label=f"Exit Threshold (±{pair_params['exit_threshold']}σ)")
        ax2.axhline(exit_lower, color='orange', linestyle=':', alpha=0.5)
        
        for i, trade in enumerate(backtest_result.trades):
            try:
                entry_idx = spread_series.index.get_loc(trade['entry_date'])
                exit_idx = spread_series.index.get_loc(trade['exit_date'])
                
                entry_spread = spread_series.iloc[entry_idx]
                exit_spread = spread_series.iloc[exit_idx]
                
                is_winning = trade['pnl'] > 0
                position_type = trade['position']
                
                if position_type == 1:
                    if is_winning:
                        ax2.scatter(trade['entry_date'], entry_spread, color='darkgreen', marker='^', 
                                   s=200, zorder=5, edgecolors='black', linewidth=1.5)
                        ax2.scatter(trade['exit_date'], exit_spread, color='lime', marker='o', 
                                   s=200, zorder=5, edgecolors='darkgreen', linewidth=2)
                        ax2.plot([trade['entry_date'], trade['exit_date']], [entry_spread, exit_spread], 
                                color='darkgreen', linewidth=1.5, linestyle='-', alpha=0.6)
                    else:
                        ax2.scatter(trade['entry_date'], entry_spread, color='darkred', marker='^', 
                                   s=200, zorder=5, edgecolors='black', linewidth=1.5)
                        ax2.scatter(trade['exit_date'], exit_spread, color='salmon', marker='o', 
                                   s=200, zorder=5, edgecolors='darkred', linewidth=2)
                        ax2.plot([trade['entry_date'], trade['exit_date']], [entry_spread, exit_spread], 
                                color='darkred', linewidth=1.5, linestyle='-', alpha=0.6)
                else:
                    if is_winning:
                        ax2.scatter(trade['entry_date'], entry_spread, color='darkgreen', marker='v', 
                                   s=200, zorder=5, edgecolors='black', linewidth=1.5)
                        ax2.scatter(trade['exit_date'], exit_spread, color='lime', marker='o', 
                                   s=200, zorder=5, edgecolors='darkgreen', linewidth=2)
                        ax2.plot([trade['entry_date'], trade['exit_date']], [entry_spread, exit_spread], 
                                color='darkgreen', linewidth=1.5, linestyle='-', alpha=0.6)
                    else:
                        ax2.scatter(trade['entry_date'], entry_spread, color='darkred', marker='v', 
                                   s=200, zorder=5, edgecolors='black', linewidth=1.5)
                        ax2.scatter(trade['exit_date'], exit_spread, color='salmon', marker='o', 
                                   s=200, zorder=5, edgecolors='darkred', linewidth=2)
                        ax2.plot([trade['entry_date'], trade['exit_date']], [entry_spread, exit_spread], 
                                color='darkred', linewidth=1.5, linestyle='-', alpha=0.6)
            except KeyError:
                continue
        
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], marker='^', color='w', markerfacecolor='darkgreen', markersize=10, 
                   markeredgecolor='black', markeredgewidth=1.5, label='LONG Entry (Winning)', linestyle='None'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='lime', markersize=10, 
                   markeredgecolor='darkgreen', markeredgewidth=1.5, label='Exit (Winning)', linestyle='None'),
            Line2D([0], [0], marker='^', color='w', markerfacecolor='darkred', markersize=10, 
                   markeredgecolor='black', markeredgewidth=1.5, label='LONG Entry (Losing)', linestyle='None'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='salmon', markersize=10, 
                   markeredgecolor='darkred', markeredgewidth=1.5, label='Exit (Losing)', linestyle='None'),
            Line2D([0], [0], marker='v', color='w', markerfacecolor='darkgreen', markersize=10, 
                   markeredgecolor='black', markeredgewidth=1.5, label='SHORT Entry (Winning)', linestyle='None'),
            Line2D([0], [0], marker='v', color='w', markerfacecolor='darkred', markersize=10, 
                   markeredgecolor='black', markeredgewidth=1.5, label='SHORT Entry (Losing)', linestyle='None'),
        ]
        
        ax2.set_ylabel('Spread', fontsize=11)
        ax2.set_xlabel('Date', fontsize=11)
        ax2.set_title(f'Spread with Trade Entry/Exit Points: {ticker1} vs {ticker2}', 
                     fontsize=12, fontweight='bold')
        ax2.legend(handles=legend_elements, loc='best', fontsize=9)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_dir, f'{pair_name}_backtest_results.png'), 
                   dpi=100, bbox_inches='tight')
        plt.close()
    
    # ========================================================================
    # Combined Results Analysis
    # ========================================================================
    def _combined_results_analysis(self, backtest_data: Tuple[List[Dict], List]):
        """Combined analysis and visualization"""
        backtest_summary, all_results = backtest_data
        
        print("\n" + "="*80)
        print("COMBINED RESULTS SUMMARY")
        print("="*80)
        
        if len(backtest_summary) == 0:
            print("\n⚠️  No pairs were successfully backtested!")
            return
        
        summary_df = pd.DataFrame(backtest_summary)
        print(f"\nTotal pairs backtested: {len(summary_df)}")
        
        print("\n" + "="*80)
        print("DETAILED PERFORMANCE TABLE")
        print("="*80)
        display_df = summary_df.copy()
        display_df['total_return'] = display_df['total_return'].apply(lambda x: f"{x:.2%}")
        display_df['sharpe_ratio'] = display_df['sharpe_ratio'].apply(lambda x: f"{x:.2f}")
        display_df['max_drawdown'] = display_df['max_drawdown'].apply(lambda x: f"{x:.2%}")
        display_df['win_rate'] = display_df['win_rate'].apply(lambda x: f"{x:.2%}")
        
        print(display_df[['ticker1', 'ticker2', 'total_return', 'sharpe_ratio', 'max_drawdown', 'num_trades', 'win_rate']].to_string(index=False))
        
        summary_df.to_csv(
            os.path.join(self.results_dir, 'backtest_summary.csv'),
            index=False
        )
        
        print(f"\n" + "="*80)
        print("AGGREGATE STATISTICS")
        print("="*80)
        print(f"\nAverage Metrics:")
        print(f"  Total Return: {summary_df['total_return'].mean():.2%}")
        print(f"  Sharpe Ratio: {summary_df['sharpe_ratio'].mean():.2f}")
        print(f"  Max Drawdown: {summary_df['max_drawdown'].mean():.2%}")
        print(f"  Win Rate: {summary_df['win_rate'].mean():.2%}")
        print(f"  Avg Trades: {summary_df['num_trades'].mean():.0f}")
        
        print(f"\nBest Performers:")
        best_return = summary_df.loc[summary_df['total_return'].idxmax()]
        print(f"  Highest Return: {best_return['ticker1']}-{best_return['ticker2']} ({best_return['total_return']:.2%})")
        
        best_sharpe = summary_df.loc[summary_df['sharpe_ratio'].idxmax()]
        print(f"  Highest Sharpe: {best_sharpe['ticker1']}-{best_sharpe['ticker2']} ({best_sharpe['sharpe_ratio']:.2f})")
        
        if len(all_results) > 0:
            self._plot_combined_results(summary_df, all_results)
        
        print(f"\n{'='*80}")
        print("ANALYSIS COMPLETE!")
        print("="*80)
        print(f"\nResults saved to: {self.results_dir}")
        print(f"Plots saved to: {self.plots_dir}")
    
    def _plot_combined_results(self, summary_df: pd.DataFrame, all_results: List):
        """Plot combined equity curves and analysis"""
        print("\nGenerating combined analysis plots...")
        
        fig, ax = plt.subplots(figsize=(14, 8))
        
        for i, result in enumerate(all_results):
            ticker1 = summary_df.iloc[i]['ticker1']
            ticker2 = summary_df.iloc[i]['ticker2']
            ax.plot(result['backtest_result'].equity_curve, 
                   label=f"{ticker1}-{ticker2}", alpha=0.6, linewidth=1.5)
        
        ensemble_equity = pd.DataFrame({
            f"pair_{i}": all_results[i]['backtest_result'].equity_curve 
            for i in range(len(all_results))
        }).mean(axis=1)
        
        ax.plot(ensemble_equity, label='Ensemble', color='black', linewidth=3, linestyle='--')
        ax.axhline(100000, color='red', linestyle=':', alpha=0.5, linewidth=2)
        
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Equity ($)', fontsize=12)
        ax.set_title('Combined Equity Curves - All Pairs', fontsize=14, fontweight='bold')
        ax.legend(loc='best', fontsize=9, ncol=2)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_dir, 'combined_equity_curve.png'), dpi=100, bbox_inches='tight')
        plt.close()
        print("✓ Combined equity curve saved")


def main():
    """Main entry point"""
    pipeline = StrategyPipeline(config_file='config.yml')
    pipeline.run()


if __name__ == '__main__':
    main()
