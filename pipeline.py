# """
# Pipeline runner for pairs trading strategy.
# Encapsulates steps 3-7: spread analysis, z-score signals, backtesting, and visualization.
# """

# import os
# import warnings
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# from dataclasses import dataclass
# from typing import Tuple, Optional, Dict, Any

# from stats import calculate_rolling_spread, calculate_zscore
# from backtest import SimpleBacktester, BacktestResult


# @dataclass
# class PipelineConfig:
#     """Configuration for the pipeline"""
#     pair_index: int = 50
#     window: int = 60
#     zscore_window: int = 60
#     entry_threshold: float = 2.0
#     exit_threshold: float = 0.5
#     position_size: float = 0.5
#     initial_capital: float = 100000.0
#     transaction_cost: float = 0.0
#     output_dir: str = "."


# @dataclass
# class PipelineResult:
#     """Results from the pipeline"""
#     ticker1: str
#     ticker2: str
#     p_value: float
#     spread_series: pd.Series
#     rolling_betas: pd.Series
#     zscore: pd.Series
#     backtest_result: BacktestResult
#     pair_info: Dict[str, Any]


# class PairsStrategyPipeline:
#     """Pipeline runner for pairs trading strategy"""
    
#     def __init__(self, config: Optional[PipelineConfig] = None):
#         """
#         Initialize the pipeline
        
#         Args:
#             config: Pipeline configuration
#         """
#         self.config = config or PipelineConfig()
        
#     def run(self, 
#             series1: pd.Series,
#             series2: pd.Series,
#             ticker1: str = "Asset1",
#             ticker2: str = "Asset2",
#             p_value: float = None,
#             pair_name: str = None) -> PipelineResult:
#         """
#         Run the complete pipeline from step 3 onwards
        
#         Args:
#             series1: First price series
#             series2: Second price series
#             ticker1: Name/ticker of first asset
#             ticker2: Name/ticker of second asset
#             p_value: Cointegration p-value (optional)
#             pair_name: Name for plot files (optional, defaults to ticker1_ticker2)
            
#         Returns:
#             PipelineResult with all results
#         """
#         if pair_name is None:
#             pair_name = f"{ticker1}_{ticker2}"
        
#         self.pair_name = pair_name
        
#         print("\n" + "="*80)
#         print("PAIRS STRATEGY PIPELINE - STEPS 3-7")
#         print("="*80)
        
#         # Step 3: Rolling Spread Analysis
#         spread_series, rolling_betas, pair_info = self._step_rolling_spread(
#             series1, series2, ticker1, ticker2, p_value
#         )
        
#         # Step 4: Z-Score Signal Analysis
#         zscore = self._step_zscore_signals(spread_series, ticker1, ticker2)
        
#         # Step 5: Backtest Trading Strategy
#         backtest_result = self._step_backtest(spread_series, zscore, ticker1, ticker2)
        
#         # Step 6: Trade Details and Visualization
#         self._step_trade_details_and_viz(
#             spread_series, rolling_betas, zscore, backtest_result, 
#             ticker1, ticker2
#         )
        
#         # Step 7: Detailed Performance Analysis
#         self._step_performance_analysis(backtest_result)
        
#         return PipelineResult(
#             ticker1=ticker1,
#             ticker2=ticker2,
#             p_value=pair_info['p_value'],
#             spread_series=spread_series,
#             rolling_betas=rolling_betas,
#             zscore=zscore,
#             backtest_result=backtest_result,
#             pair_info=pair_info
#         )
    
#     def _step_rolling_spread(self, series1: pd.Series, series2: pd.Series,
#                             ticker1: str, ticker2: str, p_value: float = None) \
#             -> Tuple[pd.Series, pd.Series, Dict]:
#         """Step 3: Rolling Spread Analysis"""
#         print("\n" + "="*80)
#         print("STEP 3: ROLLING SPREAD ANALYSIS")
#         print("="*80)
        
#         print(f"\nAnalyzing rolling spread for {ticker1} vs {ticker2}")
#         if p_value is not None:
#             print(f"Cointegration p-value: {p_value:.6f}")
        
#         # Calculate rolling spread
#         spread_series, rolling_betas = calculate_rolling_spread(
#             series1,
#             series2,
#             window=self.config.window
#         )
        
#         # Print statistics
#         print(f"\nRolling spread statistics:")
#         print(f"  Mean: {spread_series.mean():.4f}")
#         print(f"  Std Dev: {spread_series.std():.4f}")
#         print(f"  Min: {spread_series.min():.4f}")
#         print(f"  Max: {spread_series.max():.4f}")
        
#         print(f"\nRolling beta statistics:")
#         print(f"  Mean: {rolling_betas.mean():.4f}")
#         print(f"  Std Dev: {rolling_betas.std():.4f}")
#         print(f"  Min: {rolling_betas.min():.4f}")
#         print(f"  Max: {rolling_betas.max():.4f}")
        
#         # Visualize
#         self._plot_rolling_spread(series1, series2, spread_series, rolling_betas, ticker1, ticker2)
        
#         pair_info = {
#             'ticker1': ticker1,
#             'ticker2': ticker2,
#             'p_value': p_value or 0.0,
#             'rank': -1
#         }
        
#         return spread_series, rolling_betas, pair_info
    
#     def _step_zscore_signals(self, spread_series: pd.Series, ticker1: str, ticker2: str) \
#             -> pd.Series:
#         """Step 4: Z-Score Signal Analysis"""
#         print("\n" + "="*80)
#         print("STEP 4: Z-SCORE SIGNAL ANALYSIS")
#         print("="*80)
        
#         zscore = calculate_zscore(spread_series, window=self.config.zscore_window)
        
#         print(f"\nZ-score statistics:")
#         print(f"  Mean: {zscore.mean():.6f}")
#         print(f"  Std Dev: {zscore.std():.4f}")
#         print(f"  Min: {zscore.min():.4f}")
#         print(f"  Max: {zscore.max():.4f}")
        
#         # Count trading signals
#         entry_signals = (zscore.abs() > self.config.entry_threshold).sum()
#         extreme_signals = (zscore.abs() > (self.config.entry_threshold + 1)).sum()
#         print(f"\nTrading Signals (based on z-score thresholds):")
#         print(f"  |z-score| > {self.config.entry_threshold}: {entry_signals} days")
#         print(f"  |z-score| > {self.config.entry_threshold + 1}: {extreme_signals} days")
        
#         # Visualize
#         self._plot_zscore_signals(zscore, ticker1, ticker2)
        
#         return zscore
    
#     def _step_backtest(self, spread_series: pd.Series, zscore: pd.Series, 
#                       ticker1: str, ticker2: str) -> BacktestResult:
#         """Step 5: Backtest Trading Strategy"""
#         print("\n" + "="*80)
#         print("STEP 5: BACKTEST TRADING STRATEGY")
#         print("="*80)
        
#         backtester = SimpleBacktester(
#             initial_capital=self.config.initial_capital,
#             transaction_cost=self.config.transaction_cost
#         )
        
#         print(f"\nRunning backtest with:")
#         print(f"  Initial Capital: ${backtester.initial_capital:,.2f}")
#         print(f"  Entry Threshold: ±{self.config.entry_threshold} (z-score)")
#         print(f"  Exit Threshold: ±{self.config.exit_threshold} (z-score)")
#         print(f"  Position Size: {self.config.position_size:.0%}")
#         print(f"  Transaction Cost: {self.config.transaction_cost:.2%}")
        
#         result = backtester.backtest_pairs_strategy(
#             spread_data=spread_series,
#             signals=zscore,
#             entry_threshold=self.config.entry_threshold,
#             exit_threshold=self.config.exit_threshold,
#             position_size=self.config.position_size
#         )
        
#         backtester.print_results(result)
        
#         return result
    
#     def _step_trade_details_and_viz(self, 
#                                     spread_series: pd.Series,
#                                     rolling_betas: pd.Series,
#                                     zscore: pd.Series,
#                                     backtest_result: BacktestResult,
#                                     ticker1: str, 
#                                     ticker2: str) -> None:
#         """Step 6: Trade Details and Visualization"""
#         print("\n" + "="*80)
#         print("STEP 6: TRADE EXECUTION DETAILS")
#         print("="*80)
        
#         if backtest_result.trades:
#             trades_df = pd.DataFrame(backtest_result.trades)
#             trades_df['duration'] = (trades_df['exit_date'] - trades_df['entry_date']).dt.days
#             print("\n=== TRADES EXECUTED ===")
#             print(trades_df.to_string())
#             print(f"\nTotal Trades: {len(backtest_result.trades)}")
#         else:
#             print("No trades were executed.")
        
#         # Visualize
#         self._plot_backtest_results(
#             spread_series, zscore, backtest_result, ticker1, ticker2
#         )
    
#     def _step_performance_analysis(self, result: BacktestResult) -> None:
#         """Step 7: Detailed Performance Analysis"""
#         print("\n" + "="*80)
#         print("STEP 7: DETAILED PERFORMANCE ANALYSIS")
#         print("="*80)
        
#         initial_equity = 100000.0  # Assuming from config
#         final_equity = result.equity_curve.iloc[-1]
        
#         print(f"\nStarting Equity: ${initial_equity:,.2f}")
#         print(f"Ending Equity: ${final_equity:,.2f}")
#         print(f"Total P&L: ${final_equity - initial_equity:,.2f}")
#         print(f"Total Return: {result.total_return:.2%}")
#         print(f"Annual Return: {result.annual_return:.2%}")
        
#         print(f"\nRisk Metrics:")
#         print(f"  Sharpe Ratio: {result.sharpe_ratio:.2f}")
#         print(f"  Max Drawdown: {result.max_drawdown:.2%}")
        
#         print(f"\nTrade Statistics:")
#         print(f"  Total Trades: {len(result.trades)}")
#         print(f"  Win Rate: {result.win_rate:.2%}")
        
#         if result.trades:
#             winning_trades = sum(1 for t in result.trades if t['pnl'] > 0)
#             losing_trades = len(result.trades) - winning_trades
#             total_pnl = sum(t['pnl'] for t in result.trades)
#             avg_win = sum(t['pnl'] for t in result.trades if t['pnl'] > 0) / winning_trades if winning_trades > 0 else 0
#             avg_loss = sum(t['pnl'] for t in result.trades if t['pnl'] <= 0) / losing_trades if losing_trades > 0 else 0
            
#             print(f"  Winning Trades: {winning_trades}")
#             print(f"  Losing Trades: {losing_trades}")
#             print(f"  Total PnL: ${total_pnl:,.2f}")
#             print(f"  Avg Win: ${avg_win:,.2f}")
#             print(f"  Avg Loss: ${avg_loss:,.2f}")
#             if avg_loss != 0:
#                 print(f"  Profit Factor: {abs(avg_win / avg_loss):.2f}")
        
#         print("\n" + "="*80)
#         print("PIPELINE COMPLETE!")
#         print("="*80)
#         print("\nGenerated files:")
#         print(f"  - {self.pair_name}_rolling_spread_analysis.png")
#         print(f"  - {self.pair_name}_zscore_signals.png")
#         print(f"  - {self.pair_name}_backtest_results.png")
#         print("="*80 + "\n")
    
#     def _plot_rolling_spread(self, series1: pd.Series, series2: pd.Series, 
#                             spread_series: pd.Series, rolling_betas: pd.Series, ticker1: str, ticker2: str) -> None:
#         """Plot rolling spread analysis"""
#         print("\nGenerating rolling spread visualization...")
#         fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 10))
        
#         # Plot 1: Price series
#         ax1_twin = ax1.twinx()
#         ax1.plot(series1, label=ticker1, color='blue', alpha=0.7)
#         ax1_twin.plot(series2, label=ticker2, color='red', alpha=0.7)
#         ax1.set_ylabel(f'{ticker1} Price', color='blue')
#         ax1_twin.set_ylabel(f'{ticker2} Price', color='red')
#         ax1.set_title(f'Price Series: {ticker1} vs {ticker2}')
#         ax1.legend(loc='upper left')
#         ax1_twin.legend(loc='upper right')
        
#         # Plot 2: Rolling beta
#         ax2.plot(rolling_betas, label='Rolling Beta (60-day)', color='green', linewidth=1.5)
#         ax2.axhline(rolling_betas.mean(), color='black', linestyle='--', 
#                     label=f'Mean Beta: {rolling_betas.mean():.4f}')
#         ax2.set_ylabel('Beta (Hedge Ratio)')
#         ax2.set_title('Time-Varying Hedge Ratio (Rolling Beta)')
#         ax2.legend()
#         ax2.grid(True, alpha=0.3)
        
#         # Plot 3: Rolling spread
#         ax3.plot(spread_series, label='Rolling Spread', color='purple', linewidth=1.5)
#         ax3.axhline(spread_series.mean(), color='black', linestyle='--', label='Mean')
#         ax3.axhline(spread_series.mean() + spread_series.std(), color='red', linestyle='--', 
#                     alpha=0.5, label='+1 Std Dev')
#         ax3.axhline(spread_series.mean() - spread_series.std(), color='green', linestyle='--', 
#                     alpha=0.5, label='-1 Std Dev')
#         ax3.fill_between(spread_series.index, spread_series.mean() - spread_series.std(), 
#                           spread_series.mean() + spread_series.std(), alpha=0.1, color='gray')
#         ax3.set_ylabel('Spread')
#         ax3.set_xlabel('Date')
#         ax3.set_title('Rolling Stationary Spread')
#         ax3.legend()
#         ax3.grid(True, alpha=0.3)
        
#         plt.tight_layout()
#         plt.savefig(os.path.join(self.config.output_dir, f'{self.pair_name}_rolling_spread_analysis.png'), 
#                    dpi=100, bbox_inches='tight')
#         plt.close()
    
#     def _plot_zscore_signals(self, zscore: pd.Series, ticker1: str, ticker2: str) -> None:
#         """Plot z-score signals"""
#         print("\nGenerating z-score visualization...")
#         fig, ax = plt.subplots(figsize=(14, 6))
        
#         ax.plot(zscore, label='Z-score', color='blue', linewidth=1.5)
#         ax.axhline(0, color='black', linestyle='-', alpha=0.3, linewidth=0.5)
#         ax.axhline(self.config.entry_threshold, color='red', linestyle='--', alpha=0.5, 
#                   label=f'Entry threshold (±{self.config.entry_threshold})')
#         ax.axhline(-self.config.entry_threshold, color='red', linestyle='--', alpha=0.5)
#         ax.axhline(self.config.entry_threshold + 1, color='darkred', linestyle='--', alpha=0.5, 
#                   label=f'Extreme threshold (±{self.config.entry_threshold + 1})')
#         ax.axhline(-(self.config.entry_threshold + 1), color='darkred', linestyle='--', alpha=0.5)
        
#         # Shade regions
#         ax.fill_between(zscore.index, self.config.entry_threshold, 
#                        self.config.entry_threshold + 1, alpha=0.1, color='green', 
#                        label='Sell region')
#         ax.fill_between(zscore.index, -self.config.entry_threshold, 
#                        -(self.config.entry_threshold + 1), alpha=0.1, color='red', 
#                        label='Buy region')
        
#         ax.set_ylabel('Z-Score')
#         ax.set_xlabel('Date')
#         ax.set_title(f'Z-Score of Rolling Spread: {ticker1} vs {ticker2}')
#         ax.legend()
#         ax.grid(True, alpha=0.3)
#         plt.tight_layout()
#         plt.savefig(os.path.join(self.config.output_dir, f'{self.pair_name}_zscore_signals.png'), 
#                    dpi=100, bbox_inches='tight')
#         plt.close()
    
#     def _plot_backtest_results(self, spread_series: pd.Series, zscore: pd.Series,
#                                backtest_result: BacktestResult, 
#                                ticker1: str, ticker2: str) -> None:
#         """Plot backtest results with detailed trade markers"""
#         print("\nGenerating backtest results visualization...")
#         fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10))
        
#         # Plot 1: Equity curve with trade markers
#         ax1.plot(backtest_result.equity_curve, label='Equity Curve', color='blue', linewidth=2)
#         ax1.axhline(self.config.initial_capital, color='black', linestyle='--', alpha=0.5, 
#                    label='Initial Capital')
#         ax1.fill_between(backtest_result.equity_curve.index, self.config.initial_capital, 
#                          backtest_result.equity_curve.values,
#                          where=(backtest_result.equity_curve.values >= self.config.initial_capital), 
#                          alpha=0.1, color='green', label='Profit')
#         ax1.fill_between(backtest_result.equity_curve.index, self.config.initial_capital, 
#                          backtest_result.equity_curve.values,
#                          where=(backtest_result.equity_curve.values < self.config.initial_capital), 
#                          alpha=0.1, color='red', label='Loss')
        
#         # Mark winning and losing trades on equity curve
#         for trade in backtest_result.trades:
#             entry_idx = backtest_result.equity_curve.index.get_loc(trade['entry_date'])
#             exit_idx = backtest_result.equity_curve.index.get_loc(trade['exit_date'])
            
#             entry_equity = backtest_result.equity_curve.iloc[entry_idx]
#             exit_equity = backtest_result.equity_curve.iloc[exit_idx]
            
#             if trade['pnl'] > 0:
#                 # Winning trade
#                 ax1.scatter(trade['entry_date'], entry_equity, color='darkgreen', marker='^', 
#                            s=150, zorder=5, edgecolors='black', linewidth=1.5, label='Entry (Win)' if trade == backtest_result.trades[0] else '')
#                 ax1.scatter(trade['exit_date'], exit_equity, color='lime', marker='o', 
#                            s=150, zorder=5, edgecolors='darkgreen', linewidth=1.5, label='Exit (Win)' if trade == backtest_result.trades[0] else '')
#             else:
#                 # Losing trade
#                 ax1.scatter(trade['entry_date'], entry_equity, color='darkred', marker='^', 
#                            s=150, zorder=5, edgecolors='black', linewidth=1.5, label='Entry (Loss)' if trade == backtest_result.trades[0] else '')
#                 ax1.scatter(trade['exit_date'], exit_equity, color='salmon', marker='o', 
#                            s=150, zorder=5, edgecolors='darkred', linewidth=1.5, label='Exit (Loss)' if trade == backtest_result.trades[0] else '')
        
#         ax1.set_ylabel('Equity ($)', fontsize=11)
#         ax1.set_title('Backtest Equity Curve with Trade Markers', fontsize=12, fontweight='bold')
#         ax1.legend(loc='best', fontsize=9)
#         ax1.grid(True, alpha=0.3)
        
#         # Plot 2: Spread with detailed trade markers
#         mean_spread = spread_series.mean()
#         std_spread = spread_series.std()
#         entry_upper = mean_spread + self.config.entry_threshold * std_spread
#         entry_lower = mean_spread - self.config.entry_threshold * std_spread
#         exit_upper = mean_spread + self.config.exit_threshold * std_spread
#         exit_lower = mean_spread - self.config.exit_threshold * std_spread
        
#         ax2.plot(spread_series, label='Spread', color='blue', linewidth=1.5)
#         ax2.axhline(mean_spread, color='black', linestyle='-', alpha=0.3, label='Mean')
#         ax2.axhline(entry_upper, color='red', linestyle='--', alpha=0.5, 
#                    label=f'Entry Threshold (±{self.config.entry_threshold}σ)')
#         ax2.axhline(entry_lower, color='red', linestyle='--', alpha=0.5)
#         ax2.axhline(exit_upper, color='orange', linestyle=':', alpha=0.5, 
#                    label=f'Exit Threshold (±{self.config.exit_threshold}σ)')
#         ax2.axhline(exit_lower, color='orange', linestyle=':', alpha=0.5)
        
#         # Mark trades with clear distinction
#         for i, trade in enumerate(backtest_result.trades):
#             try:
#                 entry_idx = spread_series.index.get_loc(trade['entry_date'])
#                 exit_idx = spread_series.index.get_loc(trade['exit_date'])
                
#                 entry_spread = spread_series.iloc[entry_idx]
#                 exit_spread = spread_series.iloc[exit_idx]
                
#                 is_winning = trade['pnl'] > 0
#                 position_type = trade['position']  # 1 = long, -1 = short
                
#                 if position_type == 1:
#                     # LONG position (went long spread, short ticker1 / long ticker2)
#                     if is_winning:
#                         # Entry: green triangle pointing up (bottom entry)
#                         ax2.scatter(trade['entry_date'], entry_spread, color='darkgreen', marker='^', 
#                                    s=200, zorder=5, edgecolors='black', linewidth=1.5)
#                         # Exit: lime circle (exit point)
#                         ax2.scatter(trade['exit_date'], exit_spread, color='lime', marker='o', 
#                                    s=200, zorder=5, edgecolors='darkgreen', linewidth=2)
#                         # Draw line connecting entry to exit
#                         ax2.plot([trade['entry_date'], trade['exit_date']], [entry_spread, exit_spread], 
#                                 color='darkgreen', linewidth=1.5, linestyle='-', alpha=0.6)
#                     else:
#                         # Entry: dark red triangle pointing up (bottom entry)
#                         ax2.scatter(trade['entry_date'], entry_spread, color='darkred', marker='^', 
#                                    s=200, zorder=5, edgecolors='black', linewidth=1.5)
#                         # Exit: salmon circle (exit point)
#                         ax2.scatter(trade['exit_date'], exit_spread, color='salmon', marker='o', 
#                                    s=200, zorder=5, edgecolors='darkred', linewidth=2)
#                         # Draw line connecting entry to exit
#                         ax2.plot([trade['entry_date'], trade['exit_date']], [entry_spread, exit_spread], 
#                                 color='darkred', linewidth=1.5, linestyle='-', alpha=0.6)
                
#                 else:  # position_type == -1
#                     # SHORT position (went short spread, long ticker1 / short ticker2)
#                     if is_winning:
#                         # Entry: green triangle pointing down (top entry)
#                         ax2.scatter(trade['entry_date'], entry_spread, color='darkgreen', marker='v', 
#                                    s=200, zorder=5, edgecolors='black', linewidth=1.5)
#                         # Exit: lime circle (exit point)
#                         ax2.scatter(trade['exit_date'], exit_spread, color='lime', marker='o', 
#                                    s=200, zorder=5, edgecolors='darkgreen', linewidth=2)
#                         # Draw line connecting entry to exit
#                         ax2.plot([trade['entry_date'], trade['exit_date']], [entry_spread, exit_spread], 
#                                 color='darkgreen', linewidth=1.5, linestyle='-', alpha=0.6)
#                     else:
#                         # Entry: dark red triangle pointing down (top entry)
#                         ax2.scatter(trade['entry_date'], entry_spread, color='darkred', marker='v', 
#                                    s=200, zorder=5, edgecolors='black', linewidth=1.5)
#                         # Exit: salmon circle (exit point)
#                         ax2.scatter(trade['exit_date'], exit_spread, color='salmon', marker='o', 
#                                    s=200, zorder=5, edgecolors='darkred', linewidth=2)
#                         # Draw line connecting entry to exit
#                         ax2.plot([trade['entry_date'], trade['exit_date']], [entry_spread, exit_spread], 
#                                 color='darkred', linewidth=1.5, linestyle='-', alpha=0.6)
                
#             except KeyError:
#                 # Handle case where trade dates don't exist in spread_series
#                 continue
        
#         # Add legend with clear explanations
#         from matplotlib.lines import Line2D
#         legend_elements = [
#             Line2D([0], [0], marker='^', color='w', markerfacecolor='darkgreen', markersize=10, 
#                    markeredgecolor='black', markeredgewidth=1.5, label='LONG Entry (Winning)', linestyle='None'),
#             Line2D([0], [0], marker='o', color='w', markerfacecolor='lime', markersize=10, 
#                    markeredgecolor='darkgreen', markeredgewidth=1.5, label='Exit (Winning)', linestyle='None'),
#             Line2D([0], [0], marker='^', color='w', markerfacecolor='darkred', markersize=10, 
#                    markeredgecolor='black', markeredgewidth=1.5, label='LONG Entry (Losing)', linestyle='None'),
#             Line2D([0], [0], marker='o', color='w', markerfacecolor='salmon', markersize=10, 
#                    markeredgecolor='darkred', markeredgewidth=1.5, label='Exit (Losing)', linestyle='None'),
#             Line2D([0], [0], marker='v', color='w', markerfacecolor='darkgreen', markersize=10, 
#                    markeredgecolor='black', markeredgewidth=1.5, label='SHORT Entry (Winning)', linestyle='None'),
#             Line2D([0], [0], marker='v', color='w', markerfacecolor='darkred', markersize=10, 
#                    markeredgecolor='black', markeredgewidth=1.5, label='SHORT Entry (Losing)', linestyle='None'),
#         ]
        
#         ax2.set_ylabel('Spread', fontsize=11)
#         ax2.set_xlabel('Date', fontsize=11)
#         ax2.set_title(f'Spread with Trade Entry/Exit Points: {ticker1} vs {ticker2}', 
#                      fontsize=12, fontweight='bold')
#         ax2.legend(handles=legend_elements, loc='best', fontsize=9)
#         ax2.grid(True, alpha=0.3)
        
#         plt.tight_layout()
#         plt.savefig(os.path.join(self.config.output_dir, f'{self.pair_name}_backtest_results.png'), 
#                    dpi=100, bbox_inches='tight')
#         plt.close()
