"""
Simple backtesting framework for pairs trading strategies.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Tuple, List, Dict, Optional


@dataclass
class BacktestResult:
    """Store backtest results"""
    total_return: float
    annual_return: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    trades: List[Dict]
    equity_curve: pd.Series


class SimpleBacktester:
    """Simple backtester for pairs trading strategies"""
    
    def __init__(self, 
                 initial_capital: float = 100000,
                 transaction_cost: float = 0.001):
        """
        Initialize backtester
        
        Args:
            initial_capital: Starting capital in dollars
            transaction_cost: Transaction cost as fraction (0.001 = 0.1%)
        """
        self.initial_capital = initial_capital
        self.transaction_cost = transaction_cost
    
    def backtest_pairs_strategy(self,
                                 spread_data: pd.DataFrame,
                                 signals: pd.Series,
                                 entry_threshold: float = 2.0,
                                 exit_threshold: float = 0.5,
                                 position_size: float = 0.5) -> BacktestResult:
        """
        Backtest a pairs trading strategy using z-score signals
        
        Args:
            price_data: DataFrame with columns ['ticker1', 'ticker2'] for two price series
            signals: Series of z-scores for the spread (index must align with price_data)
            entry_threshold: z-score threshold for entry
            exit_threshold: z-score threshold for exit
            position_size: Fraction of capital to allocate per trade
            
        Returns:
            BacktestResult with performance metrics
        """
        
        equity = self.initial_capital
        position = 0  # 0 = no position, 1 = long spread, -1 = short spread
        entry_price = None
        trades = []
        equity_curve = []
        
        for date, z_score in signals.items():
            if pd.isna(z_score):
                equity_curve.append(equity)
                continue
            
            # Entry logic
            if position == 0:
                if z_score > entry_threshold:
                    # Short spread (entry: z-score is high, expect mean-reversion down)
                    position = -1
                    entry_price = spread_data.loc[date]
                    entry_z_score = z_score
                    entry_date = date
                    entry_size = position_size
                elif z_score < -entry_threshold:
                    # Long spread (entry: z-score is low, expect mean-reversion up)
                    position = 1
                    entry_price = spread_data.loc[date]
                    entry_z_score = z_score
                    entry_date = date
                    entry_size = position_size
            
            # Exit logic
            elif position != 0:
                should_exit = False
                pnl = 0
                
                if position == 1 and z_score > -exit_threshold:
                    # Long position: exit when z-score recovers toward 0 (mean-reversion occurred)
                    should_exit = True
                    exit_price = spread_data.loc[date]
                    exit_z_score = z_score
                    pnl = (exit_price - entry_price) * entry_size  # Profit if exit_price > entry_price
                elif position == -1 and z_score < exit_threshold:
                    # Short position: exit when z-score recovers toward 0 (mean-reversion occurred)
                    should_exit = True
                    exit_price = spread_data.loc[date]
                    exit_z_score = z_score
                    pnl = (entry_price - exit_price) * entry_size  # Profit if entry_price > exit_price
                
                if should_exit:
                    # Apply transaction costs
                    transaction_cost_amount = equity * position_size * self.transaction_cost * 2
                    pnl_after_costs = pnl - transaction_cost_amount
                    equity += pnl_after_costs
                    
                    trades.append({
                        'entry_date': entry_date,
                        'exit_date': date,
                        'entry_price': entry_price,
                        'exit_price': exit_price,
                        'entry_z_score': entry_z_score,
                        'exit_z_score': exit_z_score,
                        'pnl': pnl_after_costs,
                        'position': position
                    })
                    
                    position = 0
                    entry_price = None
            
            equity_curve.append(equity)
        
        # Convert to series
        equity_curve = pd.Series(equity_curve, index=signals.index)
        
        # Calculate metrics
        total_return = (equity - self.initial_capital) / self.initial_capital
        years = len(signals) / 252  # Assuming daily data
        annual_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0
        
        # Sharpe ratio
        returns = equity_curve.pct_change().dropna()
        sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252) if len(returns) > 0 else 0
        
        # Max drawdown
        running_max = equity_curve.expanding().max()
        drawdown = (equity_curve - running_max) / running_max
        max_drawdown = drawdown.min()
        
        # Win rate
        if len(trades) > 0:
            winning_trades = sum(1 for t in trades if t['pnl'] > 0)
            win_rate = winning_trades / len(trades)
        else:
            win_rate = 0.0
        
        return BacktestResult(
            total_return=total_return,
            annual_return=annual_return,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            win_rate=win_rate,
            trades=trades,
            equity_curve=equity_curve
        )
    
    def print_results(self, result: BacktestResult):
        """Print backtest results in a readable format"""
        print("=" * 60)
        print("BACKTEST RESULTS")
        print("=" * 60)
        print(f"Initial Capital:        ${self.initial_capital:,.2f}")
        print(f"Final Equity:           ${result.equity_curve.iloc[-1]:,.2f}")
        print(f"Total Return:           {result.total_return:>8.2%}")
        print(f"Annual Return:          {result.annual_return:>8.2%}")
        print(f"Sharpe Ratio:           {result.sharpe_ratio:>8.2f}")
        print(f"Max Drawdown:           {result.max_drawdown:>8.2%}")
        print(f"Number of Trades:       {len(result.trades):>8d}")
        print(f"Win Rate:               {result.win_rate:>8.2%}")
        
        if len(result.trades) > 0:
            total_pnl = sum(t['pnl'] for t in result.trades)
            avg_trade = total_pnl / len(result.trades)
            print(f"Total PnL:              ${total_pnl:>10,.2f}")
            print(f"Average Trade PnL:      ${avg_trade:>10,.2f}")
        print("=" * 60)
