import pandas as pd
import yfinance as yf
from typing import List
import urllib.request

# Common S&P 500 tickers as a fallback
FALLBACK_SP500_TICKERS = [
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'TSLA', 'META', 'BERKB', 
    'JPM', 'JNJ', 'V', 'WMT', 'PG', 'XOM', 'MA', 'HD', 'MCD', 'BAC',
    'INTC', 'CSCO', 'CVX', 'PEP', 'KO', 'NFLX', 'ADBE', 'CMCSA'
]


def get_sp500_tickers(use_fallback: bool = False) -> List[str]:
    """
    Fetch S&P 500 tickers from Wikipedia.
    Falls back to a predefined list if the request fails.
    
    Args:
        use_fallback: If True, use predefined ticker list instead of fetching
        
    Returns:
        List[str]: List of S&P 500 ticker symbols
    """
    if use_fallback:
        print(f"Using fallback ticker list ({len(FALLBACK_SP500_TICKERS)} tickers)")
        return FALLBACK_SP500_TICKERS
    
    try:
        url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req) as response:
            table = pd.read_html(response)[0]
        tickers = table["Symbol"].tolist()
        print(f"Found {len(tickers)} tickers from Wikipedia")
        return tickers
    except Exception as e:
        print(f"Warning: Failed to fetch tickers from Wikipedia ({e})")
        print(f"Using fallback ticker list ({len(FALLBACK_SP500_TICKERS)} tickers)")
        return FALLBACK_SP500_TICKERS


def download_stock_data(
    tickers: List[str],
    start_date: str = "2020-01-01",
    end_date: str = "2025-12-31",
    batch_size: int = 50
) -> dict:
    """
    Download stock data for multiple tickers in batches.
    
    Args:
        tickers: List of ticker symbols
        start_date: Start date (YYYY-MM-DD format)
        end_date: End date (YYYY-MM-DD format)
        batch_size: Number of tickers to download per batch
        
    Returns:
        dict: Dictionary mapping ticker symbols to their OHLCV DataFrames
    """
    all_data = {}
    
    for batch_start in range(0, len(tickers), batch_size):
        batch = tickers[batch_start: batch_start + batch_size]
        print(f"Downloading batch {batch_start // batch_size + 1} ({len(batch)} tickers)...")
        
        try:
            df = yf.download(batch, start=start_date, end=end_date, group_by="ticker", progress=False)
            
            for ticker in batch:
                if ticker in df:
                    all_data[ticker] = df[ticker]
                else:
                    print(f"Warning: no data for {ticker}")
        except Exception as e:
            print(f"Error downloading batch: {e}")
            continue
    
    return all_data


def get_close_prices(all_data: dict) -> pd.DataFrame:
    """
    Extract close prices from downloaded stock data.
    
    Args:
        all_data: Dictionary mapping tickers to OHLCV DataFrames
        
    Returns:
        pd.DataFrame: DataFrame with close prices for all tickers
    """
    close_prices = pd.DataFrame({
        ticker: data["Close"] for ticker, data in all_data.items()
    })
    return close_prices


def fetch_and_save_sp500_data(
    output_path: str = "sp500_close_prices.csv",
    start_date: str = "2020-01-01",
    end_date: str = "2025-12-31",
    use_fallback: bool = False
) -> pd.DataFrame:
    """
    Fetch S&P 500 data and save to CSV.
    
    Args:
        output_path: Path to save the CSV file
        start_date: Start date (YYYY-MM-DD format)
        end_date: End date (YYYY-MM-DD format)
        use_fallback: If True, use fallback ticker list
        
    Returns:
        pd.DataFrame: Close prices DataFrame
    """
    tickers = get_sp500_tickers(use_fallback=use_fallback)
    all_data = download_stock_data(tickers, start_date, end_date)
    close_prices = get_close_prices(all_data)
    close_prices.to_csv(output_path)
    print(f"Data saved to {output_path}")
    return close_prices


if __name__ == "__main__":
    # Example usage with fallback (if Wikipedia access fails)
    df = fetch_and_save_sp500_data(use_fallback=True)
    print(df.head())