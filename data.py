import ccxt
import pandas as pd
from ta.trend import SMAIndicator


def retrieve_data():
    exchange = ccxt.kraken()
    ohlc = exchange.fetch_ohlcv(symbol='BTC/AUD', timeframe='1d')
    df = pd.DataFrame(ohlc, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    return df

#  add SMAs to the dataframe based on closing prices
def sma_indicator(df, window_low_freq=21, window_high_freq=9):
    sma_low_freq = SMAIndicator(df['close'], window=window_low_freq)
    sma_high_freq = SMAIndicator(df['close'], window=window_high_freq)
    df['sma_' + str(window_low_freq)] = sma_low_freq.sma_indicator()
    df['sma_' + str(window_high_freq)] = sma_high_freq.sma_indicator()
    df['position'] = 0
    df.loc[df['sma_9'] > df['sma_21'], 'position'] = 1
    df.loc[df['sma_9'] < df['sma_21'], 'position'] = -1
    return df
