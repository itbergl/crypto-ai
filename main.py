import ccxt
import pandas as pd
from ta import volatility

exchange = ccxt.kraken()
ohlc = exchange.fetch_ohlcv('BTC/AUD')
df = pd.DataFrame(ohlc, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
bb_indicator = volatility.BollingerBands(df['close'])
df['moving_average'] = bb_indicator.bollinger_mavg()
print(df)
