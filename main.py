import ccxt
exchange = ccxt.kraken()
ohlc = exchange.fetch_ohlcv('BTC/AUD')
for candle in ohlc:
    print(candle)
