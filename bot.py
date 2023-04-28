class Bot:
    def __init__(self, params=None):
        self._aud = 100
        self._btc = 0
        self.fee = 0.02
        self.params = params
        self._previous_signal = 'sell'

    def buy(self, t, price):
        self._btc = self._aud * (1 - self.fee) / price
        print(f'Day {t}: buy {round(self._btc, 5)} BTC, spend {round(self._aud, 3)} AUD')
        self._aud = 0

    def sell(self, t, price):
        self._aud = self._btc * price * (1 - self.fee)
        print(f'Day {t}: sell {round(self._btc, 5)} BTC, get {round(self._aud, 3)} AUD')
        self._btc = 0

    def trigger(self, position):
        if position == 1 and self._previous_signal != 'buy':
            return 'buy'
        elif position == -1 and self._previous_signal != 'sell':
            return 'sell'
        else:
            return None

    def step(self, t, position, price):
        if t == 0:
            self.buy(t, price)
            self._previous_signal = 'buy'
            return
        signal = self.trigger(position)
        if signal is None:
            return
        elif signal == 'buy':
            self.buy(t, price)
        elif signal == 'sell':
            self.sell(t, price)
        self._previous_signal = signal

    def get_aud(self):
        return self._aud

    def get_btc(self):
        return self._btc
