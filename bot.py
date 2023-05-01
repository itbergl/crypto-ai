class Bot:

    def __init__(self, params=None):
        self._starting_aud = 100
        self._aud = self._starting_aud

        self._btc = 0
        self.fee = 0.02
        self.params = params
        self._previous_signal = 'sell'

    def buy(self, t, price):
        if self._aud == 0:
            return 
        
        self._btc = self._aud * (1 - self.fee) / price
        print(f'Day {t:03}\tbuy\t{self._btc:.5f} BTC\t{self._aud:.3f} AUD spent')
        self._aud = 0

    def sell(self, t, price):
        if self._btc == 0:
            return 
        
        self._aud = self._btc * price * (1 - self.fee)
        print(f'Day {t:03}\tsell\t{self._btc:.5f} BTC\t{self._aud:.3f} AUD recieved')
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

    @property
    def aud(self):
        return self._aud

    @property
    def btc(self):
        return self._btc
    
    # @property
    # def roi(self):
    #     return self._aud / self._starting_aud 
