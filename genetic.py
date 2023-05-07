import random
import numpy as np

class Expression:

	def __init__(self, df):
		'''
		Randomly create an expression of `A > c * B` where `A` and `B` are indicator or candle values.
		'''
		# choose two different random headers from df
		self.lhs, self.rhs = random.sample(list(range(len(df.columns))), 2)
		
		# get c based on joint normal distribution of A and B
		self.c = self.rand_constant(df)

	def rand_constant(self, df, std = 1):
		# A, B = df.iloc[0, self.lhs], df.iloc[0, self.rhs]
		return np.random.normal(0, std, 1)[0]
		# random float from -10,000 to +10,000


		# neg = 0 if B == 0 else np.random.normal(-A/B or 0, std, 1)
		# pos = 0 if B == 0 else np.random.normal(A/B or 0, std, 1)
	
		# return random.choice([neg, pos])

	def evaluate(self, df_row):
		return df_row[self.lhs] > self.c * df_row[self.rhs]

	def to_string(self, df):
		return f'{df.columns[self.lhs]} > {self.c} * {df.columns[self.rhs]}'
	
class Gene:

	def __init__(self, df, n_expressions=2):
		self.buy_trigger = [Expression(df) for _ in range(n_expressions)]
		self.sell_trigger = [Expression(df) for _ in range(n_expressions)]

	def evaluate_buys(self, df_row):
		return all(exp.evaluate(df_row) for exp in self.buy_trigger)
	
	def evaluate_sells(self, df_row):
		return all(exp.evaluate(df_row) for exp in self.sell_trigger)

	def to_string(self, df):
		return f'buy: {" & ".join(exp.to_string(df) for exp in self.buy_trigger)}\nsell: {" & ".join(exp.to_string(df) for exp in self.sell_trigger)}'
	
class Population:

	def __init__(self, df, population_size=501):
		self.df = df
		self.df_rows = [row for _, row in df.iterrows()]
		self.pool = [Gene(df) for _ in range(population_size)]


	def evaluate(self):
		'''
		Call the fitness evaluation for each gene in the pool and return the restuls in fitnesses.
		Add up the total fitness sum of all the genes in the whole generation and find the best one to return.
		'''
		fitnesses = [self.fitness(gene) for gene in self.pool]
		fit_sum = sum(fitnesses)
		max_pos = np.argmax(fitnesses)
		max_fit = fitnesses[max_pos]

		return max_pos, max_fit, fit_sum, fitnesses


	def fitness(self, gene: Gene) -> float:
		'''
		Calculate the amount of money at the end of all the trades starting with $100.
		'''
		amount = 100.
		buy_trigger = False

		for row in self.df_rows:
			if not buy_trigger:
				if gene.evaluate_buys(row):
					amount -= amount * 0.02
					amount /= row['close']
					buy_trigger = True

			elif buy_trigger:
				if gene.evaluate_sells(row):
					amount *= row['close'] * (1 - 0.02)
					buy_trigger = False
		return amount

	def selection(self, fit_sum: float, fitnesses: list[float]):
		'''
		Implement wheel of fitness to select a random gene biased to reflect its fitness.
		Wheel segments are covered with gene symbols, where the segment size matches the relative fitness level (twice the fitness means twice the area).
		Assuming each spin is random, it will select a random gene with the desired bias according to their fitness levels.
		'''
		wheel = random.randrange(0, round(fit_sum))
		i = 0
		count = fitnesses[0]
		while count < wheel:
			i += 1
			count += fitnesses[i]
		return i

	def crossover(self, fitsum: float, fitnesses: list[float]):
		'''
		Given `2` genes `g1` and `g2` which were selected through the selection process.
		Randomly pick a cut out of the `4` expressions.
		Then the left side of the cut of `g1` is glued to the right side of the cut of `g2` and vice versa.
		These two new genes will enter the next generation of the process.
		'''
		for i in range(1, len(self.pool), 2):
			a = self.selection(fitsum, fitnesses)
			b = self.selection(fitsum, fitnesses)

			geneA, geneB = self.pool[a], self.pool[b]
			for triggerA, triggerB in [(geneA.buy_trigger, geneB.buy_trigger), (geneA.sell_trigger, geneB.sell_trigger)]:
				cut = random.randrange(1, len(triggerA))
				triggerA[:cut], triggerB[cut:] = triggerB[cut:], triggerA[:cut]
			

	def mutate(self, N = 5):
		'''
		Randomly change one of the constant values in each of the `N` genes picked randomly from the pool.
		'''
		for gene in random.sample(self.pool, N):
			trigger = gene.buy_trigger if random.choice([True, False]) else gene.sell_trigger
			exp = random.choice(trigger)
			exp.c = exp.rand_constant(self.df)
