import random
import pandas as pd
import numpy as np
from data import StrSeriesPairs

POPULATION = 500
Gene = list[int, float, int, int, float, int, int, float, int, int, float, int]
Population = list[Gene]


'''
Proposal for more general approach:

	Solution is an OR chain [GENE1, GENE2, GENE3, GENE4, ...], where GENEX is 
	an AND chain of [LIT, LIT, LIT, LIT, ...] and LIT is a function IND > c*IND.

	We could have a max size for Solutions and max size for GENEs. 

	class Solution:
		--> property) genes: list[GENE]
	
	class GENE:
		--> property) literals: list[Literal]
	
	class Literal:
		--> property) A -> string
		--> property) B -> string
		--> properrty) C -> float

		--> function) call(df_row) -> (df_row[A] > c*df_row[B])
	
	^ Just ideas they don't have to be classes.

	We could implement a crossover / mutation in each.

	Since there is a fixed size for each solution (max_gene_len * max_literal_len)
	each could even be represented by an array.

	 (A > b*C)	(D > e*f)	(G > h*I) ...
	 (J > k*L)	(M > n*O)	(P > q*R) ...
	 (S > t*U)	(V > w*X)	(Y > z*(AA))
	 .
	 .
	 .

	 Which could be expressed as a string 

	Solution =  [A > b*C OR D > e*f OR G > h*I OR ...] AND [J > k*L OR M > n*O OR P > q*R OR ...] AND [S > t*U OR V > w*X OR Y > z*(AA) OR ...] AND ...

	Evalution could be vectorised on this array (using numpy.vectorize) to be faster than O(n^2)


'''
def rand_trigger(indicators_and_candle_values: StrSeriesPairs):
	'''
	Randomly create an expression of `A > c * B` where `A` and `B` are indicator or candle values.
	'''
	lhs = rand_indicator_or_candle_val(indicators_and_candle_values)
	rhs = rand_indicator_or_candle_val(indicators_and_candle_values)
	while lhs == rhs:
		rhs = rand_indicator_or_candle_val(indicators_and_candle_values)
	c = rand_constant(lhs, rhs, indicators_and_candle_values)
	return lhs, c, rhs

def rand_constant(lhs: int, rhs: int, indicators_and_candle_values: StrSeriesPairs) -> float:
	'''
	Randomly create a constant in the range of `- A / B` to `A / B`.
	Calculate this range using the median value of `A` and `B`.
	'''
	values1 = indicators_and_candle_values[lhs][1]
	values2 = indicators_and_candle_values[rhs][1]
	val1 = find_median(values1[len(values1) / 2], values1)
	val2 = find_median(values2[len(values2) / 2], values2)
	return 0 if val2 == 0 else (random.random() - 0.5) * (val1 // val2)

def find_median(value, values):
	'''
	If the value is `0` or `NAN` find another value that is not, otherwise return `0`.
	'''
	if value != 0 and not np.isnan(value):
		return value
	for val in values:
		if val != 0 and not np.isnan(val):
			return val
	return 0

def rand_indicator_or_candle_val(indicators_and_candle_values: StrSeriesPairs):
	'''
	Randomly pick an indicator or candle value and return its index.
	'''
	return random.randrange(0, len(indicators_and_candle_values))

def evaluate(df_rows: list, pool: Population, indicators_and_candle_values: StrSeriesPairs):
	'''
	Call the fitness evaluation for each gene in the pool and return the restuls in fitnesses.
	Add up the total fitness sum of all the genes in the whole generation and find the best one to return.
	'''
	max_pos = 0
	max_fit = 0.
	fit_sum = 0.
	fitnesses = [0. for _ in range(POPULATION)]
	for i in range(POPULATION):
		fitnesses[i] = fitness(df_rows, pool[i], indicators_and_candle_values)
		fit_sum += fitnesses[i]
		if fitnesses[i] > max_fit:
			max_fit = fitnesses[i]
			max_pos = i
	return max_pos, max_fit, fit_sum, fitnesses

def get_indicator_and_candle_values_from_gene(gene: Gene, indicators_and_candle_values: StrSeriesPairs):
	'''
	Return the buy and sell triggers in the expression of:
	`buy_trigger = a > b * c and d > e * f`,
	`sell_trigger = g > h * i and j > k * l`
	'''
	a, b, c, d, e, f, g, h, i, j, k, l = gene
	return (
		indicators_and_candle_values[a][0],
		b,
		indicators_and_candle_values[c][0],
		indicators_and_candle_values[d][0],
		e,
		indicators_and_candle_values[f][0],
		indicators_and_candle_values[g][0],
		h,
		indicators_and_candle_values[i][0],
		indicators_and_candle_values[j][0],
		k,
		indicators_and_candle_values[l][0],
	)
	
def fitness(df_rows: list, gene: Gene, indicators_and_candle_values: StrSeriesPairs) -> float:
	'''
	Calculate the amount of money at the end of all the trades starting with $100.
	'''
	amount = 100.
	buy_trigger = False
	a, b, c, d, e, f, g, h, i, j, k, l = get_indicator_and_candle_values_from_gene(gene, indicators_and_candle_values)
	for row in df_rows:
		if not buy_trigger and not np.isnan(row[c]) and not np.isnan(row[f]) and row[a] > b * row[c] and row[d] > e * row[f]:
			amount -= amount * 0.02
			amount /= row['close']
			buy_trigger = True
		elif buy_trigger and not np.isnan(row[i]) and not np.isnan(row[l]) and row[g] > h * row[i] and row[j] > k * row[l]:
			amount *= row['close'] * (1 - 0.02)
			buy_trigger = False
	return amount

def selection(fit_sum: float, fitnesses: list[float]):
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

def crossover(g1: Gene, g2: Gene, pos: int, next: Population):
	'''
	Given `2` genes `g1` and `g2` which were selected through the selection process.
	Randomly pick a cut out of the `4` expressions.
	Then the left side of the cut of `g1` is glued to the right side of the cut of `g2` and vice versa.
	These two new genes will enter the next generation of the process.
 	'''
	cut = random.choice((3, 6, 9))
	next[pos] = [*[g1[i] for i in range(cut)], *[g2[i] for i in range(cut, len(g1))]]
	next[pos + 1] = [*[g2[i] for i in range(cut)], *[g1[i] for i in range(cut, len(g1))]]

def mutation(pool: Population, indicators_and_candle_values: StrSeriesPairs):
	'''
	Randomly change one of the constant values in each of the `5` genes picked randomly from the pool.
 	'''
	for _ in range(5):
		i = random.randrange(0, POPULATION)
		j = random.choice((1, 4, 7, 10))
		pool[i][j] = rand_constant(pool[i][j - 1], pool[i][j + 1], indicators_and_candle_values)
