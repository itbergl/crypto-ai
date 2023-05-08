import random
import pandas as pd
import numpy as np
from data import StrSeriesPairs

POPULATION = 501 # Has to be an odd number
MUTATION_PERCENT = 0.01 # The percentage of the population to mutate

# Language to express candidate solutions which are defined by Gene, a subset of the dnf
Literal = tuple[int, float, int]
Conjunctive = tuple[Literal, Literal]
Disjunctive = tuple[Conjunctive, Conjunctive]
Gene = tuple[Disjunctive, Disjunctive]
Population = list[Gene]
Expression = tuple[str, float, str]

def rand_trigger(indicators_and_candle_values: StrSeriesPairs) -> Disjunctive:
    return [
		[
			rand_expression(indicators_and_candle_values),
			rand_expression(indicators_and_candle_values),
		],
		[
			rand_expression(indicators_and_candle_values),
			rand_expression(indicators_and_candle_values),
		],
	]

def rand_expression(indicators_and_candle_values: StrSeriesPairs) -> Literal:
	'''
	Randomly create an expression of `A > c * B` where `A` and `B` are indicator or candle values.
	'''
	lhs = rand_indicator_or_candle_val(indicators_and_candle_values)
	rhs = rand_indicator_or_candle_val(indicators_and_candle_values)
	while lhs == rhs:
		rhs = rand_indicator_or_candle_val(indicators_and_candle_values)
	c = rand_constant(lhs, rhs, indicators_and_candle_values)
	return [lhs, c, rhs]

def rand_constant(lhs: int, rhs: int, indicators_and_candle_values: StrSeriesPairs) -> float:
	'''
	Randomly create a constant in the range of `- A / B` to `A / B`.
	Calculate this range using the median value of `A` and `B`.
	'''
	values1 = indicators_and_candle_values[lhs][1]
	values2 = indicators_and_candle_values[rhs][1]
	# TODO change how to pick the constant value
	val1 = find_median(values1[len(values1) // 2], values1)
	val2 = find_median(values2[len(values2) // 2], values2)
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

def evaluate(df_rows: pd.Series, pool: Population, indicators_and_candle_values: StrSeriesPairs):
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

def get_indicator_and_candle_values_from_gene(gene: Gene) -> list[Literal]:
	'''
	Return the buy and sell triggers in the expression of:
	`buy_trigger = A and B or C and D`,
	`sell_trigger = E and F or G and H`,
	where A, B, C, D, E, F, G, H are expressions in the form of `a > b * c`
	'''
	buy_trigger, sell_trigger = gene
	buy_conj1, buy_conj2 = buy_trigger
	sell_conj1, sell_conj2 = sell_trigger
	A, B = buy_conj1
	C, D = buy_conj2
	E, F = sell_conj1
	G, H = sell_conj2
	return [A, B, C, D, E, F, G, H]

def get_expression(expression: Literal, indicators_and_candle_values: StrSeriesPairs) -> Expression:
	a, b, c = expression
	return indicators_and_candle_values[a][0], b, indicators_and_candle_values[c][0]

def evaluate_expressions(row, expressions: list[Expression]):
	A, B, C, D = expressions
	return evaluate_expression(row, A) and evaluate_expression(row, B) or evaluate_expression(row, C) and evaluate_expression(row, D)

def evaluate_expression(row, expression: Expression):
	a, b, c = expression
	return row[a] > b * row[c]

def fitness(df_rows: pd.Series, gene: Gene, indicators_and_candle_values: StrSeriesPairs) -> float:
	'''
	Calculate the amount of money at the end of all the trades starting with $100.
	'''
	amount = 100.
	buy_trigger = False
	expressions = [get_expression(expression, indicators_and_candle_values) for expression in get_indicator_and_candle_values_from_gene(gene)]
	for row in df_rows:
		if not buy_trigger and all(not (np.isnan(row[indicator1]) or np.isnan(row[indicator2])) for (indicator1, indicator2) in [(expression[0], expression[2]) for expression in expressions[:4]]) and evaluate_expressions(row, expressions[:4]):
			amount -= amount * 0.02
			amount /= row['close']
			buy_trigger = True
		elif buy_trigger and all(not (np.isnan(row[indicator1]) or np.isnan(row[indicator2])) for (indicator1, indicator2) in [(expression[0], expression[2]) for expression in expressions[4:]]) and evaluate_expressions(row, expressions[4:]):
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

def crossover(g1: Gene, g2: Gene, pos: int, next_gen: Population):
	'''
	Given `2` genes `g1` and `g2` which were selected through the selection process.
	Randomly pick a cut out of the `4` expressions.
	Then the left side of the cut of `g1` is glued to the right side of the cut of `g2` and vice versa.
	These two new genes will enter the next generation of the process.
	`buy_trigger = A and B or C and D`,
	`sell_trigger = E and F or G and H`,
 	'''
	expressions1 = get_indicator_and_candle_values_from_gene(g1)
	expressions2 = get_indicator_and_candle_values_from_gene(g2)
	cut = random.choice(range(1, 7))
	A, B, C, D, E, F, G, H = expressions1[:cut] + expressions2[cut:]
	I, J, K, L, M, N, O, P = expressions2[:cut] + expressions1[cut:]

	next_gen[pos] = [[[A, B], [C, D]], [[E, F], [G, H]]]
	next_gen[pos + 1] = [[[I, J], [K, L]], [[M, N], [O, P]]]

def mutation(pool: Population, indicators_and_candle_values: StrSeriesPairs):
	'''
	Randomly change the constant value in one of the 8 expressions in a small percentage of the genes picked randomly from the population.
 	'''
	for _ in range(round(POPULATION * MUTATION_PERCENT)):
		i = random.randrange(0, POPULATION)
		match random.choice(range(8)):
			case 0:
				expression = pool[i][0][0][0]
			case 1:
				expression = pool[i][0][0][1]
			case 2:
				expression = pool[i][0][1][0]
			case 3:
				expression = pool[i][0][1][1]
			case 4:
				expression = pool[i][1][0][0]
			case 5:
				expression = pool[i][1][0][1]
			case 6:
				expression = pool[i][1][1][0]
			case 7:
				expression = pool[i][1][1][1]
		expression[1] = rand_constant(expression[0], expression[2], indicators_and_candle_values)

def format_trigger(expressions: list[Expression]):
	return f'{format_expression(expressions[0])} & {format_expression(expressions[1])} or {format_expression(expressions[2])} & {format_expression(expressions[3])}'

def format_expression(expression: Expression):
	a, b, c = expression
	return f'{a} > {b:.5f} * {c}'
