import pandas as pd
from tqdm import tqdm
from data import retrieve_data, list_indicators_and_candle_values, add_all_indicators
from genetic import POPULATION, Population, rand_trigger, evaluate, selection, crossover, mutation, get_indicator_and_candle_values_from_gene

MAX_ITER = 30

if __name__=='__main__':
	# Allow printing the entire data frame
	pd.set_option('display.max_columns', None)
	pd.set_option('display.max_rows', None)

	# Set up data frame
	df = retrieve_data()
	indicators_and_candle_values = list_indicators_and_candle_values(df)
	df = add_all_indicators(df, indicators_and_candle_values)
	df_rows = [row for _, row in df.iterrows()]

	# Initialise gene pools
	pool: Population = [
		[
			*rand_trigger(indicators_and_candle_values),
			*rand_trigger(indicators_and_candle_values),
			*rand_trigger(indicators_and_candle_values),
			*rand_trigger(indicators_and_candle_values),
		]
		for _ in range(POPULATION)
	]
	next_gen: Population = [[] for _ in range(POPULATION)]

	# Run genetic algorithm for some number of iterations
	for _ in tqdm(range(MAX_ITER), total=MAX_ITER):
		max_pos, max_fit, fit_sum, fitnesses = evaluate(df_rows, pool, indicators_and_candle_values)
		# Preserve best gene and replicate it so it appears twice in the next generation
		next_gen[0] = pool[max_pos]
		# Do crossover for the rest of genes and mutate a small amount of them randomly
		for i in range(1, POPULATION, 2):
			crossover(pool[selection(fit_sum, fitnesses)], pool[selection(fit_sum, fitnesses)], i, next_gen)
		mutation(next_gen, indicators_and_candle_values)
		pool = next_gen

	# Print out the best gene after all the evolution
	max_pos, max_fit, fit_sum, fitnesses = evaluate(df_rows, pool, indicators_and_candle_values)
	print(f'best bot earns ${max_fit:.5f}')
	a, b, c, d, e, f, g, h, i, j, k, l = get_indicator_and_candle_values_from_gene(pool[max_pos], indicators_and_candle_values)
	print(f'buy trigger: {a} > {b:.5f} * {c} & {d} > {e:.5f} * {f}')
	print(f'sell trigger: {g} > {h:.5f} * {i} & {j} > {k:.5f} * {l}')
