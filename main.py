import pandas as pd
from tqdm import tqdm
from data import retrieve_data, list_indicators_and_candle_values, add_all_indicators, save_data
from genetic import POPULATION, Population, format_trigger, get_expression, rand_trigger, evaluate, selection, crossover, mutation, get_indicator_and_candle_values_from_gene

MAX_ITER = 30

if __name__=='__main__':
	# Allow printing the entire data frame
	pd.set_option('display.max_columns', None)
	pd.set_option('display.max_rows', None)

	# Set up data frame
	df = retrieve_data()
	indicators_and_candle_values = list_indicators_and_candle_values(df)
	df = add_all_indicators(df, indicators_and_candle_values)
	df.to_csv('data.csv', index=False)
	df_rows = [row for _, row in df.iterrows()]

	# Initialise gene pools
	pool: Population = [
		[
			rand_trigger(indicators_and_candle_values), # buy trigger
			rand_trigger(indicators_and_candle_values), # sell trigger
		]
		for _ in range(POPULATION)
	]
	next_gen: Population = [[] for _ in range(POPULATION)]

	# Record bot values for visualisation
	bot_record = []

	# Run genetic algorithm for some number of iterations
	for _ in tqdm(range(MAX_ITER), total=MAX_ITER):
		max_pos, max_fit, fit_sum, fitnesses = evaluate(df_rows, pool, indicators_and_candle_values)

		# append the value record of the best bot
		item = {'max_pos': max_pos, 'max_fit': max_fit, 'fit_sum': fit_sum, 'fitnesses': fitnesses}
		bot_record.append(item)

		# Preserve the best gene for in the next generation
		next_gen[0] = pool[max_pos]
		# Do crossover for the rest of genes
		for i in range(1, POPULATION, 2):
			crossover(pool[selection(fit_sum, fitnesses)], pool[selection(fit_sum, fitnesses)], i, next_gen)
		# Mutate a small percentage of the population randomly
		mutation(next_gen, indicators_and_candle_values)
		pool = next_gen

	# Print out the best gene after all the evolution
	max_pos, max_fit, fit_sum, fitnesses = evaluate(df_rows, pool, indicators_and_candle_values)
	print(f'best bot earns ${max_fit:.5f}')
	expressions = [get_expression(expression, indicators_and_candle_values) for expression in get_indicator_and_candle_values_from_gene(pool[max_pos])]
	print(f'buy trigger: {format_trigger(expressions[:4])}')
	print(f'sell trigger: {format_trigger(expressions[4:])}')

	save_data('bot_record.csv', bot_record)
