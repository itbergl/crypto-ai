import pandas as pd
from tqdm import tqdm
from data import retrieve_data, list_indicators_and_candle_values, add_all_indicators, save_data
from genetic import Population

MAX_ITER = 30

if __name__=='__main__':
	# Allow printing the entire data frame
	pd.set_option('display.max_columns', None)
	pd.set_option('display.max_rows', None)

	# Set up dataframe
	df = retrieve_data()
	indicators_and_candle_values = list_indicators_and_candle_values(df)
	df = add_all_indicators(df, indicators_and_candle_values)
	df.to_csv('data.csv')

	# Initialise gene pools
	pool = Population(df)

	# record bot values for visualisation
	bot_record = []

	# Run genetic algorithm for some number of iterations
	for _ in tqdm(range(MAX_ITER), total=MAX_ITER):
		max_pos, max_fit, fit_sum, fitnesses = pool.evaluate()

		# append the value record of the best bot
		item = {'max_pos': max_pos, 'max_fit': max_fit, 'fit_sum': fit_sum, 'fitnesses': fitnesses}
		bot_record.append(item)

		# Preserve best gene and replicate it so it appears twice in the next generation
		pool.pool[0] = pool.pool[max_pos]
		# Do crossover for the rest of genes and mutate a small amount of them randomly

		pool.crossover(fit_sum, fitnesses)
		pool.mutate()

	# Print out the best gene after all the evolution
	max_pos, max_fit, fit_sum, fitnesses = pool.evaluate()
	print(f'best bot earns ${max_fit:.5f}')

	best_gene = pool.pool[max_pos]

	print(best_gene.to_string(df))
	save_data('bot_record.csv', bot_record)
