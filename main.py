import pandas as pd
from tqdm import tqdm
from data import retrieve_data, list_indicators_and_candle_values, add_all_indicators, save_data
from genetic import Population, format_trigger, get_expression, rand_trigger, evaluate, selection, crossover, mutation, get_indicator_and_candle_values_from_gene
import random
import optuna

MAX_ITER = 30
POPULATION = 501

USE_OPTUNA = False

# Used when not using OPTUNA
DEFAULT_PARAMS = {
	'INIT_STD': 1,		# The standard deviation of the normal distribution used to initialise
	'MUTATION_STD': 1,	# The standard deviation of the normal distribution used to mutate
	'N_MUTATIONS': 5, # The number of genes to mutate
	'N_CROSSOVER': 500, # The number of crossovers 
}

# Defines the searchspace for OPTUNA
OPTUNA_SEARCHSPACE = {
	'INIT_STD': [_ for _ in range(1, 5)],
	'MUTATION_STD': [_ for _ in range(1, 5)],
	'N_MUTATIONS': [_ for _ in range(1, 20, 2)],
	'N_CROSSOVER': [_ for _ in range(100, POPULATION, 100)],
}

# random.seed(100)
# np.random.seed(100)

def run(df_rows: list, indicators_and_candle_values, trial: optuna.trial=None):
	
	# select parameters
	params = DEFAULT_PARAMS if trial is None else {k: trial.suggest_int(k, OPTUNA_SEARCHSPACE[k][0], OPTUNA_SEARCHSPACE[k][-1]) for k in OPTUNA_SEARCHSPACE}

	# Initialise gene pools
	pool: Population = [
		[
			rand_trigger(indicators_and_candle_values, params['INIT_STD']), # buy trigger
			rand_trigger(indicators_and_candle_values, params['INIT_STD']), # sell trigger
		]
		for _ in range(POPULATION)
	]
	next_gen: Population = [[] for _ in range(POPULATION)]

	# Record bot values for visualisation
	bot_record = []

	# Run genetic algorithm for some number of iterations
	epochs = range(MAX_ITER) if trial is not None else tqdm(range(MAX_ITER), total=MAX_ITER)
	for epoch in epochs:
		# Shuffle the pool to avoid bias
		random.shuffle(pool)
		max_pos, max_fit, fit_sum, fitnesses = evaluate(df_rows, pool, indicators_and_candle_values)

		# report to optuna
		if trial is not None:
			trial.report(max_fit, epoch)
			if trial.should_prune():
				raise optuna.exceptions.TrialPruned()

		# append the value record of the best bot
		bot_record.append({'max_pos': max_pos, 'max_fit': max_fit, 'fit_sum': fit_sum, 'fitnesses': fitnesses})

		# Preserve the best gene for the next generation
		next_gen[0] = pool[max_pos]
		
		# Do crossover for the rest of genes
		for i in range(1, len(pool), 2):
			g1, g2 = [selection(fit_sum, fitnesses) for _ in range(2)]
			cross_g1, cross_g2 = crossover(g1, g2, pool)
			next_gen[i], next_gen[i+1] = cross_g1, cross_g2
		
		# Only use the crossovers for N_CROSSOVER genes
		next_gen[1:-params['N_CROSSOVER']] = pool[1:-params['N_CROSSOVER']]

		# Mutate a small number of the population randomly
		mutation(next_gen, n_mutations = params['N_MUTATIONS'], mutation_std = params['MUTATION_STD'])
		pool = next_gen

	# Print out the best gene after all the evolution
	max_pos, max_fit, fit_sum, fitnesses = evaluate(df_rows, pool, indicators_and_candle_values)

	if trial is None:
		return (max_pos, max_fit, fit_sum, fitnesses), bot_record, pool
	else:
		return max_fit

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

	# Run the genetic algorithm with default values 
	if not USE_OPTUNA:
		(max_pos, max_fit, fit_sum, fitnesses), bot_record, pool = run(df_rows, indicators_and_candle_values)
		print(f'best bot earns ${max_fit:.5f}')
		expressions = [get_expression(expression, indicators_and_candle_values) for expression in get_indicator_and_candle_values_from_gene(pool[max_pos])]
		print(f'buy trigger: {format_trigger(expressions[:4])}')
		print(f'sell trigger: {format_trigger(expressions[4:])}')

		save_data('bot_record.csv', bot_record)

	# do a hyperparameter search with OPTUNA
	else:
		study = optuna.create_study(direction="maximize", 
							sampler = optuna.samplers.GridSampler(search_space=OPTUNA_SEARCHSPACE), 
							pruner = optuna.pruners.MedianPruner(),
							storage = f'sqlite:///optuna.db',
							# study_name = 'mango2',
							)
		study.optimize(lambda trial: run(df_rows, indicators_and_candle_values, trial=trial))



