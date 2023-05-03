import pandas as pd
from data import retrieve_data, list_indicators_and_candle_values, add_all_indicators
from genetic import Population, rand_trigger, evaluate, selection, crossover, mutation, get_indicator_and_candle_values_from_gene
from tqdm import tqdm
import optuna
from random import shuffle


def objective(trial: optuna.trial, df_rows, indicators_and_candle_values):

	# TRAINING_PARAMS = {
	# 	# 'max_iter': 20,
	# 	# 'population': 500,
	# 	# 'n_mutations': 5,
	# 	# 'fittest_copies': 2,
	# }

	# MAX_ITER =  trial.suggest_int('MAX_ITER', 1,50)
	MAX_ITER = 50
	# POPULATION =  trial.suggest_int('POPULATION', 100,1000)
	POPULATION = 500
	N_MUTATIONS =  trial.suggest_float('N_MUTATIONS', 0, 500*4)
	FITTEST_COPIES =  trial.suggest_float('FITTEST_COPIES', 0, 500)
	SHUFFLE_POPULATION =  trial.suggest_int('SHUFFLE_POPULATION', 0, 1)

	pool: Population = [
		[
			*rand_trigger(indicators_and_candle_values),
			*rand_trigger(indicators_and_candle_values),
			*rand_trigger(indicators_and_candle_values),
			*rand_trigger(indicators_and_candle_values),
		]
		for _ in range(POPULATION)
	]
	next: Population = [[] for _ in range(POPULATION)]

	for iter in range(MAX_ITER):

		max_pos, max_fit, fit_sum, fitnesses = evaluate(df_rows, pool, indicators_and_candle_values)
		# Preserve best gene and replicate it so it appears twice in the next generation
		for i in range(FITTEST_COPIES):
			next[i] = [_ for _ in pool[max_pos]]

		# maybe we could shuffle the population?
		if SHUFFLE_POPULATION == 1:
			shuffle(pool)
		
		for i in range(2, POPULATION - 1, 2):
			crossover(pool[selection(fit_sum, fitnesses)], pool[selection(fit_sum, fitnesses)], i, next)

		mutation(next, indicators_and_candle_values, N_MUTATIONS)
		pool = next

		trial.report(max_fit, iter)    
		if trial.should_prune():
			raise optuna.exceptions.TrialPruned()

	max_pos, max_fit, fit_sum, fitnesses = evaluate(df_rows, pool, indicators_and_candle_values)
	return max_fit
	# print(max_fit)
	# a, b, c, d, e, f, g, h, i, j, k, l = get_indicator_and_candle_values_from_gene(pool[max_pos], indicators_and_candle_values)
	# print(f'buy trigger: ({a} > {b} * {c}) & ({d} > {e} * {f})')
	# print(f'sell trigger: ({g} > {h} * {i}) & ({j} > {k} * {l})')

if __name__=='__main__':

	pd.set_option('display.max_columns', None)
	pd.set_option('display.max_rows', None)

	df = retrieve_data()
	indicators_and_candle_values = list_indicators_and_candle_values(df)
	df = add_all_indicators(df, indicators_and_candle_values)
	
	# precompute the df rows for iteration
	df_rows = [row for _, row in df.iterrows()]
	searchspace = {
		'N_MUTATIONS':  [_ for _ in range(500*4+1)],
		'FITTEST_COPIES':  [_ for _ in range(500+1)],
		'SHUFFLE_POPULATION':  [0, 1],
	}
	study = optuna.create_study(direction="maximize", 
								sampler = optuna.samplers.GridSampler(search_space=searchspace), 
								pruner = optuna.pruners.MedianPruner(),
								storage = f'sqlite:///optuna.db',
								# study_name = 'mango2',
								)
	study.optimize(lambda trial: objective(trial, df_rows, indicators_and_candle_values))