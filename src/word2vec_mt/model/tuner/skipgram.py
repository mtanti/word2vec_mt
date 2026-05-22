'''
Classes and functions for tuning the hyperparameters of the skipgram model and then training it.
The hyperparameters are evaluated by training the model for 1 epoch, then it is trained for as
many epochs as are specified in the configuration file.
'''

import json
import csv
import timeit
import torch
import h5py
import numpy as np
import optuna
from optuna.samplers import RandomSampler
from word2vec_mt.model.tuner.common import Listener, DuplicateHyperparametersAttempted
from word2vec_mt.model.trainer import train_skipgram_model, NegativeTokenSampler
from word2vec_mt.model.data import load_synonym_data_set
from word2vec_mt.model.evaluate import synonym_mean_average_precision, get_synonym_report
from word2vec_mt.paths import (
    vocab_mt_path, freqs_mt_path,
    proccorpus_mt_path, word2vec_mt_path,
    skipgram_hyperparams_config_path, skipgram_hyperparams_db_path,
    skipgram_hyperparams_result_path, skipgram_hyperparams_best_path, skipgram_model_path,
)


#########################################
def skipgram_model_objective(
    trial: optuna.Trial,
) -> float:
    '''
    An Optuna objective function for evaluating a set of hyperparameters.

    :param trial: The hyperparameter sampler.
    :return: The mean average precision of retrieving the most similar Maltese token vectors to
        other Maltese token vectors.
    '''
    data_set = load_synonym_data_set()
    with open(vocab_mt_path, 'r', encoding='utf-8') as f:
        vocab_mt = f.read().strip().split('\n')
    proccorp = h5py.File(proccorpus_mt_path, 'r')
    with open(freqs_mt_path, 'r', encoding='utf-8') as f:
        neg_token_sampler = NegativeTokenSampler(json.load(f))
    with open(skipgram_hyperparams_config_path, 'r', encoding='utf-8') as f:
        hyperparams = json.load(f)

    init_stddev = trial.suggest_categorical('init_stddev', hyperparams['init_stddev'])
    learning_rate = trial.suggest_categorical('learning_rate', hyperparams['learning_rate'])
    radius = trial.suggest_categorical('radius', hyperparams['radius'])
    neg_sample_ratio = trial.suggest_categorical(
        'neg_sample_ratio',
        hyperparams['neg_sample_ratio'],
    )
    seed = trial.suggest_categorical('seed', hyperparams['seed'])
    if any(
        t.params == trial.params
        for t in trial.study.trials
        if t.state == optuna.trial.TrialState.COMPLETE
    ):
        raise DuplicateHyperparametersAttempted()

    print(
        'Now training model with'
        f' init_stddev: {init_stddev},'
        f' learning_rate: {learning_rate},'
        f' radius: {radius},'
        f' neg_sample_ratio: {neg_sample_ratio},'
        f' seed: {seed}'
    )
    model = train_skipgram_model(
        vocab_size=len(vocab_mt),
        embedding_size=hyperparams['embedding_size'],
        init_stddev=init_stddev,
        learning_rate=learning_rate,
        max_epochs=1,
        train_data=proccorp['radius_'+str(radius)],
        negative_token_sampler=neg_token_sampler,
        negative_sample_ratio=neg_sample_ratio,
        val_data=data_set.val,
        superbatch_size=hyperparams['superbatch_size'],
        batch_size=hyperparams['batch_size'],
        patience=hyperparams['patience'],
        device=hyperparams['device'],
        seed=seed,
        listener=Listener(),
    )

    dev_map = synonym_mean_average_precision(model.get_embeddings(), data_set.dev)
    return dev_map


#########################################
def tune_skipgram_model(
) -> None:
    '''
    Tune the skipgram hyperparameters with Optuna.
    '''
    with open(skipgram_hyperparams_config_path, 'r', encoding='utf-8') as f:
        hyperparams = json.load(f)

    study = optuna.create_study(
        direction='maximize',
        study_name='word2vec_mt',
        sampler=RandomSampler(seed=0),
        storage='sqlite:///' + skipgram_hyperparams_db_path,
        load_if_exists=True,
    )
    try:
        with open(skipgram_hyperparams_result_path, 'x', encoding='utf-8', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'init_stddev',
                'learning_rate',
                'radius',
                'neg_sample_ratio',
                'seed',
                'dev_map',
                'duration',
            ])
    except FileExistsError:
        pass

    tuning_trials = hyperparams['tuning_trials']
    num_complete_trials = len(study.get_trials(states=[optuna.trial.TrialState.COMPLETE]))
    trials_left = max(0, tuning_trials - num_complete_trials)
    print(f'Tuning for {trials_left} iterations')
    for i in range(num_complete_trials + 1, tuning_trials + 1):
        print()
        print('===========================================')
        print(f'Iteration {i}/{tuning_trials}')
        while True:
            if hyperparams['device'].startswith('cuda'):
                torch.cuda.empty_cache()
            trial = study.ask()
            start_time = timeit.default_timer()
            try:
                value = skipgram_model_objective(trial)
                print(f'dev map: {value}')
            except DuplicateHyperparametersAttempted:
                print('(duplicate hyperparameters attempted, trying again)')
                continue
            duration = timeit.default_timer() - start_time
            study.tell(trial, value)
            break

        with open(skipgram_hyperparams_result_path, 'a', encoding='utf-8', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                trial.params['init_stddev'],
                trial.params['learning_rate'],
                trial.params['radius'],
                trial.params['neg_sample_ratio'],
                trial.params['seed'],
                value,
                duration,
            ])


#########################################
def train_best_skipgram_model(
) -> None:
    '''
    Train the skipgram model using the best hyperparameters found.
    '''
    data_set = load_synonym_data_set()
    with open(vocab_mt_path, 'r', encoding='utf-8') as f:
        vocab_mt = f.read().strip().split('\n')
    proccorp = h5py.File(proccorpus_mt_path, 'r')
    with open(freqs_mt_path, 'r', encoding='utf-8') as f:
        neg_token_sampler = NegativeTokenSampler(json.load(f))
    with open(skipgram_hyperparams_config_path, 'r', encoding='utf-8') as f:
        hyperparams = json.load(f)

    study = optuna.create_study(
        direction='maximize',
        study_name='word2vec_mt',
        storage='sqlite:///' + skipgram_hyperparams_db_path,
        load_if_exists=True,
    )

    print('training model')
    model = train_skipgram_model(
        vocab_size=len(vocab_mt),
        embedding_size=hyperparams['embedding_size'],
        init_stddev=study.best_params['init_stddev'],
        learning_rate=study.best_params['learning_rate'],
        max_epochs=hyperparams['max_epochs'],
        train_data=proccorp['radius_'+str(study.best_params['radius'])],
        negative_token_sampler=neg_token_sampler,
        negative_sample_ratio=study.best_params['neg_sample_ratio'],
        val_data=data_set.val,
        superbatch_size=hyperparams['superbatch_size'],
        batch_size=hyperparams['batch_size'],
        patience=hyperparams['patience'],
        device=hyperparams['device'],
        seed=study.best_params['seed'],
        listener=Listener(),
    )

    print('saving model')
    torch.save(model, skipgram_model_path)

    print('saving word2vec embeddings')
    np.save(word2vec_mt_path, model.get_embeddings(), allow_pickle=False)

    print('evaluating model')
    test_map = synonym_mean_average_precision(model.get_embeddings(), data_set.test)
    hyperparams.update(study.best_params)
    hyperparams['test_set_map'] = test_map

    print('saving model hyperparameters')
    with open(skipgram_hyperparams_best_path, 'w', encoding='utf-8') as f:
        json.dump(hyperparams, f, ensure_ascii=False, indent=4)

    print('writing report on word2vec embeddings')
    get_synonym_report(model.get_embeddings(), data_set.test)
