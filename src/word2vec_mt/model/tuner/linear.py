'''
Classes and functions for tuning the hyperparameters of the linear model and then training it.
The hyperparameters are evaluated by training the model for 1 epoch, then it is trained for as
many epochs as are specified in the configuration file.
'''

import json
import csv
import torch
import numpy as np
import optuna
from optuna.samplers import RandomSampler
from word2vec_mt.model.tuner.common import Listener, DuplicateHyperparametersAttempted
from word2vec_mt.model.trainer import train_linear_model
from word2vec_mt.model.data import load_translation_data_set
from word2vec_mt.model.evaluate import translation_mean_average_precision, get_translation_report
from word2vec_mt.paths import (
    word2vec_mt_path,
    word2vec_en_path, word2vec_mten_path,
    linear_hyperparams_config_path, linear_hyperparams_db_path,
    linear_hyperparams_result_path, linear_hyperparams_best_path, linear_model_path,
)


#########################################
def linear_model_objective(
    trial: optuna.Trial,
) -> float:
    '''
    An Optuna objective function for evaluating a set of hyperparameters.

    :param trial: The hyperparameter sampler.
    :return: The mean average precision of retrieving the most similar English token vectors to
        Maltese token vectors.
    '''
    data_set = load_translation_data_set()
    word2vec_mt = np.load(word2vec_mt_path)
    word2vec_en = np.load(word2vec_en_path)
    with open(linear_hyperparams_config_path, 'r', encoding='utf-8') as f:
        hyperparams = json.load(f)

    init_stddev = trial.suggest_categorical('init_stddev', hyperparams['init_stddev'])
    use_bias = trial.suggest_categorical('use_bias', hyperparams['use_bias'])
    weight_decay = trial.suggest_categorical('weight_decay', hyperparams['weight_decay'])
    learning_rate = trial.suggest_categorical('learning_rate', hyperparams['learning_rate'])
    batch_size = trial.suggest_categorical('batch_size', hyperparams['batch_size'])
    seed = trial.suggest_categorical('seed', hyperparams['seed'])
    if any(
        t.params == trial.params
        for t in trial.study.trials
        if t.state == optuna.trial.TrialState.COMPLETE
    ):
        raise DuplicateHyperparametersAttempted()

    print(
        f'Now training model with init_stddev: {init_stddev}, use_bias: {use_bias},'
        f' weight_decay: {weight_decay}, learning_rate: {learning_rate}, batch_size: {batch_size}'
    )
    model = train_linear_model(
        source_embedding_size=hyperparams['source_embedding_size'],
        target_embedding_size=hyperparams['target_embedding_size'],
        init_stddev=init_stddev,
        use_bias=use_bias,
        weight_decay=weight_decay,
        learning_rate=learning_rate,
        max_epochs=1,
        source_embedding_matrix=word2vec_mt,
        target_embedding_matrix=word2vec_en,
        train_data=data_set.train.flatten(),
        val_data=data_set.val,
        batch_size=batch_size,
        patience=hyperparams['patience'],
        device=hyperparams['device'],
        seed=seed,
        listener=Listener(),
    )

    with torch.no_grad():
        word2vec_mten = model(
            torch.from_numpy(word2vec_mt).to(hyperparams['device'])
        ).cpu().numpy()
    dev_map = translation_mean_average_precision(word2vec_mten, word2vec_en, data_set.dev)
    return dev_map


#########################################
def tune_linear_model(
) -> None:
    '''
    Tune the linear hyperparameters with Optuna.
    '''
    with open(linear_hyperparams_config_path, 'r', encoding='utf-8') as f:
        hyperparams = json.load(f)

    study = optuna.create_study(
        direction='maximize',
        study_name='word2vec_mt',
        sampler=RandomSampler(seed=0),
        storage='sqlite:///' + linear_hyperparams_db_path,
        load_if_exists=True,
    )
    try:
        with open(linear_hyperparams_result_path, 'x', encoding='utf-8', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'init_stddev',
                'use_bias',
                'weight_decay',
                'learning_rate',
                'batch_size',
                'seed',
                'dev_map',
            ])
    except FileExistsError:
        pass

    tuning_trials = hyperparams['tuning_trials']
    num_complete_trials = len(study.get_trials(states=[optuna.trial.TrialState.COMPLETE]))
    trials_left = max(0, tuning_trials - num_complete_trials)
    print(f'Tuning for {trials_left} iterations')
    for i in range(num_complete_trials + 1, hyperparams['tuning_trials'] + 1):
        print()
        print('===========================================')
        print(f'Iteration {i}/{tuning_trials}')
        while True:
            if hyperparams['device'].startswith('cuda'):
                torch.cuda.empty_cache()
            trial = study.ask()
            try:
                value = linear_model_objective(trial)
                print(f'dev map: {value}')
            except DuplicateHyperparametersAttempted:
                print('(duplicate hyperparameters attempted, trying again)')
                continue
            study.tell(trial, value)
            break

        with open(linear_hyperparams_result_path, 'a', encoding='utf-8', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                trial.params['init_stddev'],
                trial.params['use_bias'],
                trial.params['weight_decay'],
                trial.params['learning_rate'],
                trial.params['batch_size'],
                trial.params['seed'],
                value,
            ])


#########################################
def train_best_linear_model(
) -> None:
    '''
    Train the linear model using the best hyperparameters found.
    '''
    data_set = load_translation_data_set()
    word2vec_mt = np.load(word2vec_mt_path)
    word2vec_en = np.load(word2vec_en_path)
    with open(linear_hyperparams_config_path, 'r', encoding='utf-8') as f:
        hyperparams = json.load(f)

    study = optuna.create_study(
        direction='maximize',
        study_name='word2vec_mt',
        storage='sqlite:///' + linear_hyperparams_db_path,
        load_if_exists=True,
    )

    print('training model')
    model = train_linear_model(
        source_embedding_size=hyperparams['source_embedding_size'],
        target_embedding_size=hyperparams['target_embedding_size'],
        init_stddev=study.best_params['init_stddev'],
        use_bias=study.best_params['use_bias'],
        weight_decay=study.best_params['weight_decay'],
        learning_rate=study.best_params['learning_rate'],
        max_epochs=hyperparams['max_epochs'],
        source_embedding_matrix=word2vec_mt,
        target_embedding_matrix=word2vec_en,
        train_data=data_set.train.flatten(),
        val_data=data_set.val,
        batch_size=study.best_params['batch_size'],
        patience=hyperparams['patience'],
        device=hyperparams['device'],
        seed=study.best_params['seed'],
        listener=Listener(),
    )

    print('saving model')
    torch.save(model, linear_model_path)

    print('saving word2vec embeddings')
    with torch.no_grad():
        word2vec_mten = model(
            torch.from_numpy(word2vec_mt).to(hyperparams['device'])
        ).cpu().numpy()
    np.save(word2vec_mten_path, word2vec_mten, allow_pickle=False)

    print('evaluating model')
    test_map = translation_mean_average_precision(word2vec_mt, word2vec_en, data_set.test)
    hyperparams.update(study.best_params)
    hyperparams['test_set_map'] = test_map

    print('saving model hyperparameters')
    with open(linear_hyperparams_best_path, 'w', encoding='utf-8') as f:
        json.dump(hyperparams, f, ensure_ascii=False, indent=4)

    print('writing report on word2vec embeddings')
    get_translation_report(word2vec_mt, word2vec_en, data_set.test)
