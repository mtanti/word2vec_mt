'''
x
'''
import os
import json
import random
import argparse
from typing import Callable
import tqdm
from hyperparameter_config import Hyperparameters, HyperparameterSearchConfig


#########################################
def list_hyperparams(
    seed: int,
    listener: Callable[[], None] = lambda: None
) -> None:
    '''
    x
    '''
    rng = random.Random(seed)
    seen = set()
    with open(
        os.path.join('hyperparameter_list', 'hyperparameter_configs.jsonl'),
        'r', encoding='utf-8',
    ) as f_in:
        with open(
            os.path.join('hyperparameter_list', 'hyperparameter_list.jsonl'),
            'w', encoding='utf-8',
        ) as f_out:
            for line in f_in:
                config = HyperparameterSearchConfig(**json.loads(line))
                for _ in range(config.num_iters):
                    while True:
                        hyperparams = Hyperparameters(
                            gensim_window=rng.choice(config.space.gensim_window),
                            gensim_sg=rng.choice(config.space.gensim_sg),
                            gensim_negative=rng.choice(config.space.gensim_negative),
                            gensim_sample=rng.choice(config.space.gensim_sample),
                            gensim_epochs=rng.choice(config.space.gensim_epochs),
                            sklearn_fit_intercept=rng.choice(config.space.sklearn_fit_intercept),
                            sklearn_alpha=rng.choice(config.space.sklearn_alpha),
                        )
                        tuple_hyperparams = tuple(hyperparams)
                        if tuple_hyperparams not in seen:
                            seen.add(tuple_hyperparams)
                            break
                    print(hyperparams.model_dump_json(), file=f_out)
                    listener()


#########################################
def main(
) -> None:
    '''
    x
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', required=False, type=int, default=0)
    args = parser.parse_args()
    print('Arguments used:', args)

    print('Hyperparameter lister')
    print('- Getting total number of iterations')
    total_num_iters = 0
    with open(
        os.path.join('hyperparameter_list', 'hyperparameter_configs.jsonl'),
        'r', encoding='utf-8',
    ) as f_in:
        for line in f_in:
            config = HyperparameterSearchConfig(**json.loads(line))
            total_num_iters += config.num_iters

    print('- Generating hyperparameters')
    prog_bar = tqdm.tqdm(total=total_num_iters)
    list_hyperparams(
        seed=args.seed,
        listener=lambda:prog_bar.update(1)
    )
    prog_bar.close()

    print('- Done')


#########################################
if __name__ == '__main__':
    main()
