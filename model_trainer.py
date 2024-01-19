'''
x
'''
import os
import timeit
import random
from typing import Optional, Callable, Any
from gensim.models import KeyedVectors
from gensim.models.callbacks import CallbackAny2Vec
import gensim.models
import sklearn.linear_model
from hyperparameter_config import SearchSpace, Hyperparameters


#########################################
class EpochListener(CallbackAny2Vec):
    '''
    x
    '''

    def __init__(
        self,
        listener: Callable[[int], None],
    ) -> None:
        '''
        x
        '''
        self.listener = listener
        self.epoch_num: int = 0

    def on_epoch_end(
        self,
        model: Any,
    ) -> None:
        '''
        x
        '''
        self.epoch_num += 1
        self.listener(self.epoch_num)


#########################################
def train_model(
    train_mt: list[str],
    train_en: list[str],
    wordvecs_en: KeyedVectors,
    hyperparams: Hyperparameters,
    seed: int = 0,
    workers: int = 1,
    epoch_listener: Callable[[int], None] = lambda epoch_num: None,
) -> tuple[
    gensim.models.Word2Vec,
    sklearn.linear_model.Ridge,
]:
    '''
    x
    '''
    os.environ['PYTHONHASHSEED'] = str(seed)
    model_mt = gensim.models.Word2Vec(
        corpus_file=os.path.join('data', 'mt_corpus.txt'),
        vector_size=300,
        min_count=5,
        workers=workers,
        seed=seed,
        ns_exponent=0.75,
        epochs=hyperparams.gensim_epochs,
        window=hyperparams.gensim_window,
        sg=hyperparams.gensim_sg,
        negative=hyperparams.gensim_negative,
        sample=hyperparams.gensim_sample,
        callbacks=[EpochListener(epoch_listener)]
    )
    wordvecs_mt = model_mt.wv

    translator = sklearn.linear_model.Ridge(
        fit_intercept=hyperparams.sklearn_fit_intercept,
        alpha=hyperparams.sklearn_alpha,
        max_iter=None,
        tol=0.001,
        solver='auto',
        random_state=0,
    ).fit(
        [wordvecs_mt[word] for word in train_mt],
        [wordvecs_en[word] for word in train_en],
    )

    return (model_mt, translator)


#########################################
def eval_model(
    model_mt: gensim.models.Word2Vec,
    translator: sklearn.linear_model.Ridge,
    test_mt: list[str],
    test_en: list[str],
    wordvecs_en: KeyedVectors,
) -> float:
    '''
    x
    '''
    wordvecs_mt = model_mt.wv

    transformed_dev_wordvecs_mt = [
        translator.predict(wordvecs_mt[word][None, :])[0, :]
        for word in test_mt
    ]

    cosine_sims = [
        KeyedVectors.cosine_similarities(vec_mt, [vec_en])[0].tolist()
        for (vec_mt, vec_en) in zip(
            transformed_dev_wordvecs_mt,
            [wordvecs_en[word].tolist() for word in test_en]
        )
    ]

    return sum(cosine_sims)/len(cosine_sims)


#########################################
class SearchListener:
    '''
    x
    '''

    def iteration_start(
        self,
        iteration: int,
        hyperparams: Hyperparameters,
        num_epochs: int,
    ) -> None:
        '''
        x
        '''

    def iteration_fail(
        self,
        iteration: int,
        hyperparams: Hyperparameters,
        duration: float,
        error: str,
    ) -> None:
        '''
        x
        '''

    def iteration_end(
        self,
        iteration: int,
        hyperparams: Hyperparameters,
        model_mt: gensim.models.Word2Vec,
        translator: sklearn.linear_model.Ridge,
        mean_cosine: float,
        duration: float,
    ) -> None:
        '''
        x
        '''

    def epoch_end(
        self,
        epoch_num: int,
    ) -> None:
        '''
        x
        '''


#########################################
def search_hyperparams(
    train_mt: list[str],
    train_en: list[str],
    dev_mt: list[str],
    dev_en: list[str],
    wordvecs_en: KeyedVectors,
    num_tuning_iters: int,
    listener: SearchListener = SearchListener(),
    seed: int = 0,
    workers: int = 1,
    skip_to_iter: Optional[int] = None,
) -> None:
    '''
    x
    '''
    rng = random.Random(seed)
    seen = set()
    for i in range(1, num_tuning_iters + 1):
        if skip_to_iter is not None and i < skip_to_iter:
            continue
        while True:
            hyperparams = Hyperparameters(
                gensim_window=rng.choice(SearchSpace.gensim_window),
                gensim_sg=rng.choice(SearchSpace.gensim_sg),
                gensim_negative=rng.choice(SearchSpace.gensim_negative),
                gensim_sample=rng.choice(SearchSpace.gensim_sample),
                gensim_epochs=rng.choice(SearchSpace.gensim_epochs),
                sklearn_fit_intercept=rng.choice(SearchSpace.sklearn_fit_intercept),
                sklearn_alpha=rng.choice(SearchSpace.sklearn_alpha),
            )
            tuple_hyperparams = tuple(hyperparams)
            if tuple_hyperparams not in seen:
                seen.add(tuple_hyperparams)
                break

        listener.iteration_start(i, hyperparams, hyperparams.gensim_epochs)
        start_time = timeit.default_timer()

        error_msg: Optional[str] = None
        try:
            (model_mt, translator) = train_model(
                train_mt,
                train_en,
                wordvecs_en,
                hyperparams,
                seed,
                workers,
                listener.epoch_end,
            )
            mean_cosine = eval_model(
                model_mt,
                translator,
                dev_mt,
                dev_en,
                wordvecs_en,
            )
        except ValueError as ex:
            error_msg = str(ex)

        duration = timeit.default_timer() - start_time
        if error_msg is None:
            listener.iteration_end(
                i, hyperparams, model_mt, translator, mean_cosine, duration
            )
        else:
            listener.iteration_fail(
                i, hyperparams, duration, error_msg
            )
