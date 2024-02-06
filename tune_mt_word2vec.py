'''
x
'''
import os
import json
import pickle
import datetime
import argparse
from typing import Optional
import numpy as np
import tqdm
from gensim.models import KeyedVectors
import gensim.models
import sklearn.linear_model
import model_trainer
from hyperparameter_config import Hyperparameters


#########################################
class Listener(model_trainer.SearchListener):
    '''
    x
    '''

    def __init__(
        self,
        best_mean_cosine: float = 0.0
    ) -> None:
        '''
        x
        '''
        super().__init__()
        self.best_mean_cosine: float = best_mean_cosine
        self.prog_bar: Optional[tqdm.tqdm] = None

    def iteration_start(
        self,
        iteration: int,
        hyperparams: Hyperparameters,
        num_epochs: int,
    ) -> None:
        '''
        x
        '''
        timestamp = str(datetime.datetime.now())
        print(f'  - iteration {iteration}')
        print(f'    - timestamp: {timestamp}')
        print(f'    - hyperparams: {hyperparams.model_dump_json()}')
        with open('tune_log.txt', 'a', encoding='utf-8') as f:
            print(f'iteration: {iteration}', file=f)
            print(f'timestamp: {timestamp}', file=f)
            print(f'hyperparams: {hyperparams.model_dump_json()}', file=f)
        self.prog_bar = tqdm.tqdm(total=num_epochs)

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
        assert self.prog_bar is not None
        self.prog_bar.close()

        print(f'    - failed: {error}')
        print(f'    - duration: {duration}')
        with open('tune_log.txt', 'a', encoding='utf-8') as f:
            print(f'failed: {error}', file=f)
            print(f'duration: {duration}', file=f)
            print('', file=f)

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
        assert self.prog_bar is not None
        self.prog_bar.close()

        new_best = False
        if mean_cosine > self.best_mean_cosine:
            for fname in os.listdir('model'):
                os.rename(
                    os.path.join('model', fname),
                    os.path.join('model', '_'+fname)
                )
            self.new_best_found(hyperparams, mean_cosine, model_mt, translator)
            self.best_mean_cosine = mean_cosine
            new_best = True

        print(f'    - mean cosine: {mean_cosine}')
        print(f'    - new best? {new_best}')
        print(f'    - duration: {duration}')

        with open('tune_log.txt', 'a', encoding='utf-8') as f:
            print(f'mean cosine: {mean_cosine}', file=f)
            print(f'new best? {new_best}', file=f)
            print(f'duration: {duration}', file=f)
            print('', file=f)

        for fname in os.listdir('model'):
            if fname.startswith('_'):
                os.remove(os.path.join('model', fname))

    def epoch_end(
        self,
        epoch_num: int,
    ) -> None:
        assert self.prog_bar is not None
        self.prog_bar.update(1)

    def new_best_found(
        self,
        hyperparams: Hyperparameters,
        best_mean_cosine: float,
        model_mt: gensim.models.Word2Vec,
        translator: sklearn.linear_model.Ridge,
    ) -> None:
        '''
        x
        '''
        with open(os.path.join('model', 'best_hyperparameters.json'), 'w', encoding='utf-8') as f:
            print(hyperparams.model_dump_json(indent=4), file=f)

        with open(os.path.join('model', 'best_dev_mean_cosine.json'), 'w', encoding='utf-8') as f:
            json.dump(best_mean_cosine, f)

        model_mt.wv.save(os.path.join('model', 'word2vec_mt.wordvectors'))
        with open(os.path.join('model', 'word2vec_mt2en.pickle'), 'wb') as f:
            pickle.dump(translator, f, protocol=3)

        with open(os.path.join('data', 'mt_vocab.txt'), 'r', encoding='utf-8') as f:
            vocab_mt = f.read().strip().split('\n')
        transformed_wordvecs_mt = np.array([
            translator.predict(model_mt.wv[word][None, :])[0, :]
            for word in vocab_mt
        ], np.float32)
        np.save(os.path.join('model', 'mt2en_projected_word2vec.npy'), transformed_wordvecs_mt)

        wordvecs_en = KeyedVectors.load(os.path.join('data', 'word2vec_en.wordvectors'))
        wordvecs_mt = model_mt.wv
        with open(os.path.join('data', 'bilingual_dict_test.json'), 'r', encoding='utf-8') as f:
            rows = json.load(f)
            test_mt = [row['mt'] for row in rows]
            test_en = [row['en'] for row in rows]
        with open(os.path.join('data', 'mt_vocab.txt'), 'r', encoding='utf-8') as f:
            word2index_mt = {
                word_mt: i
                for (i, word_mt) in enumerate(f.read().strip().split('\n'))
            }
        with open(os.path.join('model', 'test_results.txt'), 'w', encoding='utf-8') as f:
            for (word_mt, word_en) in zip(test_mt, test_en):
                print(f'Most similar words to {word_mt} ({word_en})', file=f)
                print('In Maltese:', file=f)
                for hyp_word_mt in wordvecs_mt.similar_by_key(word_mt, topn=5):
                    print(f'- {hyp_word_mt}', file=f)
                print('In English:', file=f)
                wordvec_mt = transformed_wordvecs_mt[word2index_mt[word_mt], :]
                for hyp_word_en in wordvecs_en.similar_by_vector(wordvec_mt, topn=5):
                    print(f'- {hyp_word_en}', file=f)
                print('', file=f)


#########################################
def main(
) -> None:
    '''
    x
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', required=False, type=int, default=0)
    parser.add_argument('--workers', required=False, type=int, default=1)
    parser.add_argument('--skip_to_iter', required=False, type=int, default=None)
    args = parser.parse_args()
    print('Arguments used:', args)
    os.makedirs('model', exist_ok=True)

    print('Tuning Maltese word2vec')

    print('- Loading data sets')
    with open(os.path.join('data', 'bilingual_dict_train.json'), 'r', encoding='utf-8') as f:
        rows = json.load(f)
        train_mt = [row['mt'] for row in rows]
        train_en = [row['en'] for row in rows]

    with open(os.path.join('data', 'bilingual_dict_dev.json'), 'r', encoding='utf-8') as f:
        rows = json.load(f)
        dev_mt = [row['mt'] for row in rows]
        dev_en = [row['en'] for row in rows]

    wordvecs_en = KeyedVectors.load(os.path.join('data', 'word2vec_en.wordvectors'))

    best_mean_cosine = 0.0
    if args.skip_to_iter is not None:
        with open(os.path.join('model', 'best_dev_mean_cosine.json'), 'r', encoding='utf-8') as f:
            best_mean_cosine = json.load(f)

    print('- Tuning')
    model_trainer.search_hyperparams(
        train_mt,
        train_en,
        dev_mt,
        dev_en,
        wordvecs_en,
        Listener(best_mean_cosine),
        seed=args.seed,
        workers=args.workers,
        skip_to_iter=args.skip_to_iter,
    )

    print('- Done')


#########################################
if __name__ == '__main__':
    main()
