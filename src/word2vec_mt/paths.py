'''
Constants that are common throughout the project.
'''

import os
import word2vec_mt


#########################################

corpus_mt_path: str = os.path.abspath(os.path.join(
    word2vec_mt.path, '..', '..', 'data', 'corpus_mt.txt',
))
'''
The path to the Maltese corpus text file.
'''

vocab_mt_path: str = os.path.abspath(os.path.join(
    word2vec_mt.path, '..', '..', 'output', 'vocab_mt.txt',
))
'''
The path to the Maltese corpus text file.
'''

synonyms_mt_path: str = os.path.abspath(os.path.join(
    word2vec_mt.path, '..', '..', 'output', 'synonyms_mt.jsonl',
))
'''
The path to the Maltese text file of synonyms.
'''

synonyms_mt_split_path: str = os.path.abspath(os.path.join(
    word2vec_mt.path, '..', '..', 'data', 'synonyms_mt.json',
))
'''
The path to the Maltese text file of synonyms split into data sets.
'''

proccorpus_mt_path: str = os.path.abspath(os.path.join(
    word2vec_mt.path, '..', '..', 'data', 'proccorpus_mt.hdf5',
))
'''
The path to the processed Maltese corpus file.
'''

freqs_mt_path: str = os.path.abspath(os.path.join(
    word2vec_mt.path, '..', '..', 'data', 'token_freqs_mt.json',
))
'''
The path to the token frequencies in the Maltese corpus JSON file.
'''

skipgram_hyperparams_config_path: str = os.path.abspath(os.path.join(
    word2vec_mt.path, '..', '..', 'data', 'skipgram_hyperparams.json',
))
'''
The path to the skipgram model's hyperparameters configuration file.
'''

skipgram_hyperparams_db_path: str = os.path.abspath(os.path.join(
    word2vec_mt.path, '..', '..', 'data', 'skipgram_hyperparams.sqlite3',
))
'''
The path to the skipgram model's hyperparameter search results file (machine readable).
'''

skipgram_hyperparams_result_path: str = os.path.abspath(os.path.join(
    word2vec_mt.path, '..', '..', 'data', 'skipgram_hyperparams.csv',
))
'''
The path to the skipgram model's hyperparameter search results file (human readable).
'''

skipgram_hyperparams_best_path: str = os.path.abspath(os.path.join(
    word2vec_mt.path, '..', '..', 'data', 'skipgram_best_hyperparams.json',
))
'''
The path to the skipgram model's best hyperparameters found.
'''

skipgram_model_path: str = os.path.abspath(os.path.join(
    word2vec_mt.path, '..', '..', 'data', 'skipgram_model.pt',
))
'''
The path to the saved final skipgram model (after tuning).
'''

word2vec_mt_report_path: str = os.path.abspath(os.path.join(
    word2vec_mt.path, '..', '..', 'output', 'word2vec_mt_report.txt',
))
'''
The path to the report showing how good the Maltese word2vec embeddings are.
'''

word2vec_mt_path: str = os.path.abspath(os.path.join(
    word2vec_mt.path, '..', '..', 'output', 'word2vec_mt.npy',
))
'''
The path to the Maltese word2vec NumPy file.
'''

vocab_en_path: str = os.path.abspath(os.path.join(
    word2vec_mt.path, '..', '..', 'data', 'vocab_en.txt',
))
'''
The path to the English vocabulary text file.
'''

word2vec_en_path: str = os.path.abspath(os.path.join(
    word2vec_mt.path, '..', '..', 'data', 'word2vec_en.npy',
))
'''
The path to the English word2vec NumPy file.
'''

translations_mten_path: str = os.path.abspath(os.path.join(
    word2vec_mt.path, '..', '..', 'output', 'translations_mten.jsonl',
))
'''
The path to the Maltese-English text file of translated words.
'''

translations_mten_split_path: str = os.path.abspath(os.path.join(
    word2vec_mt.path, '..', '..', 'data', 'translations_mten.json',
))
'''
The path to the Maltese-English text file of translated words split into data sets.
'''

linear_hyperparams_config_path: str = os.path.abspath(os.path.join(
    word2vec_mt.path, '..', '..', 'data', 'linear_hyperparams.json',
))
'''
The path to the linear model's hyperparameters configuration file.
'''

linear_hyperparams_db_path: str = os.path.abspath(os.path.join(
    word2vec_mt.path, '..', '..', 'data', 'linear_hyperparams.sqlite3',
))
'''
The path to the linear model's hyperparameter search results file (machine readable).
'''

linear_hyperparams_result_path: str = os.path.abspath(os.path.join(
    word2vec_mt.path, '..', '..', 'data', 'linear_hyperparams.csv',
))
'''
The path to the linear model's hyperparameter search results file (human readable).
'''

linear_hyperparams_best_path: str = os.path.abspath(os.path.join(
    word2vec_mt.path, '..', '..', 'data', 'linear_best_hyperparams.json',
))
'''
The path to the linear model's best hyperparameters found from search file.
'''

linear_model_path: str = os.path.abspath(os.path.join(
    word2vec_mt.path, '..', '..', 'data', 'linear_model.pt',
))
'''
The path to the saved final linear model (after tuning).
'''

word2vec_mten_report_path: str = os.path.abspath(os.path.join(
    word2vec_mt.path, '..', '..', 'output', 'word2vec_mten_report.txt',
))
'''
The path to the report showing how good the English-aligned Maltese word2vec embeddings are.
'''

word2vec_mten_path: str = os.path.abspath(os.path.join(
    word2vec_mt.path, '..', '..', 'output', 'word2vec_mten.npy',
))
'''
The path to the English-aligned Maltese word2vec NumPy file.
'''
