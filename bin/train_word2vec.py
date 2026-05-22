#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright © 2026 Marc Tanti
#
# This file is part of word2vec_mt project.
'''
Train the word2vec model for Maltese on the best hyperparameters found.
'''

import argparse
from word2vec_mt.paths import (
    vocab_mt_path, synonyms_mt_split_path, proccorpus_mt_path, skipgram_hyperparams_config_path,
    skipgram_hyperparams_db_path, skipgram_model_path, word2vec_mt_path,
    skipgram_hyperparams_best_path, word2vec_mt_report_path,
)
from word2vec_mt.model import train_best_skipgram_model


#########################################
def main(
) -> None:
    '''
    Main function.
    '''
    parser = argparse.ArgumentParser(
        description=(
            'Train the word2vec model for Maltese on the best hyperparameters found.'
            ' | Input files:'
            f' * {skipgram_hyperparams_config_path} (manually set config file),'
            f' * {vocab_mt_path} (extract_vocab.py),'
            f' * {synonyms_mt_split_path} (split_synonym_data_set.py),'
            f' * {proccorpus_mt_path} (preprocess_corpus_to_train_set.py),'
            f' * {skipgram_hyperparams_db_path} (tune_word2vec.py)'
            ' | Output files:'
            f' * {word2vec_mt_path},'
            f' * {skipgram_model_path},'
            f' * {skipgram_hyperparams_best_path}'
            f' * {word2vec_mt_report_path}'
        ),
    )
    parser.parse_args()

    train_best_skipgram_model()


#########################################
if __name__ == '__main__':
    main()
