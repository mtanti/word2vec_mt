#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright © 2026 Marc Tanti
#
# This file is part of word2vec_mt project.
'''
Train the Maltese to English word2vec aligner model on the best hyperparameters found.
'''

import argparse
from word2vec_mt.paths import (
    vocab_mt_path, vocab_en_path, translations_mten_split_path, word2vec_mt_path, word2vec_en_path,
    linear_hyperparams_config_path, linear_hyperparams_db_path, linear_model_path,
    linear_hyperparams_best_path, word2vec_mten_path, word2vec_mten_report_path,
)
from word2vec_mt.model import train_best_linear_model


#########################################
def main(
) -> None:
    '''
    Main function.
    '''
    parser = argparse.ArgumentParser(
        description=(
            'Train the Maltese to English word2vec aligner model on the best hyperparameters found.'
            ' | Input files:'
            f' * {linear_hyperparams_config_path} (manually set config file)'
            f' * {vocab_mt_path} (extract_vocab.py),'
            f' * {vocab_en_path} (download_data_en.py),'
            f' * {word2vec_mt_path} (train_word2vec.py),'
            f' * {word2vec_en_path} (download_data_en.py),'
            f' * {translations_mten_split_path} (split_translation_data_set.py),'
            f' * {linear_hyperparams_db_path} (tune_word2vec_aligner.py)'
            ' | Output files:'
            f' * {word2vec_mten_path},'
            f' * {linear_model_path},'
            f' * {linear_hyperparams_best_path}'
            f' * {word2vec_mten_report_path}'
        ),
    )
    parser.parse_args()

    train_best_linear_model()


#########################################
if __name__ == '__main__':
    main()
