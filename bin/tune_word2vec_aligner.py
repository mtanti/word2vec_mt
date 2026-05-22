#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright © 2026 Marc Tanti
#
# This file is part of word2vec_mt project.
'''
Tune the hyperparameters of the Maltese to English word2vec aligner model.
'''

import argparse
from word2vec_mt.paths import (
    vocab_mt_path, vocab_en_path, translations_mten_split_path, word2vec_mt_path, word2vec_en_path,
    linear_hyperparams_config_path, linear_hyperparams_db_path, linear_hyperparams_result_path,
)
from word2vec_mt.model import tune_linear_model


#########################################
def main(
) -> None:
    '''
    Main function.
    '''
    parser = argparse.ArgumentParser(
        description=(
            'Tune the hyperparameters of the Maltese to English word2vec aligner model.'
            ' | Input files:'
            f' * {linear_hyperparams_config_path} (manually set config file),'
            f' * {vocab_mt_path} (extract_vocab.py),'
            f' * {vocab_en_path}  (download_data_en.py),'
            f' * {word2vec_mt_path} (train_word2vec.py),'
            f' * {word2vec_en_path} (download_data_en.py),'
            f' * {translations_mten_split_path} (split_translation_data_set.py)'
            ' | Output files:'
            f' * {linear_hyperparams_db_path},'
            f' * {linear_hyperparams_result_path}'
        ),
    )
    parser.parse_args()

    tune_linear_model()


#########################################
if __name__ == '__main__':
    main()
