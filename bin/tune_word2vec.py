#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright © 2026 Marc Tanti
#
# This file is part of word2vec_mt project.
'''
Tune the hyperparameters of the word2vec model for Maltese.
'''

import argparse
from word2vec_mt.paths import (
    vocab_mt_path, synonyms_mt_split_path, proccorpus_mt_path, skipgram_hyperparams_config_path,
    skipgram_hyperparams_db_path, skipgram_hyperparams_result_path,
)
from word2vec_mt.model import tune_skipgram_model



#########################################
def main(
) -> None:
    '''
    Main function.
    '''
    parser = argparse.ArgumentParser(
        description=(
            'Tune the hyperparameters of the word2vec model for Maltese.'
            ' | Input files:'
            f' * {skipgram_hyperparams_config_path} (manually set config file),'
            f' * {vocab_mt_path} (extract_vocab.py),'
            f' * {synonyms_mt_split_path} (split_synonym_data_set.py),'
            f' * {proccorpus_mt_path} (preprocess_corpus_to_train_set.py)'
            ' | Output files:'
            f' * {skipgram_hyperparams_db_path},'
            f' * {skipgram_hyperparams_result_path}'
        ),
    )
    parser.parse_args()

    tune_skipgram_model()


#########################################
if __name__ == '__main__':
    main()
