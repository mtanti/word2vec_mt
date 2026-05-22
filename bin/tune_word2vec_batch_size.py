#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright © 2026 Marc Tanti
#
# This file is part of word2vec_mt project.
'''
Tune the batch size of the word2vec model for Maltese such that a batch uses all available VRAM.
'''

import argparse
from word2vec_mt.paths import (
    vocab_mt_path, synonyms_mt_split_path, proccorpus_mt_path, skipgram_hyperparams_config_path,
)
from word2vec_mt.model import optimise_skipgram_batch_size



#########################################
def main(
) -> None:
    '''
    Main function.
    '''
    parser = argparse.ArgumentParser(
        description=(
            'Tune the batch size of the word2vec model for Maltese such that a batch uses all'
            ' available VRAM.'
            ' | Input files:'
            f' * {skipgram_hyperparams_config_path} (manually set config file),'
            f' * {vocab_mt_path} (extract_vocab.py),'
            f' * {synonyms_mt_split_path} (split_synonym_data_set.py),'
            f' * {proccorpus_mt_path} (preprocess_corpus_to_train_set.py)'
            ' | Output files: none'
        ),
    )
    parser.add_argument(
        'memory_for_superbatch',
        type=float,
        help='The amount of main memory in GB to be reserved for the superbatch.',
    )
    args = parser.parse_args()

    optimise_skipgram_batch_size(int(args.memory_for_superbatch*1e9))


#########################################
if __name__ == '__main__':
    main()
