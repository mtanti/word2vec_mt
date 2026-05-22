#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright © 2026 Marc Tanti
#
# This file is part of word2vec_mt project.
'''
Split the Maltese synonyms data set into train, val, dev, and test splits.
'''

import argparse
from word2vec_mt.paths import vocab_mt_path, synonyms_mt_path, synonyms_mt_split_path
from word2vec_mt.data_maker.data_splitter import split_synonym_data_set


#########################################
def main(
) -> None:
    '''
    Main function.
    '''
    parser = argparse.ArgumentParser(
        description=(
            'Split the Maltese translation data set into train, val, dev, and test splits.'
            ' | Input files:'
            f' * {vocab_mt_path} (extract_vocab.py),'
            f' * {synonyms_mt_path} (help_make_synonym_data_set.py)'
            ' | Output files:'
            f' * {synonyms_mt_split_path}'
        ),
    )
    parser.parse_args()

    split_synonym_data_set()


#########################################
if __name__ == '__main__':
    main()
