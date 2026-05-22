#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright © 2026 Marc Tanti
#
# This file is part of word2vec_mt project.
'''
Split the Maltese translation data set into train, val, dev, and test splits.
'''

import argparse
from word2vec_mt.paths import (
    vocab_mt_path, vocab_en_path,
    translations_mten_path, translations_mten_split_path,
)
from word2vec_mt.data_maker.data_splitter import split_translation_data_set


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
            f' * {vocab_en_path} (download_data_en.py),'
            f' * {translations_mten_path} (help_make_translation_data_set.py)'
            ' | Output files:'
            f' * {translations_mten_split_path}'
        ),
    )
    parser.parse_args()

    split_translation_data_set()


#########################################
if __name__ == '__main__':
    main()
