#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright © 2026 Marc Tanti
#
# This file is part of word2vec_mt project.
'''
Help make a Maltese translation data set.
'''

import argparse
from word2vec_mt.paths import vocab_mt_path, translations_mten_path
from word2vec_mt.data_maker import help_make_translations_data_set, NUM_TRANSLATION_ENTRIES_NEEDED


#########################################
def main(
) -> None:
    '''
    Main function.
    '''
    parser = argparse.ArgumentParser(
        description=(
            'Help make a Maltese translation data set consisting of'
            f' {NUM_TRANSLATION_ENTRIES_NEEDED} entries.'
            ' | Input files:'
            f' * {vocab_mt_path} (extract_vocab.py)'
            ' | Output files:'
            f' * {translations_mten_path}'
        ),
    )
    parser.parse_args()

    help_make_translations_data_set()


#########################################
if __name__ == '__main__':
    main()
