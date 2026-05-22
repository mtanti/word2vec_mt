#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright © 2026 Marc Tanti
#
# This file is part of word2vec_mt project.
'''
Extract a vocabulary from the Maltese corpus.
'''

import argparse
from word2vec_mt.paths import corpus_mt_path, vocab_mt_path
from word2vec_mt.vocab_extractor import extract_vocab, MIN_FREQ_VOCAB


#########################################
def main(
) -> None:
    '''
    Main function.
    '''
    parser = argparse.ArgumentParser(
        description=(
            'Extract a vocabulary from the Maltese corpus consisting of tokens made up of'
            f' letters of the alphabet that have a frequency of {MIN_FREQ_VOCAB} or more.'
            ' | Input files:'
            f' * {corpus_mt_path} (download_data_mt.py)'
            ' | Output files:'
            f' * {vocab_mt_path}'
        )
    )
    parser.parse_args()

    extract_vocab()


#########################################
if __name__ == '__main__':
    main()
