#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright © 2026 Marc Tanti
#
# This file is part of word2vec_mt project.
'''
Collect the token frequencies of the tokens in the vocabulary in the Maltese corpus.
'''

import argparse
from word2vec_mt.paths import vocab_mt_path, corpus_mt_path, freqs_mt_path
from word2vec_mt.token_freqs_collector import collect_token_freqs


#########################################
def main(
) -> None:
    '''
    Main function.
    '''
    parser = argparse.ArgumentParser(
        description=(
            'Collect the token frequencies of the tokens in the vocabulary in the Maltese corpus.'
            ' | Input files:'
            f' * {vocab_mt_path} (extract_vocab.py),'
            f' * {corpus_mt_path} (download_data_mt.py)'
            ' | Output files:'
            f' * {freqs_mt_path}'
        ),
    )
    parser.parse_args()

    collect_token_freqs()


#########################################
if __name__ == '__main__':
    main()
