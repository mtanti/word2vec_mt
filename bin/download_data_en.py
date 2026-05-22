#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright © 2026 Marc Tanti
#
# This file is part of word2vec_mt project.
'''
Download the English word2vec embeddings (Google News 300).
'''

import argparse
from word2vec_mt.paths import vocab_en_path, word2vec_en_path
from word2vec_mt.data_downloader import download_en


#########################################
def main(
) -> None:
    '''
    Main function.
    '''
    parser = argparse.ArgumentParser(
        description=(
            'Download the English word2vec embeddings (Google News 300).'
            ' | Input files: none'
            ' | Output files:'
            f' * {vocab_en_path},'
            f' * {word2vec_en_path}'
        )
    )
    parser.parse_args()

    download_en()


#########################################
if __name__ == '__main__':
    main()
