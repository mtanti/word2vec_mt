#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright © 2026 Marc Tanti
#
# This file is part of word2vec_mt project.
'''
Download the MLRS Maltese corpus to be used for creating word2vec embeddings.
'''

import argparse
from word2vec_mt.paths import corpus_mt_path
from word2vec_mt.data_downloader import download_mt


#########################################
def main(
) -> None:
    '''
    Main function.
    '''
    parser = argparse.ArgumentParser(
        description=(
            'Download the MLRS Maltese corpus to be used for creating word2vec embeddings.'
            ' | Input files: none'
            ' | Output files:'
            f' * {corpus_mt_path}'
        )
    )
    parser.parse_args()

    download_mt()


#########################################
if __name__ == '__main__':
    main()
