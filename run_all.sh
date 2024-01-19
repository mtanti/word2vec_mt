#!/bin/bash
set -e

source venv/bin/activate

python get_en_word2vec.py
python get_mt_corpus.py
python get_bilingual_dict.py
python tune_mt_word2vec.py
