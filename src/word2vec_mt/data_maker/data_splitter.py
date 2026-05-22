'''
Functions for splitting the manually produced synonyms and translations data sets.
'''

import random
import json
from word2vec_mt.data_maker.load_data import load_file
from word2vec_mt.paths import (
    vocab_mt_path, vocab_en_path,
    synonyms_mt_path, translations_mten_path,
    synonyms_mt_split_path, translations_mten_split_path,
)


#########################################

SYNONYM_VAL_FRAC = 0.2
'''
The fraction of the synonym data set to split into a validation set.
'''

SYNONYM_DEV_FRAC = 0.4
'''
The fraction of the synonym data set to split into a development set.
'''

SYNONYM_TEST_FRAC = 0.4
'''
The fraction of the synonym data set to split into a test set.
'''

assert (
    SYNONYM_VAL_FRAC
    + SYNONYM_DEV_FRAC
    + SYNONYM_TEST_FRAC
    == 1.0
)

TRANSLATION_TRAIN_FRAC = 0.5
'''
The fraction of the translation data set to split into a train set.
'''

TRANSLATION_VAL_FRAC = 0.1
'''
The fraction of the translation data set to split into a validation set.
'''

TRANSLATION_DEV_FRAC = 0.2
'''
The fraction of the translation data set to split into a development set.
'''

TRANSLATION_TEST_FRAC = 0.2
'''
The fraction of the translation data set to split into a test set.
'''

assert (
    TRANSLATION_TRAIN_FRAC
    + TRANSLATION_VAL_FRAC
    + TRANSLATION_DEV_FRAC
    + TRANSLATION_TEST_FRAC
    == 1.0
)


#########################################
def split_synonym_data_set(
) -> None:
    '''
    Randomly split the file referred to in synonyms_mt_path into val, dev, and test splits.
    '''
    with open(vocab_mt_path, 'r', encoding='utf-8') as f:
        vocab_mt = set(line.strip() for line in f)
    data = load_file(vocab_mt, vocab_mt, synonyms_mt_path)

    rng = random.Random(0)
    rng.shuffle(data)

    val_size = int(len(data)*(SYNONYM_VAL_FRAC))
    dev_size = int(len(data)*(SYNONYM_DEV_FRAC))
    val = data[0:val_size]
    dev = data[val_size:val_size+dev_size]
    test = data[val_size+dev_size:]

    with open(synonyms_mt_split_path, 'w', encoding='utf-8') as f:
        json.dump({
            "val": {
                "source": [entry.source for entry in val],
                "targets": [sorted(entry.similars) for entry in val],
            },
            "dev": {
                "source": [entry.source for entry in dev],
                "targets": [sorted(entry.similars) for entry in dev],
            },
            "test": {
                "source": [entry.source for entry in test],
                "targets": [sorted(entry.similars) for entry in test],
            },
        }, f, ensure_ascii=False, indent=4)


#########################################
def split_translation_data_set(
) -> None:
    '''
    Randomly split the file referred to in translations_mten_path into train val, dev, and test
    splits.
    '''
    with open(vocab_mt_path, 'r', encoding='utf-8') as f:
        vocab_mt = set(line.strip() for line in f)
    with open(vocab_en_path, 'r', encoding='utf-8') as f:
        vocab_en = set(line.strip() for line in f)
    data = load_file(vocab_mt, vocab_en, translations_mten_path)

    rng = random.Random(0)
    rng.shuffle(data)

    train_size = int(len(data)*(TRANSLATION_TRAIN_FRAC))
    val_size = int(len(data)*(TRANSLATION_VAL_FRAC))
    dev_size = int(len(data)*(TRANSLATION_DEV_FRAC))
    train = data[0:train_size]
    val = data[train_size:train_size+val_size]
    dev = data[train_size+val_size:train_size+val_size+dev_size]
    test = data[train_size+val_size+dev_size:]

    with open(translations_mten_split_path, 'w', encoding='utf-8') as f:
        json.dump({
            "train": {
                "source": [entry.source for entry in train],
                "targets": [sorted(entry.similars) for entry in train],
            },
            "val": {
                "source": [entry.source for entry in val],
                "targets": [sorted(entry.similars) for entry in val],
            },
            "dev": {
                "source": [entry.source for entry in dev],
                "targets": [sorted(entry.similars) for entry in dev],
            },
            "test": {
                "source": [entry.source for entry in test],
                "targets": [sorted(entry.similars) for entry in test],
            },
        }, f, ensure_ascii=False, indent=4)
