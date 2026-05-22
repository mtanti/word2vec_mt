'''
Classes and functions related to the skipgram model (the actual Maltese word2vec model).
'''

import json
from dataclasses import dataclass
from word2vec_mt.model.data.common import DataSplit
from word2vec_mt.paths import vocab_mt_path, synonyms_mt_split_path


#########################################
@dataclass
class SynonymDataSplits:
    '''
    The collection of data splits for the skipgram model (no train set as the unlabelled corpus is
    used for that).
    '''

    val: DataSplit
    '''
    The validation set (for early stopping).
    '''

    dev: DataSplit
    '''
    The development set (for hyperparameter tuning).
    '''

    test: DataSplit
    '''
    The test set (for final evaluation).
    '''


#########################################
def load_synonym_data_set(
) -> SynonymDataSplits:
    '''
    Load the skipgram model's data splits.

    :return: The skipgram model's data splits.
    '''
    with open(vocab_mt_path, 'r', encoding='utf-8') as f:
        token2index_mt = {token: i for (i, token) in enumerate(line.strip() for line in f)}
    with open(synonyms_mt_split_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        val = data['val']
        dev = data['dev']
        test = data['test']

    return SynonymDataSplits(
        val=DataSplit(
            source_token_indexes=[token2index_mt[source] for source in val['source']],
            targets_token_indexes=[
                [token2index_mt[target] for target in targets]
                for targets in val['targets']
            ],
        ),
        dev=DataSplit(
            source_token_indexes=[token2index_mt[source] for source in dev['source']],
            targets_token_indexes=[
                [token2index_mt[target] for target in targets]
                for targets in dev['targets']
            ],
        ),
        test=DataSplit(
            source_token_indexes=[token2index_mt[source] for source in test['source']],
            targets_token_indexes=[
                [token2index_mt[target] for target in targets]
                for targets in test['targets']
            ],
        ),
    )
