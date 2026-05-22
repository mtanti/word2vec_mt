'''
Data set related modules.
'''

from word2vec_mt.model.data.common import DataSplit, FlatDataSplit
from word2vec_mt.model.data.skipgram import SynonymDataSplits, load_synonym_data_set
from word2vec_mt.model.data.linear import TranslationDataSplits, load_translation_data_set
