'''
Evaluation related modules.
'''

from word2vec_mt.model.evaluate.skipgram import (
    synonym_mean_average_precision, get_synonym_report
)
from word2vec_mt.model.evaluate.linear import (
    translation_mean_average_precision, get_translation_report
)
