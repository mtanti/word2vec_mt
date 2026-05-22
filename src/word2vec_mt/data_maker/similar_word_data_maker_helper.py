'''
A helper program to assist with manually constructing a similar word data
set consisting of a sample of (source) words and their similar words taken from
the vocabulary.
'''

from word2vec_mt.data_maker.user_input_logic import (
    help_make_similar_word_data_set
)
from word2vec_mt.paths import (
    vocab_mt_path, vocab_en_path, synonyms_mt_path, translations_mten_path
)


#########################################

NUM_SYNONYM_ENTRIES_NEEDED = 100
'''
The number of synonym words to ask the user to enter.
'''

NUM_TRANSLATION_ENTRIES_NEEDED = 200
'''
The number of translation words to ask the user to enter.
'''


#########################################
def help_make_synonym_data_set(
) -> None:
    '''
    Help make a manually constructed synonymous words data set.
    '''
    print('Helping to construct a synonymous words data set')
    print()

    with open(vocab_mt_path, 'r', encoding='utf-8') as f:
        vocab_mt = set(line.strip() for line in f)

    help_make_similar_word_data_set(
        source_vocab=vocab_mt,
        similars_vocab=vocab_mt,
        num_entries_needed=NUM_SYNONYM_ENTRIES_NEEDED,
        output_path=synonyms_mt_path,
        seed=0,
    )


#########################################
def help_make_translations_data_set(
) -> None:
    '''
    Help make a manually constructed translated words data set.
    '''
    print('Helping to construct a translated words data set')
    print()

    with open(vocab_mt_path, 'r', encoding='utf-8') as f:
        vocab_mt = set(line.strip() for line in f)
    with open(vocab_en_path, 'r', encoding='utf-8') as f:
        vocab_en = set(line.strip() for line in f)

    help_make_similar_word_data_set(
        source_vocab=vocab_mt,
        similars_vocab=vocab_en,
        num_entries_needed=NUM_TRANSLATION_ENTRIES_NEEDED,
        output_path=translations_mten_path,
        seed=1,
    )
