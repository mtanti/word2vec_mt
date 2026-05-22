'''
Classes and functions related to evaluating the linear model (for Maltese to English embedding
translation).
'''

import numpy as np
from word2vec_mt.paths import vocab_mt_path, vocab_en_path, word2vec_mten_report_path
from word2vec_mt.model.data import DataSplit
from word2vec_mt.model.evaluate.common import get_average_precision, get_report_for_one_source


#########################################
def translation_mean_average_precision(
    embedding_matrix_mt: np.ndarray,
    embedding_matrix_en: np.ndarray,
    data: DataSplit,
) -> float:
    '''
    Get the mean average precision (MAP) of retrieving the most similar English tokens to Maltese
    tokens according to the cosine similarity between the Maltese and English token vectors.

    :param embedding_matrix_mt: The Maltese matrix of token vectors (the row-vectors) for the whole
        Maltese vocabulary.
    :param embedding_matrix_en: The English matrix of token vectors (the row-vectors) for the whole
        English vocabulary.
    :param data: The data set indicating which English tokens are similar to which Maltese tokens.
    :return: The mean average precision, which is the mean of the average precision of each source
        token in the data set. It is a number between 0 and 1, where the closer the expected similar
        tokens are to the front of the list of tokens sorted in descending order of cosine
        similarity to source token, the higher the MAP.
    '''
    total_average_precision = 0.0
    for (source_index, targets_indexes) in zip(
        data.source_token_indexes,
        data.targets_token_indexes,
    ):
        total_average_precision += get_average_precision(
            embedding_matrix_mt[source_index],
            embedding_matrix_en,
            targets_indexes,
        )
    return total_average_precision/len(data.source_token_indexes)


#########################################
def get_translation_report(
    embedding_matrix_mt: np.ndarray,
    embedding_matrix_en: np.ndarray,
    data: DataSplit,
) -> None:
    '''
    Create the evaluation report file describing how well the most similar English tokens to Maltese
    tokens are retrieved when using the cosine similarity between the Maltese and English token
    vectors.
    Report is saved as a text file.

    :param embedding_matrix_mt: The Maltese matrix of token vectors (the row-vectors) for the whole
        Maltese vocabulary.
    :param embedding_matrix_en: The English matrix of token vectors (the row-vectors) for the whole
        English vocabulary.
    :param data: The data set indicating which English tokens are similar to which Maltese tokens.
    '''
    with open(vocab_mt_path, 'r', encoding='utf-8') as f:
        vocab_mt = f.read().strip().split('\n')
    with open(vocab_en_path, 'r', encoding='utf-8') as f:
        vocab_en = f.read().strip().split('\n')

    with open(word2vec_mten_report_path, 'w', encoding='utf-8') as f:
        for (source_index, targets_indexes) in zip(
            data.source_token_indexes,
            data.targets_token_indexes,
        ):
            report = get_report_for_one_source(
                embedding_matrix_mt[source_index],
                embedding_matrix_en,
                targets_indexes,
                vocab_en,
            )
            print('Source:', vocab_mt[source_index], file=f)
            print('Top 5 most similar:', ', '.join(report.top_5_tokens), file=f)
            print('Ranks of actual similars:', ', '.join(
                f'{token} - {rank}' for (token, rank) in report.similars_ranks),
                file=f,
            )
            print('', file=f)
