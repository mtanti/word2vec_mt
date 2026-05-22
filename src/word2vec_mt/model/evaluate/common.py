'''
Classes and functions that are common to both evaluation tasks.
'''

from dataclasses import dataclass
import numpy as np


#########################################
def cosine_similarity(
    vec: np.ndarray,
    mat: np.ndarray,
) -> np.ndarray:
    '''
    Measure the cosine similarity between a vector and the row-vectors of a matrix.

    :param vec: The vector.
    :param mat: The matrix.
    :return: An array of cosine similarities for each row in the matrix.
    '''
    return (vec@mat.T)/(np.linalg.norm(vec)*np.linalg.norm(mat, axis=1))


#########################################
def get_average_precision(
    source_vec: np.ndarray,
    targets_mat: np.ndarray,
    expected_most_similar_indexes: list[int],
) -> float:
    '''
    Measure the average precision (AP) in order to evaluate how well a generated set of token
    vectors performs when used for retrieving similar tokens to a given token, according to a
    data set of similar tokens (a source token and similar target tokens).

    Given a source token vector and a matrix of token row-vectors:

        #. calculate the cosine similarity between the source and target vectors,
        #. sort the target vectors from most to least similar,
        #. and measure the AP of the ranks of the expected most similar tokens among the sorted
           target tokens: ::

            mean(
                i/rank[token]
                for (i, token) in enumerate(expected_most_similar_tokens_in_order_of_similarity)
            )

    :param source_vec: The vector of the source token.
    :param targets_mat: The matrix of row-vectors of all the tokens in the vocabulary.
    :param expected_most_similar_indexes: The token indexes that are expected to be ranked the
        highest. Indexes correspond to row indexes in targets_mat.
    :return: The average precision, a number between 1 and 0 where the closer the
        expected_most_similar_indexes are to the front of the list of tokens sorted in descending
        order of cosine similarity to source_vec, the higher the AP.
    '''
    # Get the cosine similarity of each token vector in targets_mat compared to source_vec.
    cos_sims = cosine_similarity(source_vec, targets_mat)

    # Get the indexes in cos_sims as they would be positioned if sorted in descending order.
    # e.g. [2.0, 3.0, 1.0] becomes [1, 0, 2]
    similarity_sorted_indexes = np.argsort(-cos_sims)

    # Get the ranks of the token indexes in expected_most_similar_indexes as they appear in
    # similarity_sorted_indexes.
    # np.where([True, False, True, False]) returns [0, 2], i.e. the indexes of True values.
    # Ranks start from 1, not 0, hence the +1.
    # The ranks are always returned in ascending order.
    # e.g.
    #  similarity_sorted_indexes = [1, 0, 2, 5, 3, 4]
    #  expected_most_similar_indexes = [0, 5]
    #  similars_ranks = [2, 4] (index 0 has rank 2 and index 5 has rank 4)
    similars_ranks = np.where(
        np.isin(similarity_sorted_indexes, expected_most_similar_indexes)
    )[0] + 1

    # Get the average precision from the ranks.
    # e.g. [2, 4, 6] becomes (1/2 + 2/4 + 3/6)/3 = 0.5
    average_precision = (np.arange(1, len(expected_most_similar_indexes) + 1)/similars_ranks).mean()

    return average_precision


#########################################
@dataclass
class Report:
    '''
    An evaluation report of how similar to a source token the expected similar tokens are.
    '''

    top_5_tokens: list[str]
    '''
    The actual 5 most similar tokens to the source token found.
    '''

    similars_ranks: list[tuple[str, int]]
    '''
    The ranks of the expected similar tokens when sorted by cosine similarity in descending order.
    '''


#########################################
def get_report_for_one_source(
    source_vec: np.ndarray,
    targets_mat: np.ndarray,
    expected_most_similar_indexes: list[int],
    vocab: list[str],
) -> Report:
    '''
    Build an evaluation report for one source token.

    :param source_vec: The token vector of the source token.
    :param targets_mat: The token vectors of all the vocabulary as row-vectors in a matrix.
    :param expected_most_similar_indexes: The row indexes in targets_mat that are expected to be the
        most similar to source_vec.
    :param vocab: The vocabulary list of tokens corresponding to the rows in targets_mat.
    :return: An evaluation report.
    '''
    similarity_sorted_indexes = np.argsort(-cosine_similarity(source_vec, targets_mat)).tolist()
    top_5_tokens = [vocab[index] for index in similarity_sorted_indexes[:5]]
    similars_ranks = sorted(
        [(vocab[i], similarity_sorted_indexes.index(i) + 1) for i in expected_most_similar_indexes],
        key=lambda pair: pair[1],
    )
    return Report(
        top_5_tokens=top_5_tokens,
        similars_ranks=similars_ranks,
    )
