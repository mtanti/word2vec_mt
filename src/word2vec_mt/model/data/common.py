'''
Classes and functions that are common to both data sets.
'''

from dataclasses import dataclass
import numpy as np


#########################################
@dataclass
class FlatDataSplit:
    '''
    A flat version of the token / list of similar tokens data set.
    '''

    source_token_indexes: np.ndarray
    '''
    The array of indexes of source tokens that are compared to the similar tokens.
    '''

    similar_token_indexes: np.ndarray
    '''
    The array of indexes of the similar tokens.
    '''


#########################################
@dataclass
class DataSplit:
    '''
    A data set mapping tokens to similar tokens.
    '''

    source_token_indexes: list[int]
    '''
    The source token indexes.
    '''

    targets_token_indexes: list[list[int]]
    '''
    A list of lists containing target token indexes that are similar to the corresponding source
    token index.
    '''

    def flatten(
        self,
    ) -> FlatDataSplit:
        '''
        Flatten the data set into two lists of token indexes for use with neural networks.

        :return: The flat data set.
        '''
        return FlatDataSplit(
            source_token_indexes=np.fromiter((
                self.source_token_indexes[i]
                for i in range(len(self.source_token_indexes))
                for _ in self.targets_token_indexes[i]
            ), np.int32),
            similar_token_indexes=np.fromiter((
                similar
                for i in range(len(self.source_token_indexes))
                for similar in self.targets_token_indexes[i]
            ), np.int32),
        )
