'''
Classes and functions that are common to both model tuners.
'''

from typing import Optional
import tqdm
from word2vec_mt.model.trainer import TrainListener


#########################################
class DuplicateHyperparametersAttempted(Exception):
    '''
    An exception for signalling that the hyperparameters that were attempted for evaluation were
    already attempted before.
    '''


#########################################
class Listener(TrainListener):
    '''
    A subclass of the training progress listener for showing progress while tuning and training.
    '''

    #########################################
    def __init__(
        self,
    ) -> None:
        '''
        Initialiser.
        '''
        self.progbar: Optional[tqdm.tqdm] = None

    #########################################
    def started_training(
        self,
    ) -> None:
        '''
        Listener for when training starts.
        '''
        print()

    #########################################
    def started_epoch(
        self,
        epoch_num: int,
        num_batches: int,
    ) -> None:
        '''
        Listener for when an epoch of training starts (an epoch consists of multiple batches).

        :param epoch_num: The epoch that started.
        :param num_batches: The number of batches in the epoch.
        '''
        print('-----------')
        print('epoch', epoch_num)
        self.progbar = tqdm.tqdm(total=num_batches)

    #########################################
    def ended_batch(
        self,
        batch_num: int,
        num_batches: int,
        train_error: float,
    ) -> None:
        '''
        Listener for when a batch ends.

        :param batch_num: The batch that ended.
        :param num_batches: The number of batches in the epoch.
        :param train_error: The cross-entropy error given by the batch.
        '''
        assert self.progbar is not None
        self.progbar.update()
        if batch_num == num_batches:
            self.progbar.close()
            self.progbar = None

    #########################################
    def ended_epoch(
        self,
        epoch_num: int,
        val_map: float,
        new_best: bool,
        num_bad_epochs: int,
    ) -> None:
        '''
        Listener for when an epoch ends.

        :param epoch_num: The epoch that ended.
        :param val_map: The mean average precision obtained on the validation set.
        :param new_best: Whether the mean average precision obtained was the best so far.
        :param num_bad_epochs: The number of epochs with a less than best mean average precision.
        '''
        print('ended epoch with val map:', val_map)
        print()
