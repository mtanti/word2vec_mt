'''
Classes and functions that are common to both model trainers.
'''


#########################################
class TrainListener:
    '''
    A listener for reacting to training progress.
    '''

    #########################################
    def __init__(
        self,
    ) -> None:
        '''
        Empty initialiser.
        '''

    #########################################
    def started_training(
        self,
    ) -> None:
        '''
        Listener for when training starts.
        '''

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

    #########################################
    def started_batch(
        self,
        batch_num: int,
    ) -> None:
        '''
        Listener for when a batch starts being processed.

        :param batch_num: The batch that started.
        '''

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

    #########################################
    def ended_training(
        self,
    ) -> None:
        '''
        Listener for when the training ends.
        '''
