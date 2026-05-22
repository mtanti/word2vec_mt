'''
Classes and functions related to optimising the batch size for maximise training speed on the given
computer's hardware. The superbatch is also optimised but only in terms of available memory (which
is manually specified rather than based on actual memory available) so that as much of the training
items on disk can be loaded into a superbatch in memory.
'''

import json
import timeit
import math
import torch
import h5py
from word2vec_mt.model.trainer import train_skipgram_model, NegativeTokenSampler, TrainListener
from word2vec_mt.model.data import load_synonym_data_set
from word2vec_mt.paths import (
    vocab_mt_path, freqs_mt_path,
    proccorpus_mt_path,
    skipgram_hyperparams_config_path,
)


#########################################
class BatchOptimiserListener(TrainListener):
    '''
    A subclass of the training progress listener for showing progress while optimising the batch
    size.
    '''

    #########################################
    def __init__(
        self,
    ) -> None:
        '''
        Initialiser.
        '''
        self.start_time = 0.0
        self.total_duration = 0.0
        self.num_batches_processed = 0
        self.num_batches_total = 0

    #########################################
    def started_batch(
        self,
        batch_num: int,
    ) -> None:
        '''
        Listener for when a batch starts being processed.

        :param batch_num: The batch that started.
        '''
        self.start_time = timeit.default_timer()

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
        self.total_duration += timeit.default_timer() - self.start_time
        self.num_batches_processed = batch_num
        self.num_batches_total = num_batches

    #########################################
    def estimate_epoch_duration(
        self,
    ) -> float:
        '''
        After calculating the duration of a few batches, extrapolate this to estimate the duraction
        of a whole epoch.

        :return: The estimated epoch duration.
        '''
        return self.total_duration/self.num_batches_processed*self.num_batches_total


#########################################
def estimate_superbatch_bytes(
    train_data: h5py.Dataset,
    superbatch_size: int,
) -> int:
    '''
    Estimate the amount of bytes needed in memory given a superbatch size.
    This is calculated by considering that two things are stored in memory when loading
    superbatches:

        * The superbatch of training items that grows as the superbatch size increases.
        * The indexes of the possible superbatches to extract from the disk data that are shuffled
          and selected in sequence to load the next superbatch. This grows as the superbatch size
          decreases.

    Let:

        * y be the amount of bytes needed in memory
        * x be the superbatch size
        * i be the number of bytes in an integer
        * n be the number of training items in the corpus as a sequence of pairs of target/context
          token indexes

    then y = x*2*i + i*n/x

    :param train_data: The corpus of target/context token index pairs.
    :param superbatch_size: The superbatch size to use in the calculation.
    :return: The amount of bytes needed in memory.
    '''
    int_bytes = 8
    data_size = train_data.shape[0]
    superbatch_mem = superbatch_size*2*int_bytes
    indexes_mem = int_bytes*(data_size//superbatch_size)
    return superbatch_mem + indexes_mem


#########################################
def estimate_minimum_superbatch_bytes(
    train_data: h5py.Dataset,
) -> int:
    '''
    The minimum possible amount of bytes needed to have a functioning superbatch.
    This is used for validating that the given amount of memory available is not too small.

    To calculate this, we take the equation in ``estimate_superbatch_bytes`` and find its minimum
    turning point.

    Let:

        * y be the amount of bytes needed in memory
        * x be the superbatch size
        * i be the number of bytes in an integer
        * n be the number of training items in the corpus as a sequence of pairs of target/context
          token indexes

    then

        * y = x*2*i + i*n/x (equation in ``estimate_superbatch_bytes``)
        * dy/dx = 2*i - i*n/x^2
        * 0 = 2*i - i*n/x^2 (find the turning point)
        * 2*i = i*n/x^2
        * (2*i)/(i*n) = 1/x^2
        * n/2 = x^2
        * x = sqrt(n/2)

    Therefore, the minimum superbatch size is sqrt(n/2). From this we can calculate the amount of
    bytes needed for this superbatch size.

    :param train_data: The corpus of target/context token index pairs.
    :return: The minimum amount of memory needed in bytes (for using superbatches).
    '''
    min_superbatch_size = int(math.sqrt(train_data.shape[0]/2))
    return estimate_superbatch_bytes(train_data, min_superbatch_size)


#########################################
def estimate_superbatch_size(
    train_data: h5py.Dataset,
    bytes_available: int,
) -> int:
    '''
    Estimate the largest superbatch size that can be used given the amount of memory available.
    This is done by inverting the equation in ``estimate_superbatch_bytes`` by making the superbatch
    size the subject of the formula.

    Let:

        * y be the amount of bytes needed in memory
        * x be the superbatch size
        * i be the number of bytes in an integer
        * n be the number of training items in the corpus as a sequence of pairs of target/context
          token indexes

    then

        * y = x*2*i + i*n/x (from ``estimate_superbatch_bytes``)
        * -2*i*x^2 + y*x - i*n = 0 (multiplied both sides by x and made equation equal to 0)
        * x = (-y +- sqrt(y^2 - 4(-2*i)(-i*n)))/(2(-2*i)) (use the quadratic formula to solve for x)
        * x = (y -+ sqrt(y^2 - 8*i^2*n))/(4*i)

    :param train_data: The corpus of target/context token index pairs.
    :param bytes_available: The amount of bytes available for use with superbatches in memory.
    :return: The largest superbatch size that can be used given the amount of memory available.
    '''
    int_bytes = 8
    data_size = train_data.shape[0]
    return int(
        (bytes_available + math.sqrt(bytes_available**2 - 8*(int_bytes**2)*data_size))
        /
        (4*int_bytes)
    )


#########################################
def optimise_skipgram_batch_size(
    memory_for_superbatch: int,
) -> None:
    '''
    Search for the largest batch size that can be used for training the skipgram model.
    The process is as follows:

        #. Determine the largest possible superbatch size given the amount of memory available.
        #. Repeatedly double the batch size and train the model for 3 batches until an out of
           memory exception is raised. Each time, the estimated duration of a full epoch is
           recorded.
        #. Pick the fastest batch size (usually the largest) and take its two neighbouring batch
           sizes as boundaries for a more fine grained search.
        #. Perform a line search over these two boundaries by measuring the estimated durations of a
           full epoch using batch sizes over 20 equally spaced sizes between the boundaries
           (20 was arbitrarily chosen).
        #. Return the fastest batch size.

    :param memory_for_superbatch: The amount of memory in bytes that is available for superbatches.
    '''
    data_set = load_synonym_data_set()
    with open(vocab_mt_path, 'r', encoding='utf-8') as f:
        vocab_mt = f.read().strip().split('\n')
    proccorp = h5py.File(proccorpus_mt_path, 'r')
    with open(freqs_mt_path, 'r', encoding='utf-8') as f:
        neg_token_sampler = NegativeTokenSampler(json.load(f))
    with open(skipgram_hyperparams_config_path, 'r', encoding='utf-8') as f:
        hyperparams = json.load(f)

    train_data = proccorp['radius_'+str(max(hyperparams['radius']))]

    print(
        'Determining largest possible superbatch size given'
        f' {memory_for_superbatch/1e9:.2f}GB of available main memory.'
    )
    min_memory_needed = estimate_minimum_superbatch_bytes(train_data)
    if memory_for_superbatch < min_memory_needed:
        print(f'Memory set for superbatch too small. Need at least {min_memory_needed/1e9:.9f}GB.')
        return
    superbatch_size = estimate_superbatch_size(train_data, memory_for_superbatch)
    print(f'   {superbatch_size}')

    print('Optimising batch size for device', hyperparams['device'])

    def try_batch_size(
        superbatch_size: int,
        batch_size: int,
    ) -> float:
        '''
        '''
        batch_optimiser_listener = BatchOptimiserListener()
        train_skipgram_model(
            vocab_size=len(vocab_mt),
            embedding_size=hyperparams['embedding_size'],
            init_stddev=hyperparams['init_stddev'][0],
            learning_rate=hyperparams['learning_rate'][0],
            max_epochs=1,
            train_data=train_data,
            negative_token_sampler=neg_token_sampler,
            negative_sample_ratio=0, # Full batch size is obtained when this is 0.
            val_data=data_set.val,
            superbatch_size=superbatch_size,
            batch_size=batch_size,
            patience=hyperparams['patience'],
            device=hyperparams['device'],
            seed=hyperparams['seed'][0],
            listener=batch_optimiser_listener,
            stop_at_batch=3,
        )
        predicted_epoch_duration = batch_optimiser_listener.estimate_epoch_duration()/60/60
        return predicted_epoch_duration

    batch_size = 1
    results: list[tuple[int, float]] = []
    while True:
        print(f'- now trying batch size of {batch_size}')
        try:
            if hyperparams['device'].startswith('cuda'):
                torch.cuda.empty_cache()
            predicted_epoch_duration = try_batch_size(superbatch_size, batch_size)
            results.append((batch_size, predicted_epoch_duration))
        except torch.OutOfMemoryError:
            print('   batch size is too big, optimising fastest found batch size')
            break
        print(f'   {predicted_epoch_duration:.2f} hours per epoch')
        batch_size *= 2

    fastest_batch_size_i = min(range(len(results)), key=lambda i: results[i][1])
    lower_batch_size = results[max(fastest_batch_size_i - 1, 0)][0]
    if fastest_batch_size_i == len(results) - 1:
        upper_batch_size = results[-1][0]*2
    else:
        upper_batch_size = results[fastest_batch_size_i + 1][0]
    step = max((upper_batch_size - lower_batch_size)//21, 1) # 20 equally spaced steps.
    (best_batch_size, best_duration) = results[fastest_batch_size_i]
    for batch_size in range(lower_batch_size + step, upper_batch_size - step + 1, step):
        print(f'- now trying batch size of {batch_size}')
        if hyperparams['device'].startswith('cuda'):
            torch.cuda.empty_cache()
        try:
            predicted_epoch_duration = try_batch_size(superbatch_size, batch_size)
            print(f'   {predicted_epoch_duration:.2f} hours per epoch')
            if best_batch_size == 0 or predicted_epoch_duration < best_duration:
                best_batch_size = batch_size
                best_duration = predicted_epoch_duration
        except torch.OutOfMemoryError:
            print('   batch size is too big, ending search')
            break

    print()
    print(f'Best batch size:      {best_batch_size}')
    print(f'   Predicted speed: {best_duration:.2f} hours per epoch')
    print(f'Best superbatch size: {superbatch_size}')
    print()
    print(f'Please update the hyperparameters in {skipgram_hyperparams_config_path}.')
