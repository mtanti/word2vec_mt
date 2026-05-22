'''
Classes and functions for training the skipgram model.
'''

from typing import Optional
import os
import tempfile
import random
import math
import numpy as np
import torch
import h5py
from word2vec_mt.model.trainer.common import TrainListener
from word2vec_mt.model.model import SkipgramModel
from word2vec_mt.model.data import DataSplit
from word2vec_mt.model.evaluate import synonym_mean_average_precision


#########################################
class SkipgramDataSet(torch.utils.data.Dataset):
    '''
    A PyTorch data set for storing the skipgram model's training set.
    '''

    #########################################
    def __init__(
        self,
        target_indexes: np.ndarray,
        context_indexes: np.ndarray,
    ) -> None:
        '''
        Initialiser.

        :param target_indexes: The vector of token indexes of the target tokens.
        :param context_indexes: The vector of token indexes of the context tokens.
        '''
        super().__init__()
        self.target_indexes = target_indexes
        self.context_indexes = context_indexes

    #########################################
    def __len__(
        self,
    ) -> int:
        '''
        Get the number of training items.

        :return: The number of training items.
        '''
        return len(self.target_indexes)

    #########################################
    def __getitem__(
        self,
        index: int,
    ) -> dict[str, np.ndarray]:
        '''
        Get a training item.

        :param index: The training item index.
        :return: A dictionary mapping 'targets' to the input token indexes vector and 'contexts' to
            the target token indexes vector.
        '''
        return {
            'targets': self.target_indexes[index],
            'contexts': self.context_indexes[index],
        }


#########################################
class NegativeTokenSampler:
    '''
    A random token sampler that samples negative context tokens for training the skipgram model.
    The negative tokens are sampled with a probability equal to the token's corpus frequency
    proportion but where the proportions are smoothened to be more uniform.
    '''

    #########################################
    def __init__(
        self,
        token_index_freq_pairs: list[tuple[int, int]],
        alpha: float = 0.75
    ) -> None:
        '''
        Initialiser.

        :param token_index_freq_pairs: A list of (token index, token frequency) pairs.
        :param alpha: The amount to smoothen the token sample probabilities where 1.0 doesn't
            smoothen anything and 0.0 transforms the proportions into a completely uniform
            distribution.
            It is set to 0.75 as a default value which is supposed to work well in general.
        '''
        self.token_indexes = [token_index for (token_index, _) in token_index_freq_pairs]

        weighted_freqs = [freq**alpha for (_, freq) in token_index_freq_pairs]
        total_weighted_freqs = sum(weighted_freqs)
        weighted_probs = [
            weighted_freq/total_weighted_freqs
            for weighted_freq in weighted_freqs
        ]

        self.cum_weighted_probs = [weighted_probs[0]] # Cumulative weighted probabilities.
        for p in weighted_probs[1:]:
            self.cum_weighted_probs.append(self.cum_weighted_probs[-1] + p)

    #########################################
    def sample(
        self,
        target: int,
        context: int,
        num_negative_tokens: int,
        rng: Optional[random.Random] = None,
    ) -> list[int]:
        '''
        Sample a number of negative context tokens, making sure that the current target and context
        tokens are not included in the sample.

        :param target: The current target token index.
        :param context: The current context token index.
        :param num_negative_tokens: The number of tokens to sample.
        :param rng: The random number generator to use. If None then an unseeded RNG is used.
        :return: A list of negative token indexes.
        '''
        if rng is None:
            rng = random.Random()

        result = rng.choices(
            self.token_indexes,
            cum_weights=self.cum_weighted_probs,
            k=num_negative_tokens,
        )
        result = [i for i in result if i not in [target, context]]
        for _ in range(num_negative_tokens - len(result)):
            while True:
                i = rng.choices(
                    self.token_indexes,
                    cum_weights=self.cum_weighted_probs,
                )[0]
                if i not in [target, context]:
                    result.append(i)
                    break
        return result

    #########################################
    def add_negatives_to_batch(
        self,
        targets_batch: np.ndarray,
        contexts_batch: np.ndarray,
        negative_sample_ratio: int,
        rng: Optional[random.Random] = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        '''
        Add a sample of negative context tokens to a batch of positive target-context pairs.

        :param targets_batch: The batch of target token indexes.
        :param contexts_batch: The batch of context token indexes.
        :param negative_sample_ratio: The number of negative context tokens to add for each positive
            context token, paired with the same target token as the positive context token.
        :param rng: The random number generator to use. If None then an unseeded RNG is used.
        :return: 3 vectors are returned - the target token indexes, the context token indexes, a
            vector of 1s and -1s indicating if the context token is positive (from the corpus) or
            negative (randomly sampled).
        '''
        new_targets_batch = []
        new_contexts_batch = []
        weighting_batch = []

        for (t, c) in zip(targets_batch.tolist(), contexts_batch.tolist()):
            new_targets_batch.append(t)
            new_contexts_batch.append(c)
            weighting_batch.append(1)
            for neg_c in self.sample(t, c, negative_sample_ratio, rng):
                new_targets_batch.append(t)
                new_contexts_batch.append(neg_c)
                weighting_batch.append(-1)

        return (
            np.array(new_targets_batch, np.int64),
            np.array(new_contexts_batch, np.int64),
            np.array(weighting_batch, np.float32),
        )


#########################################
def train_skipgram_model(
    vocab_size: int,
    embedding_size: int,
    init_stddev: float,
    learning_rate: float,
    max_epochs: int,
    train_data: h5py.Dataset,
    negative_token_sampler: NegativeTokenSampler,
    negative_sample_ratio: int,
    val_data: DataSplit,
    superbatch_size: int,
    batch_size: int,
    patience: int,
    device: str,
    seed: int,
    listener: TrainListener = TrainListener(),
    stop_at_batch: Optional[int] = None,
) -> SkipgramModel:
    '''
    Train the linear model. Since the corpus was processed into a list of target/context token index
    pairs, and since this is too big to store all into memory and shuffle for batching, the corpus
    is loaded into memory one 'superbatch' at a time, each superbatch is shuffled and batches are
    extracted from it. The superbatches themselves are loaded in random order as well, but the
    corpus is segmented into sequential superbatches which are not changed.

    :param vocab_size: The vocabulary size.
    :param embedding_size: The vector size for the tokens.
    :param init_stddev: The standard deviation to use when initialising the parameters using a
        random normal distribution.
    :param learning_rate: The gradient descent learning rate to use in the optimiser.
    :param max_epochs: The maximum number of epochs to train for.
    :param train_data: The target-context token index pairs extracted from the corpus.
    :param negative_token_sampler: The negative token sampler object to use.
    :param negative_sample_ratio: The number of negative context tokens to sample for each positive
        context token.
    :param val_data: The validation set to perform early stopping on.
    :param superbatch_size: The number of training items to load into memory from disk as a
        SkipgramDataSet object in order to take batches from for training.
        This is reduced to the nearest multiple of batch_size.
    :param batch_size: The maximum number of data items to pass to the model at a time.
    :param patience: The number epochs with a less than best mean average precision to allow before
        stopping training early.
    :param device: The PyTorch device to move the data and model (e.g. 'cpu' or 'cuda' or 'cuda:1').
    :param seed: The random seed to use.
    :param listener: The listener object to react to training events.
    :param stop_at_batch: The maximum number of batches to train on before ending training.
        Used for testing.
    :return: The trained model.
    '''
    # batch_size = pos_batch_size*(1 + neg_sample_ratio)
    # pos_batch_size = batch_size/(1 + neg_sample_ratio)
    if batch_size < 1 + negative_sample_ratio:
        raise ValueError(
            f'batch_size ({batch_size}) must be at least {negative_sample_ratio + 1} when'
            f' negative_sample_ratio is ({negative_sample_ratio}).'
        )
    positive_batch_size = int(batch_size/(1 + negative_sample_ratio))

    # Make superbatch_size a multiple of positive_batch_size.
    if superbatch_size < positive_batch_size:
        raise ValueError(
            f'superbatch_size {superbatch_size} must be at least {positive_batch_size} when'
            f' batch_size is {batch_size} and negative_sample_ratio is {negative_sample_ratio}.'
        )
    superbatch_size = superbatch_size - superbatch_size%positive_batch_size

    seed_rng = random.Random(seed)
    model_seed = seed_rng.randrange(0, 0xFF_FF_FF_FF)
    superbatch_rng = random.Random(seed_rng.randrange(0, 0xFF_FF_FF_FF))
    batch_seed_rng = random.Random(seed_rng.randrange(0, 0xFF_FF_FF_FF))
    negative_sampling_rng = random.Random(seed_rng.randrange(0, 0xFF_FF_FF_FF))

    model = SkipgramModel(vocab_size, embedding_size, init_stddev)
    model.initialise(model_seed)
    model.to(device)

    optimiser = torch.optim.Adam(model.parameters(), lr=learning_rate)
    error_func = torch.nn.BCEWithLogitsLoss()
    best_val_map = 0.0
    num_bad_epochs = 0
    num_batches = math.ceil(len(train_data)/positive_batch_size)
    with tempfile.TemporaryDirectory() as tmp_dir:
        listener.started_training()

        for epoch_num in range(1, max_epochs+1):
            listener.started_epoch(epoch_num, num_batches)

            batch_num = 0
            superbatch_indexes = list(range(0, train_data.shape[0], superbatch_size))
            superbatch_rng.shuffle(superbatch_indexes)
            for superbatch_index in superbatch_indexes:
                generator = torch.Generator()
                generator.manual_seed(batch_seed_rng.randrange(0, 0xFF_FF_FF_FF))
                data_loader = torch.utils.data.DataLoader(
                    SkipgramDataSet(
                        target_indexes=train_data[
                            superbatch_index:superbatch_index+superbatch_size,
                            0,
                        ],
                        context_indexes=train_data[
                            superbatch_index:superbatch_index+superbatch_size,
                            1,
                        ],
                    ),
                    positive_batch_size, shuffle=True, generator=generator,
                )

                model.train()
                for batch in data_loader:
                    batch_num += 1
                    listener.started_batch(batch_num)

                    (
                        batch_targets_np,
                        batch_contexts_np,
                        batch_weightings_np
                    ) = negative_token_sampler.add_negatives_to_batch(
                        batch['targets'].numpy(),
                        batch['contexts'].numpy(),
                        negative_sample_ratio,
                        negative_sampling_rng,
                    )
                    batch_targets = torch.tensor(batch_targets_np, device=device)
                    batch_contexts = torch.tensor(batch_contexts_np, device=device)
                    batch_weightings = torch.tensor(batch_weightings_np, device=device)

                    optimiser.zero_grad()
                    logits = model(batch_targets, batch_contexts)
                    train_error = error_func(logits*batch_weightings, torch.ones_like(logits))
                    train_error.backward()
                    optimiser.step()

                    listener.ended_batch(
                        batch_num,
                        num_batches,
                        train_error.detach().cpu().tolist(),
                    )

                    if stop_at_batch is not None and batch_num == stop_at_batch:
                        break

                if stop_at_batch is not None and batch_num == stop_at_batch:
                    break

            model.eval()
            val_map = synonym_mean_average_precision(model.get_embeddings(), val_data)
            if val_map > best_val_map:
                torch.save(model.state_dict(), os.path.join(tmp_dir, 'model.pt'))
                best_val_map = val_map
                num_bad_epochs = 0
            else:
                num_bad_epochs += 1
                if num_bad_epochs == patience:
                    listener.ended_epoch(epoch_num, val_map, num_bad_epochs == 0, num_bad_epochs)
                    break

            listener.ended_epoch(epoch_num, val_map, num_bad_epochs == 0, num_bad_epochs)

            if stop_at_batch:
                break

        model.load_state_dict(torch.load(os.path.join(tmp_dir, 'model.pt'), weights_only=True))

        listener.ended_training()

        return model
