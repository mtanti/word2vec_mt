'''
Classes and functions for training the linear model.
'''

import os
import tempfile
import math
import numpy as np
import torch
from word2vec_mt.model.trainer.common import TrainListener
from word2vec_mt.model.model import LinearModel
from word2vec_mt.model.data import DataSplit, FlatDataSplit
from word2vec_mt.model.evaluate import translation_mean_average_precision


#########################################
class LinearDataSet(torch.utils.data.Dataset):
    '''
    A PyTorch data set for storing the linear model's training set.
    '''

    #########################################
    def __init__(
        self,
        input_embeddings: np.ndarray,
        target_embeddings: np.ndarray,
    ) -> None:
        '''
        Initialiser.

        :param input_embeddings: A matrix of input (Maltese) row-vectors to translate.
        :param target_embeddings: A matrix of target (English) row-vectors to translate into.
        '''
        super().__init__()
        self.input_embeddings = input_embeddings
        self.target_embeddings = target_embeddings

    #########################################
    def __len__(
        self,
    ) -> int:
        '''
        Get the number of training items.

        :return: The number of training items.
        '''
        return len(self.input_embeddings)

    #########################################
    def __getitem__(
        self,
        index: int,
    ) -> dict[str, np.ndarray]:
        '''
        Get a training item.

        :param index: The training item index.
        :return: A dictionary mapping 'input' to the input vector and 'target' to the target vector.
        '''
        return {
            'input': self.input_embeddings[index, :],
            'target': self.target_embeddings[index, :],
        }


#########################################
def train_linear_model(
    source_embedding_size: int,
    target_embedding_size: int,
    init_stddev: float,
    use_bias: bool,
    weight_decay: float,
    learning_rate: float,
    max_epochs: int,
    source_embedding_matrix: np.ndarray,
    target_embedding_matrix: np.ndarray,
    train_data: FlatDataSplit,
    val_data: DataSplit,
    batch_size: int,
    patience: int,
    device: str,
    seed: int,
    listener: TrainListener = TrainListener(),
) -> LinearModel:
    '''
    Train the linear model.

    :param source_embedding_size: The vector size for the source tokens (Maltese).
    :param target_embedding_size: The vector size for the target tokens (English).
    :param init_stddev: The standard deviation to use when initialising the parameters using a
        random normal distribution.
    :param use_bias: Whether to use a bias in the linear model.
    :param weight_decay: The weight decay to use in the optimiser to make weights smaller.
    :param learning_rate: The gradient descent learning rate to use in the optimiser.
    :param max_epochs: The maximum number of epochs to train for.
    :param source_embedding_matrix: The source (Maltese) word2vec token vectors matrix to translate.
    :param target_embedding_matrix: The target (English) word2vec token vectors matrix to be
        translated.
    :param train_data: The training set of Maltese tokens and expected similar English tokens.
    :param val_data: The validation set to perform early stopping on.
    :param batch_size: The maximum number of data items to pass to the model at a time.
    :param patience: The number epochs with a less than best mean average precision to allow before
        stopping training early.
    :param device: The PyTorch device to move the data and model (e.g. 'cpu' or 'cuda' or 'cuda:1').
    :param seed: The random seed to use.
    :param listener: The listener object to react to training events.
    :return: The trained model.
    '''
    model = LinearModel(source_embedding_size, target_embedding_size, init_stddev, use_bias)
    model.initialise(seed)
    model.to(device)

    optimiser = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    error_func = torch.nn.MSELoss()
    generator = torch.Generator()
    generator.manual_seed(seed)
    data_loader = torch.utils.data.DataLoader(
        LinearDataSet(
            input_embeddings=source_embedding_matrix[train_data.source_token_indexes, :],
            target_embeddings=target_embedding_matrix[train_data.similar_token_indexes, :],
        ),
        batch_size, shuffle=True, generator=generator,
    )
    best_val_map = 0.0
    num_bad_epochs = 0
    num_batches = math.ceil(len(train_data.source_token_indexes)/batch_size)
    with tempfile.TemporaryDirectory() as tmp_dir:
        listener.started_training()

        for epoch_num in range(1, max_epochs+1):
            listener.started_epoch(epoch_num, num_batches)

            model.train()
            for (batch_num, batch) in enumerate(data_loader, start=1):
                listener.started_batch(batch_num)

                batch_input = batch['input'].to(device)
                batch_target = batch['target'].to(device)
                optimiser.zero_grad()
                outputs = model(batch_input)
                train_error = error_func(outputs, batch_target)
                train_error.backward()
                optimiser.step()

                listener.ended_batch(
                    batch_num,
                    num_batches,
                    train_error.detach().cpu().tolist(),
                )

            model.eval()
            with torch.no_grad():
                new_source_embedding_matrix = model(
                    torch.from_numpy(source_embedding_matrix).to(device)
                ).cpu().numpy()
            val_map = translation_mean_average_precision(
                new_source_embedding_matrix,
                target_embedding_matrix,
                val_data,
            )
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

        model.load_state_dict(torch.load(os.path.join(tmp_dir, 'model.pt'), weights_only=True))

        listener.ended_training()

        return model
