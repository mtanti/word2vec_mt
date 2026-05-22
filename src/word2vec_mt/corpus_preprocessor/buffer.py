'''
A buffered version of HDF5 which can only be used for appending pairs of
integers (the word-in-context index and its context word index) into a data set
matrix in an HDF5 file.

Each new pair of integers is first added to a NumPy array buffer and is then
flushed into the file when full or closed.
This speeds up the insertion of data.
'''

from typing import Iterator, Optional
from contextlib import contextmanager
import numpy as np
import h5py


#########################################
class BufferedHDF5:
    '''
    The buffered version of HDF5.
    '''

    #########################################
    def __init__(
        self,
        file_path: str,
        buffer_max_size: int,
        data_set_name: str,
        num_data_items: int,
    ) -> None:
        '''
        The constructor.

        :param file_path: The path to the HDF5 file.
        :param buffer_max_size: The size of the buffer.
        :param data_set_name: The name of the data set in the HDF5 file.
        :param num_data_items: The number of rows in the HDF5 data set matrix.
        '''
        self.file: Optional[h5py.File] = None
        try:
            self.curr_index = 0
            self.file = h5py.File(file_path, 'a')
            if data_set_name not in self.file:
                self.dset = self.file.create_dataset(
                    data_set_name,
                    (num_data_items, 2),
                    dtype=np.int64,
                )
            else:
                self.dset = self.file[data_set_name]
                if (
                    self.dset.shape != (num_data_items, 2)
                    or self.dset.dtype != np.int64
                ):
                    raise ValueError(
                        'HDF5 file already has this data set and it does not'
                        ' conform to the expected shape and dtype.'
                    )

            self.buffer = np.empty((buffer_max_size, 2), np.int64)
            self.buffer_max_size = buffer_max_size
            self.buffer_curr_size = 0
        except Exception as ex:
            if self.file is not None:
                self.file.close()
            raise ex

    #########################################
    @contextmanager
    def get_handle(
        self,
    ) -> Iterator['BufferedHDF5']:
        '''
        A helper function for using the ``with`` statement to automatically close
        the file when ready.

        Example::

            with BufferedHDF5(...).get_handle() as f:
                f.append(...)
            # file is automatically closed here

        :return: A context manager to be used in a ``with`` statement.
        '''
        assert self.file is not None
        try:
            yield self
        finally:
            self.close()

    #########################################
    def append(
        self,
        word_in_context_index: int,
        context_word_index: int,
    ) -> None:
        '''
        Add a new pair of numbers into the buffer and flush the buffer into the
        HDF5 file if it is full.

        :param word_in_context_index: The index of the word-in-context to add.
        :param context_word_index: The index of the context word to add.
        '''
        assert self.file is not None
        if self.buffer_curr_size == self.buffer_max_size:
            self.dset[
                self.curr_index : self.curr_index + self.buffer_curr_size,
                :
            ] = self.buffer[:self.buffer_curr_size]
            self.curr_index += self.buffer_curr_size
            self.buffer_curr_size = 0

        self.buffer[self.buffer_curr_size, :] = [
            word_in_context_index,
            context_word_index,
        ]
        self.buffer_curr_size += 1

    #########################################
    def close(
        self,
    ) -> None:
        '''
        Flush any remaining data in the buffer into the HDF5 file and close it.
        '''
        assert self.file is not None
        self.dset[
            self.curr_index : self.curr_index + self.buffer_curr_size,
            :
        ] = self.buffer[:self.buffer_curr_size]
        self.file.close()
