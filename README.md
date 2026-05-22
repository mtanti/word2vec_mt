# word2vec_mt

A word2vec model maker for the Maltese language.

## Where to find the data

The main data files are published under MIT license here: https://zenodo.org/records/20342956

The full set of files produced (except for the text/HDF5 corpus) can be found in this Google shared folder: https://drive.google.com/drive/folders/1KcS_nico2-QrLjhCqA8iubpSkNT-74OI?usp=sharing

## How to train your own

This repo contains all the code to make word2vec embeddings yourself.
Just follow the instructions below in order.

### How to install

Start by running `create_venv.bat` (for Window) or `create_venv.sh` (for Linux).
These scripts assume that you are calling them through an Anaconda enabled command line.
A folder called `venv/` will be created with the necessary environment to run the code.

### The `bin` scripts

All the features you need to create word2vec embeddings are available in the scripts that are inside the `bin/` folder.
Make sure the you have the environment in `venv/` activate when running these scripts.

#### `bin/download_data_mt.py`

This will download and preprocess the Maltese corpus that was used to create the Maltese word2vec embeddings using a skipgram model.
If you want to use your own corpus, just create a text file `data/corpus_mt.txt` with a sentence in each line and with tokens separated by spaces.

#### `bin/extract_vocab.py`

This will extract a vocabulary from the corpus.
The vocabulary is the list of tokens that will be used in the word2vec embeddings.

#### `bin/collect_token_freqs.py`

This will collect the corpus frequencies of the tokens in the extracted vocabulary.
This is used for sampling negative context tokens.

#### `bin/preprocess_corpus_to_train_set.py`

This will preprocess the text corpus into an HDF5 file with pairs of target/context token indexes, meant to speed up the training process.

#### `bin/help_make_synonym_data_set.py`

This is used to help with creating a data set of similar tokens for evaluating the word2vec embeddings.
It is meant to check that the most similar token vectors are, in fact, from similar tokens.
There is already a completed file `output/synonyms_mt.jsonl` available (delete this if you want to make your own).

#### `bin/split_synonym_data_set.py`

This will create random data splits from the similar words data set created in the previous step.

#### `bin/tune_word2vec_batch_size.py`

Provided you have a GPU, this will look for the best batch size to use for training on your system.

#### `bin/tune_word2vec.py`

This will perform hyperparameter tuning to look for the best hyperparameters for training word2vec embeddings.
The search space is defined in `data/skipgram_hyperparams.json`.

#### `bin/train_word2vec.py`

Once the best hyperparameters are determined, this will train the actual word2vec embeddings.

#### `bin/download_data_en.py`

Everything from here onwards is for aligning the Maltese word2vec embeddings to English word2vec embeddings (for finding the most similar English words to a Maltese word).
Alignment is performed by training a linear model to transform the Maltese word2vec embeddings into similar English word2vec embeddings.
If you want to do that, this will download and process the Google News 300 word2vec.

#### `bin/help_make_translation_data_set.py`

This is used to help with creating a data set of similar English tokens to Maltese tokens for evaluating the aligned Maltese word2vec embeddings.
It is meant to check that the most similar English token vectors are, in fact, from similar English tokens.
There is already a completed file `output/translations_mten.jsonl` available (delete this if you want to make your own).

#### `bin/split_translation_data_set.py`

This will create random data splits from the similar words data set created in the previous step.

#### `bin/tune_word2vec_aligner.py`

This will perform hyperparameter tuning to look for the best hyperparameters for aligning word2vec embeddings.
The search space is defined in `data/linear_hyperparams.json`.

#### `bin/train_word2vec_aligner.py`

Once the best hyperparameters are determined, this will train the actual embeddings aligner.
