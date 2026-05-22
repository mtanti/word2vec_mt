corpus_preprocessor
===================

Convert the Maltese corpus into an array of word-in-context / context word
pairs represented as integer indexes.

Given a sentence, a word will be surrounded by other words and these other words
define the context of the first word.
The first word is called the word-in-context and the other words are called
context words.
A word is considered a context word with respect to a particular word-in-context
when it is close ot the word-in-context, usually within a context window
consisting of all words within a fixed number of words to the left or right of
the word-in-context.
The number of words distance is called the radius.

Example: Given the sentence *the dog barked at the cat* and a radius of 2, the
context words will be as follows:

.. list-table::
  :header-rows: 1

  * - word-in-context
    - context words
  * - *barked*
    - *the*, *dog*, *at*, *the*
  * - *dog*
    - *the*, *barked*, *at*
  * - *cat*
    - *at*, *the*

The skip-gram word2vec model is optimised to predict the presence of a context
word given a word-in-context.
To speed up this optimisation process, a corpus is preprocessed into a set of
integer pairs, where the first integer represents a word-in-context and the
second represents a context word (given a radius).
A word (or token) is converted into an integer depending on its position in the
vocabulary (words that are out-of-vocabulary are ignored).
This set of integer pairs is then stored in a HDF5 file for fast disk-based
retrieval.

Example: Given the sentence *the dog barked at the cat*, a radius of 2, and a
vocabulary of *barked* (0), *cat* (1), *dog* (2), *the* (3), the word-in-context
/ context word pairs will be as follows:

.. list-table::
  :header-rows: 1

  * - word-in-context token
    - word-in-context index
    - context word token
    - context word index
  * - *the*
    - 3
    - *dog*
    - 2
  * - *the*
    - 3
    - *barked*
    - 0
  * - *dog*
    - 2
    - *the*
    - 3
  * - *dog*
    - 2
    - *barked*
    - 0
  * - *barked*
    - 0
    - *the*
    - 3
  * - *barked*
    - 0
    - *dog*
    - 2
  * - *barked*
    - 0
    - *the*
    - 3
  * - *the*
    - 3
    - *barked*
    - 0
  * - *the*
    - 3
    - *cat*
    - 1
  * - *cat*
    - 1
    - *the*
    - 3

Since a context word A of a word-in-context B implies that B is also a context
word of word-in-context A (the relationship is symmetric), we can speed up the
extraction of these word pairs by only considering one side of the context
window and then duplicating and swapping the words-in-context with the context
words.

Example: Consider the above list of word-in-context / context word pairs.
Extracting only the right side context words of each word=in-context will give
us the following:

.. list-table::
  :header-rows: 1

  * - word-in-context token
    - context word token
  * - *the*
    - *dog*
  * - *the*
    - *barked*
  * - *dog*
    - *barked*
  * - *barked*
    - *the*
  * - *the*
    - *cat*

We can then fill in the missing information by duplicating the rows of the table
and swapping the words-in-context with the context words:

.. list-table::
  :header-rows: 1

  * - word-in-context token 1
    - context word token 1
    - word-in-context token 2
    - context word token 2
  * - *the*
    - *dog*
    - *dog*
    - *the*
  * - *the*
    - *barked*
    - *barked*
    - *the*
  * - *dog*
    - *barked*
    - *barked*
    - *dog*
  * - *barked*
    - *the*
    - *the*
    - *barked*
  * - *the*
    - *cat*
    - *cat*
    - *the*

The order of the pairs does not really matter because they are shuffled during
model optimisation.

.. toctree::
    :maxdepth: 1

    corpus_preprocessor/buffer.rst
    corpus_preprocessor/corpus_to_train_set.rst
