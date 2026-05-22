'''
The user input logic to manually enter a similar word data set.
'''

import random
import textwrap
from word2vec_mt.data_maker.entry import Entry
from word2vec_mt.data_maker.load_data import load_file, InvalidFileError


#########################################

NUM_SOURCE_WORD_SUGGESTIONS = 50
'''
The number of words to sample from the vocabulary to use as suggestions to the
user entering a source word.
'''


#########################################
class ControlSignal(Exception):
    '''
    Raised to indicate a user-generated signal for controlling the program.
    '''


#########################################
class EntryReady(ControlSignal):
    '''
    Raised when no more similars for a particular source word will be entered.
    '''


#########################################
class CancelEntry(ControlSignal):
    '''
    Raised when a particular source word should be cancelled and a new one used.
    '''


#########################################
def help_make_similar_word_data_set(
    source_vocab: set[str],
    similars_vocab: set[str],
    num_entries_needed: int,
    output_path: str,
    seed: int,
) -> None:
    '''
    Help make a manually constructed similar word data set.
    A similar word data set consists of source words and at least one similar
    word for each source word.
    Words from the source vocabulary are randomly sampled to provide suggestions
    to the user when entering a source word.
    The program resumes from after last completed source word entry if
    terminated midway through.

    :param source_vocab: The vocabulary to be used for the source words.
    :param similars_vocab: The vocabulary to be used for the similar words.
    :param num_entries_needed: The number of entries to input.
    :param output_path: The path to the jsonl file containing the data.
    :param seed: The seed to use for the random number generator to randomly
        suggest words from the source vocabulary.
    '''
    previously_entered_data: list[Entry] = []
    shuffled_source_words = sorted(source_vocab)
    rng = random.Random(seed)
    rng.shuffle(shuffled_source_words)
    source: str
    similars: set[str]

    # Create the output file if it doesn't exist or load it if it does.
    try:
        with open(output_path, 'x', encoding='utf-8') as f:
            pass
    except FileExistsError:
        try:
            print('Loading existing data set')
            previously_entered_data = load_file(
                source_vocab, similars_vocab, output_path,
            )
            print()
        except InvalidFileError:
            return

    # Ask user to enter new data.
    for i in range(1, num_entries_needed + 1):
        # Skip already entered entries.
        if i <= len(previously_entered_data):
            continue

        suggested_source_words = shuffled_source_words[
            i*NUM_SOURCE_WORD_SUGGESTIONS:(i+1)*NUM_SOURCE_WORD_SUGGESTIONS
        ]
        print(
            'Working on source word'
            f' #{len(previously_entered_data)+1}/{num_entries_needed}'
        )
        print('Suggested source words:')
        print(
            '    ' + '\n    '.join(
                textwrap.wrap(' '.join(suggested_source_words), 100)
            )
        )
        print()
        while True:
            source = _get_source_word(
                source_vocab, previously_entered_data,
            )
            similars = set()
            print()

            try:
                while True:
                    similar = _get_similar_word(
                        similars_vocab, {source}|similars,
                    )
                    similars.add(similar)
                    print()
            except EntryReady:
                entry = Entry(source, list(similars))
                with open(output_path, 'a', encoding='utf-8') as f:
                    print(entry.to_json(), file=f)
                    previously_entered_data.append(entry)
                print('  Saved.')
                print()
                break
            except CancelEntry:
                print('  Cancelled. Choose another source word.')
                print()

    print('  Done.')


#########################################
def _get_source_word(
    vocab: set[str],
    previously_entered_data: list[Entry],
) -> str:
    '''
    Ask the user for a source word.

    :param vocab: The vocabulary of possible words to use.
    :param previously_entered_data: The previously entered data.
    :return: The chosen source word.
    '''
    while True:
        source = input('  source: ')

        if source not in vocab:
            print('  This word is not in the vocabulary.')
        else:
            for (line_num, entry) in enumerate(previously_entered_data, 1):
                if source == entry.source or source in entry.similars:
                    print(
                        f'  This word was already used in entry {line_num}:'
                        f' {entry.source} - {" ".join(sorted(entry.similars))}'
                    )
                    break
            else:
                return source


#########################################
def _get_similar_word(
    vocab: set[str],
    words: set[str],
) -> str:
    '''
    Ask the user for a similar word of a source word.

    Will raise EntryReady if user enters the empty string and CancelEntry
    if user enters '.'.

    :param vocab: The vocabulary of possible words to use.
    :param words: The set of words consisting of the source word and its
        similar words entered up to now.
    :return: The chosen similar word.
    '''
    print(
        f'  Enter similar word #{len(words)} (or blank for done or . for'
        ' cancel).'
    )
    while True:
        similar = input('  similar: ')

        if similar == '':
            if len(words) == 1:
                print('  Must have at least one similar word.')
            else:
                raise EntryReady()
        elif similar == '.':
            raise CancelEntry()
        elif similar not in vocab:
            print('  This word is not in the vocabulary.')
        elif similar in words:
            print('  This word was already entered in the current entry.')
        else:
            return similar
