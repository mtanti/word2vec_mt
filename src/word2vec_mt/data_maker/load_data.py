'''
Load a similar word data set file and validate it.
'''

import json
from word2vec_mt.data_maker.entry import Entry


#########################################
class InvalidFileError(Exception):
    '''
    Raised when a similar words data set file is invalid.
    '''


#########################################
def load_file(
    source_vocab: set[str],
    similars_vocab: set[str],
    path: str,
) -> list[Entry]:
    '''
    Load a similar word data set file and validate it.

    :param source_vocab: The vocabulary to be used for the source words.
    :param similars_vocab: The vocabulary to be used for the similar words.
    :param path: The path to the jsonl file containing the data.
    :return: The loaded data.
    '''
    source: str
    similars: list[str]
    data_set: list[Entry] = []
    with open(path, 'r', encoding='utf-8') as f:
        for (line_num, line) in enumerate(f, 1):
            line = line.strip()
            try:
                doc = json.loads(line)
            except json.decoder.JSONDecodeError as ex:
                print(f'  Line {line_num}: Invalid JSON.')
                raise InvalidFileError() from ex
            if (
                not isinstance(doc, dict)
                or len(doc) != 2
                or 'source' not in doc
                or not isinstance(doc['source'], str)
                or 'similars' not in doc
                or not isinstance(doc['similars'], list)
                or not all(isinstance(x, str) for x in doc['similars'])
            ):
                print(
                    f'  Line {line_num}: JSON should be in the form of an'
                    ' object with source and similars as keys where source'
                    ' is associated with a string and similars is associated'
                    f' with a list of strings.'
                )
                raise InvalidFileError()
            source = doc['source']
            similars = doc['similars']

            if source not in source_vocab:
                print(
                    f'  Line {line_num}:'
                    f' The source word {source} is not in the vocabulary.'
                )
                raise InvalidFileError()
            for (line_num_, entry) in enumerate(data_set, 1):
                if source == entry.source or source in entry.similars:
                    print(
                        f'  Line {line_num}:'
                        f' The source word {source} was already used in entry'
                        f' {line_num_}.'
                    )
                    raise InvalidFileError()

            if len(similars) == 0:
                print(
                    f'  Line {line_num}:'
                    ' Must have at least one similar word.'
                )
                raise InvalidFileError()
            words: set[str] = {source}
            for similar in similars:
                if similar not in similars_vocab:
                    print(
                        f'  Line {line_num}:'
                        f' The similar word {similar} is not in the vocabulary.'
                    )
                    raise InvalidFileError()
                if similar in words:
                    print(
                        f'  Line {line_num}:'
                        f' The similar word {similar} is already in the entry.'
                    )
                    raise InvalidFileError()
            for i in range(1, len(similars)):
                if similars[i] < similars[i-1]:
                    print(
                        f'  Line {line_num}:'
                        f' The similar word {similars[i]} is not in sorted'
                        ' order.'
                    )
                    raise InvalidFileError()

            data_set.append(Entry(source, similars))

    return data_set
