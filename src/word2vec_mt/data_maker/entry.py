'''
A class for an entry (a source word and its similar words).
'''

import json


#########################################
class Entry:
    '''
    Class for an entry.
    '''

    #########################################
    def __init__(
        self,
        source: str,
        similars: list[str],
    ) -> None:
        '''
        Constructor.

        :param source: The source word.
        :param similars: The similar words.
        '''
        self.source = source
        self.similars = set(similars)

    #########################################
    def to_json(
        self,
    ) -> str:
        '''
        Convert this entry to a JSON line.

        :return: The JSON line.
        '''
        return json.dumps(
            {'source': self.source, 'similars': sorted(self.similars)},
            ensure_ascii=False,
        )
