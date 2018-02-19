from abc import ABCMeta, abstractmethod

__author__ = 'Elisha Yadgaran'


class Dataset(object):
    __metaclass__ = ABCMeta

    def __init__(self, load_raw_data=False):
        if load_raw_data:
            self.load_raw_data()

        self.dataframe = self.build_dataframe()

    @abstractmethod
    def load_raw_data(self):
        '''
        Method to parse data from /data/raw folders
        into /data/parsed
        Only needs to be executed once for new data
        '''

    @abstractmethod
    def load_parsed_data(self):
        '''
        method to read in parsed data from /data/parsed
        '''

    @abstractmethod
    def build_dataframe(self):
        '''
        inheriting classes must load and
        return the formatted dataframe
        '''
