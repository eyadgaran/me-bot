from abc import ABCMeta, abstractmethod

__author__ = 'Elisha Yadgaran'


class Dataset(object):
    __metaclass__ = ABCMeta

    def __init__(self, load_raw_data=False,
                 x_column='request', y_column='response'):
        self.x_column = x_column
        self.y_column = y_column

        if load_raw_data:
            self.load_raw_data()

        self.dataframe = self.build_dataframe()

    def __iter__(self):
        for index, row in self.dataframe.iterrows():
            yield ' '.join(row)

    @property
    def x(self):
        return self.dataframe[self.x_column]

    @property
    def y(self):
        return self.dataframe[self.y_column]

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
