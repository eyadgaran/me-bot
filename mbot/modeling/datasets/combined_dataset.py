'''
Module to combine multiple dataset classes into one dataset
'''
import pandas as pd

__author__ = 'Elisha Yadgaran'


class CombinedDataset(object):
    def __init__(self, datasets,
                 x_column='request', y_column='response'):
        self.x_column = x_column
        self.y_column = y_column
        self.dataframe = pd.concat([i.dataframe for i in datasets]).reset_index(drop=True)

    def __iter__(self):
        for index, row in self.dataframe.iterrows():
            yield ' '.join(row)

    @property
    def x(self):
        return self.dataframe[self.x_column]

    @property
    def y(self):
        return self.dataframe[self.y_column]
