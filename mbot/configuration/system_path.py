'''
Module to create absolute references for filesystem
persisted objects
'''

import os

__author__ = 'Elisha Yadgaran'

CONFIGURATION_DIRECTORY = os.path.dirname(os.path.abspath(__file__))
PACKAGE_DIRECTORY = os.path.dirname(CONFIGURATION_DIRECTORY)
REPO_DIRECTORY = os.path.dirname(PACKAGE_DIRECTORY)

DATA_DIRECTORY = os.path.join(REPO_DIRECTORY, 'data')
RAW_DATA_DIRECTORY = os.path.join(DATA_DIRECTORY, 'raw')
PARSED_DATA_DIRECTORY = os.path.join(DATA_DIRECTORY, 'parsed')
MODEL_DIRECTORY = os.path.join(DATA_DIRECTORY, 'models')
EMBEDDING_DIRECTORY = os.path.join(DATA_DIRECTORY, 'embeddings')
DATABASE_DIRECTORY = os.path.join(DATA_DIRECTORY, 'db')

LOG_DIRECTORY = os.path.join(REPO_DIRECTORY, 'logs')

for directory in [LOG_DIRECTORY, DATA_DIRECTORY, RAW_DATA_DIRECTORY, PARSED_DATA_DIRECTORY,
                  MODEL_DIRECTORY, EMBEDDING_DIRECTORY, DATABASE_DIRECTORY]:
    if not os.path.exists(directory):
        os.mkdir(directory)
