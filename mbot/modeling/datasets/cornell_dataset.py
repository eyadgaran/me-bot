from mbot.modeling.datasets.dataset import Dataset
from mbot.modeling.datasets.dataset_utils import Stack
from os.path import join
from mbot.configuration.system_path import RAW_DATA_DIRECTORY, PARSED_DATA_DIRECTORY
import pandas as pd
import re

__author__ = 'Elisha Yadgaran'

RAW_DATA_FOLDER = 'cornell_corpus'
RAW_DATA_INDICES_FILENAME = 'movie_lines.txt'
RAW_DATA_CONVERSATION_FILENAME = 'movie_conversations.txt'
PARSED_DATA_FILENAME = 'parsed_cornell.csv'

class CornellDataset(Dataset):
    def load_raw_data(self):
        dialogue_dict = self.build_raw_dialogue_dict()
        conversation_list = self.build_conversation_list()

        dialogue_list = [[dialogue_dict[i] for i in conversation] for conversation in conversation_list]
        condensed_list = [self.condense_by_sender(i) for i in dialogue_list]

        conversation_df = self.pair_conversations(condensed_list)
        conversation_df.to_csv(join(PARSED_DATA_DIRECTORY, PARSED_DATA_FILENAME), index=False)

    def load_parsed_data(self):
        '''
        Read in parsed csv
        '''
        data = pd.read_csv(join(PARSED_DATA_DIRECTORY, PARSED_DATA_FILENAME))

        return data

    def build_dataframe(self):
        '''
        Brains of the operation
        iterates through cleansing and formatting steps before returning
        final dataframe of the form:
            INPUT | MY RESPONSE
        '''
        parsed_data = self.load_parsed_data()
        non_null_data = parsed_data.dropna(how='any')
        cleaned_data = non_null_data.applymap(self.clean_message)

        return cleaned_data

    @staticmethod
    def build_raw_dialogue_dict():
        '''
        Corpus is broken up by dialogue indices
        in movie_lines.txt
        '''
        dialogue_dict = {}
        with open(join(join(RAW_DATA_DIRECTORY, RAW_DATA_FOLDER), RAW_DATA_INDICES_FILENAME)) as f:
            data = [i.split('+++$+++') for i in f.readlines()]

        # Package up line and speaker (to condense conversations later)
        for line in data:
            dialogue_dict[line[0].strip()] = (line[1].strip(), line[4].strip())

        return dialogue_dict

    @staticmethod
    def build_conversation_list():
        '''
        Conversations are grouped in movie_conversations.txt
        '''
        with open(join(join(RAW_DATA_DIRECTORY, RAW_DATA_FOLDER), RAW_DATA_CONVERSATION_FILENAME)) as f:
            data = [i.split('+++$+++') for i in f.readlines()]

        # List of lists with dialogue indices
        conversation_list = [eval(i[3].strip()) for i in data]

        return conversation_list

    @staticmethod
    def condense_by_sender(data):
        '''
        Squash sequential messages from the same sender
        '''
        stack = Stack()

        current_sender = data[0][0]
        for message in data:
            if message[0] != current_sender:
                current_sender = message[0]
                stack.pop_stack()
            stack.append_stack(message[1])
        stack.pop_stack()
        data = stack.data

        return data

    def pair_conversations(self, conversation_list):
        '''
        Pair up conversations as request: response
        '''
        def _pair(conversation):
            return [(j, conversation[i+1]) for i, j in enumerate(conversation[:-1])]

        conversations = []
        [conversations.extend(_pair(i)) for i in conversation_list]
        return pd.DataFrame(conversations, columns=[self.x_column, self.y_column])

    @staticmethod
    def clean_message(message):
        '''
        Formatting and cleansing
        '''
    	# Remove control characters within message
    	message = message.replace('\n',' ').lower()
        message = message.replace('\t', ' ')
    	# Deal with some weird tokens
    	message = message.replace("\xc2\xa0", "")
    	# Remove punctuation
    	message = re.sub('([.,!?])','', message)
    	# Remove multiple spaces in message
    	message = re.sub(' +',' ', message)

        return message
