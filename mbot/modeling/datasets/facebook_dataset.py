from mbot.modeling.datasets.dataset import Dataset
from mbot.modeling.datasets.dataset_utils import Message, Stack, MessageStack
import fbchat_archive_parser as fbcap
from subprocess import call
from os.path import join
from mbot.configuration.system_path import RAW_DATA_DIRECTORY, PARSED_DATA_DIRECTORY
from mbot.configuration.secrets import FACEBOOK_USERNAME
import arrow
import pandas as pd
import re

__author__ = 'Elisha Yadgaran'

RAW_DATA_FILENAME = 'messages.htm'
PARSED_DATA_FILENAME = 'parsed_fb_messages.txt'

class FacebookDataset(Dataset):
    def load_raw_data(self):
        '''
        Uses fbchat_archive_parser to convert facebook data export into
        text file with all chats (instead of individual html threads)
        '''
        #TODO: figure out python api - defaults to bash commands
        timezones = 'CDT=-0500,CST=-0600,EDT=-0400,EST=-0500,PST=-0700,PDT=-0600,CET=+0100'
        command = 'fbcap -z {timezones} {raw_directory}/facebook/{raw_filename} > {parsed_directory}/{parsed_filename}'.format(
            timezones=timezones,
            raw_directory=RAW_DATA_DIRECTORY,
            raw_filename=RAW_DATA_FILENAME,
            parsed_directory=PARSED_DATA_DIRECTORY,
            parsed_filename=PARSED_DATA_FILENAME
        )
        call(command, shell=True)

    def load_parsed_data(self):
        '''
        Read in raw text file
        '''
        with open(join(PARSED_DATA_DIRECTORY, PARSED_DATA_FILENAME)) as f:
            data = f.readlines()

        return data

    def build_dataframe(self):
        '''
        Brains of the operation
        iterates through cleansing and formatting steps before returning
        final dataframe of the form:
            TIME | INPUT | MY RESPONSE
        '''
        parsed_data = self.load_parsed_data()
        cleaned_data = self.clean_dataset(parsed_data)
        condensed_data = self.condense_by_sender(cleaned_data)
        response_data = self.condense_by_response(condensed_data)

        return pd.DataFrame(response_data, columns=['request', 'response'])

    @staticmethod
    def clean_dataset(data):
        '''
        Clean up fbcap and facebook export formatting
        '''
        # 1) Header
        for index, row in enumerate(data):
            if row[0] == '[':
                break
        data = data[index:]

        # 2) Conversation headers
        data = [row for row in data if row[:12] != 'Conversation']

        # 3) Multiline messages
        stack = Stack()
        for row in data:
            if row[0] == '[':
                stack.pop_stack()
            stack.append_stack(row)
        stack.pop_stack()
        data = stack.data

        return data

    @staticmethod
    def condense_by_sender(data):
        '''
        Squash sequential messages from the same sender
        Also converts to using Message objects
        '''
        def parse_timestamp(row):
            time_string = row[row.find('[')+1: row.find(']')]
            return arrow.get(time_string)

        def parse_sender(row):
            message_string = row[row.find(']')+1:].strip()
            return message_string[:message_string.find(':')]

        def parse_message(row):
            message_string = row[row.find(']')+1:].strip()
            return message_string[message_string.find(':')+1:].strip()

        formatted_data = [
            Message(
                parse_timestamp(row),
                parse_sender(row),
                parse_message(row)
            ) for row in data]

        stack = MessageStack()

        current_sender = formatted_data[0].sender
        for message in formatted_data:
            if message.sender != current_sender:
                stack.pop_stack()
            stack.append_stack(message)
        stack.pop_stack()
        data = stack.data

        return data

    def condense_by_response(self, data):
        '''
        Treats all other senders as the same and separates my responses
        based on their input
        '''
        response_list = []
        tmp_dict = {'received': [], 'sent': []}
        message_is_received = True
        for message in data:
            if message.sender == FACEBOOK_USERNAME:
                message_is_received = False
                tmp_dict['sent'].append(message.message)
            else:
                if message_is_received == False:
                    response_list.append(
                        (self.clean_message(' '.join(tmp_dict['received'])),
                         self.clean_message(' '.join(tmp_dict['sent']))))
                    tmp_dict = {'received': [], 'sent': []}
                    message_is_received = True
                tmp_dict['received'].append(message.message)
        response_list.append(
            (self.clean_message(' '.join(tmp_dict['received'])),
             self.clean_message(' '.join(tmp_dict['sent']))))

        return response_list

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
