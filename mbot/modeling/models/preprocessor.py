from sklearn.feature_extraction.text import CountVectorizer
from nltk.tokenize import TweetTokenizer
from tqdm import tqdm
import dill
import pickle
from mbot.configuration.system_path import MODEL_DIRECTORY
import os
import numpy as np
from keras.utils import np_utils


__author__ = 'Elisha Yadgaran'


class Preprocessor(object):
    def __init__(self, unknown_word='um', max_message_length=30,
                 min_document_freq=10):
        self.unknown_word = unknown_word
        self.max_message_length = max_message_length
        self.min_document_freq = min_document_freq

        # Tokens needed for seq2seq
        self.unknown_token = 0  # words that aren't found in the vocab
        self.pad_token = 1  # after message has finished, this fills all remaining vector positions
        self.start_token = 2  # provided to the model at position 0 for every response predicted

    @property
    def initial_response(self):
        '''
        When a request first comes in, only the start token is available.
        Subsequent output tokens are recursively fed back into the neural net
        '''
        return self.add_start_token(self.pad_token * np.ones((1, self.max_message_length)))

    @property
    def analyzer(self):
        return self.tokenizer.build_analyzer()

    def replace_names(self, dataset):
        '''
        Utility function to replace Proper Noun names with a constant
        '''
        return dataset

    def build_vocabulary(self, dataset):
        # Tokenizer
        tokenizer = TweetTokenizer(strip_handles=True, preserve_case=False, reduce_len=True)

        # Vectorize all the tokens so we have a unique map of id: word
        count_vec = CountVectorizer(tokenizer=tokenizer.tokenize, min_df=self.min_document_freq)
        count_vec.fit(tqdm(dataset))

        # Shift vocabulary to insert placeholder values
        vocab = {k: v + 2 for k, v in count_vec.vocabulary_.items()}
        vocab[self.unknown_word] = self.unknown_token
        vocab['__pad__'] = self.pad_token
        vocab['__start__'] = self.start_token

        # Save vocabulary and tokenizer to score new words
        self.tokenizer = count_vec
        self.vocab = vocab
        self.reverse_vocab = {v: k for k, v in vocab.items()}

    def index_sentence(self, sentence):
        index_list = [self.vocab.get(token, self.unknown_token) for token in self.analyzer(sentence)]
        # Pad remaining space in fixed sequence size
        index_list += [self.pad_token] * self.max_message_length

        return index_list[:self.max_message_length]

    def humanize_token_indices(self, index_list):
        return ' '.join(self.reverse_vocab[index] for index in index_list if index != self.pad_token).strip()

    def add_start_token(self, y_array):
        '''
        Prepend all phrases with the start token
        shifts indices by one so encoded labels are the predicted next word
        '''
        return np.hstack([
            self.start_token * np.ones((len(y_array), 1)),
            y_array[:, :-1],
        ])

    def encode_labels(self, y_array):
        '''
        Encode y into one-hot vectors
        '''
        classes = len(self.vocab)
        return np.array([np_utils.to_categorical(row, num_classes=classes) for row in y_array])

    def fit(self, dataset):
        replaced_dataset = self.replace_names(dataset)
        self.build_vocabulary(replaced_dataset)

    def transform(self, sentence):
        replaced_sentence = self.replace_names(sentence)
        return self.index_sentence(replaced_sentence)

    def inverse_tansform(self, *args):
        return self.humanize_token_indices(*args)

    def save(self, prefix=''):
        tokenizer_filename = 'tokenizer.pkl'
        pickle.dump(self.tokenizer, open(os.path.join(MODEL_DIRECTORY, tokenizer_filename), 'wb'))
        vocab_filename = 'vocab.pkl'
        pickle.dump(self.vocab, open(os.path.join(MODEL_DIRECTORY, vocab_filename), 'wb'))

    def load(self, prefix=''):
        tokenizer_filename = 'tokenizer.pkl'
        self.tokenizer = pickle.load(open(os.path.join(MODEL_DIRECTORY, tokenizer_filename), 'rb'))
        vocab_filename = 'vocab.pkl'
        self.vocab = pickle.load(open(os.path.join(MODEL_DIRECTORY, vocab_filename), 'rb'))
        self.reverse_vocab = {v: k for k, v in self.vocab.items()}
