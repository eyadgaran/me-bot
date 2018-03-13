from keras.models import Model, model_from_json
from keras.optimizers import Adam
from keras.layers import Dense, Input, LSTM, Dropout, Embedding, RepeatVector, concatenate, \
    TimeDistributed
from keras.callbacks import TensorBoard
import numpy as np
from sklearn.model_selection import train_test_split
from mbot.configuration.system_path import MODEL_DIRECTORY, LOG_DIRECTORY
import os
from time import time
from tqdm import tqdm

__author__ = 'Elisha Yadgaran'


class Seq2Seq(object):
    def __init__(self, preprocessor, **kwargs):
        self.preprocessor = preprocessor
        self.internal_model = self.create_model(**kwargs)
        self.compile_model(**kwargs)

    def create_model(self, embedding_size=100, context_size=100, dropout=0.3, **kwargs):
        vocab_size = len(self.preprocessor.vocab)
        max_message_length = self.preprocessor.max_message_length

        # Encoder
        encoder_input = Input(shape=(max_message_length,), dtype='int32', name='encoder_input')

        # Word embeddings
        shared_embedding = Embedding(output_dim=embedding_size, input_dim=vocab_size,
                                     input_length=max_message_length, name='embedding')
        embedded_input = shared_embedding(encoder_input)

        # No return_sequences - since the encoder here only produces a single value for the
        # input sequence provided.
        encoder_rnn = LSTM(context_size, name='encoder', dropout=dropout)
        context = RepeatVector(max_message_length)(encoder_rnn(embedded_input))

        # Decoder
        last_word_input = Input(shape=(max_message_length,), dtype='int32', name='last_word_input')
        embedded_last_word = shared_embedding(last_word_input)

        # Combines the context produced by the encoder and the last word uttered as inputs
        # to the decoder.
        decoder_input = concatenate([embedded_last_word, context], axis=2)

        # return_sequences causes LSTM to produce one output per timestep instead of one at the
        # end of the intput, which is important for sequence producing models.
        decoder_rnn = LSTM(context_size, name='decoder', return_sequences=True, dropout=dropout)
        decoder_output = decoder_rnn(decoder_input)

        # TimeDistributed allows the dense layer to be applied to each decoder output per timestep
        next_word_dense = TimeDistributed(
            Dense(int(vocab_size / 2), activation='relu'), name='next_word_dense'
        )(decoder_output)

        next_word = TimeDistributed(
            Dense(vocab_size, activation='softmax'), name='next_word_softmax'
        )(next_word_dense)

        return Model(inputs=[encoder_input, last_word_input], outputs=[next_word])

    def compile_model(self, learning_rate=0.005, **kwargs):
        optimizer = Adam(lr=learning_rate, clipvalue=5.0)
        self.internal_model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy')

    def train(self, x, y, epochs=250, batch_size=5, test_split=0.1, save=True):
        processed_x = np.stack(x.apply(self.preprocessor.transform).values)
        processed_y = np.stack(y.apply(self.preprocessor.transform).values)
        train_x, test_x, train_y, test_y = train_test_split(processed_x, processed_y, test_size=test_split, random_state=5)
        del processed_x
        del processed_y

        # Add start tokens to responses - shifts indices by one so encoded labels
        # are the predicted next word
        prefixed_train_y = self.preprocessor.add_start_token(train_y)
        prefixed_test_y = self.preprocessor.add_start_token(test_y)

        # Encode y into one-hot vectors
        # Too memory intensive so use with sparse_categorical_crossentropy
        # or use generator to encode on the fly
        # encoded_train_y = self.preprocessor.encode_labels(train_y)
        # encoded_test_y = self.preprocessor.encode_labels(test_y)

        # TensorBoard for visualization
        tensorboard = TensorBoard(log_dir="{}/{}".format(LOG_DIRECTORY, time()))

        # Manually loop through epochs to humanize convergence
        for epoch in tqdm(range(epochs)):
            self.internal_model.fit(
                [train_x, prefixed_train_y],
                np.expand_dims(train_y, -1),
                # encoded_train_y,
                epochs=1,
                batch_size=batch_size,
                callbacks=[tensorboard]
            )

            print 'Test results: {}'.format(self.internal_model.evaluate(
                [test_x, prefixed_test_y], np.expand_dims(test_y, -1))) #encoded_test_y))

            test_inputs = [
                "Hey, how's it going?",
                "what are you up to today",
                "wanna grab dinner tonight?"
            ]

            for input_string in test_inputs:
                output_string = self.predict(input_string)
                print input_string, '\n', output_string

            if save:
                self.save()

    def predict(self, sentence):
        max_message_length = self.preprocessor.max_message_length
        response = self.preprocessor.initial_response
        token_indices = np.array(self.preprocessor.transform(sentence)).reshape((1, max_message_length))

        # Iterate through response and predict next output token
        for position in range(max_message_length - 1):
            prediction = self.internal_model.predict([token_indices, response]).argmax(axis=2)[0]
            response[:, position + 1] = prediction[position]

        final_prediction = self.internal_model.predict([token_indices, response]).argmax(axis=2)[0]
        return self.preprocessor.inverse_tansform(final_prediction)

    def save(self, prefix=''):
        # serialize model to JSON
        model_filename = 'model_architecture.json'
        model_json = self.internal_model.to_json()
        with open(os.path.join(MODEL_DIRECTORY, prefix + model_filename), 'w') as json_file:
            json_file.write(model_json)

        # serialize weights to HDF5
        weights_filename = 'model_weights.h5'
        self.internal_model.save_weights(os.path.join(MODEL_DIRECTORY, prefix + weights_filename))

    def load(self, prefix=''):
        # load json and create model
        model_filename = 'model_architecture.json'
        json_file = open(os.path.join(MODEL_DIRECTORY, prefix + model_filename), 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = model_from_json(loaded_model_json)

        # load weights into new model
        weights_filename = 'model_weights.h5'
        loaded_model.load_weights(os.path.join(MODEL_DIRECTORY, prefix + weights_filename))

        self.internal_model = loaded_model

# References
# https://www.kaggle.com/elishay/twitter-basic-seq2seq
# https://machinelearningmastery.com/teacher-forcing-for-recurrent-neural-networks/
# https://towardsdatascience.com/how-to-create-data-products-that-are-magical-using-sequence-to-sequence-models-703f86a231f8
