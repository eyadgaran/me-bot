from mbot.modeling.models.preprocessor import Preprocessor
from mbot.modeling.models.seq2seq import Seq2Seq

__author__ = 'Elisha Yadgaran'

PREPROCESSOR = Preprocessor()
PREPROCESSOR.load()
MODEL = Seq2Seq(PREPROCESSOR)
MODEL.load()


class ChatService(object):
    def respond_to_phrase(self, input_phrase):
        response = MODEL.predict(input_phrase)

        return response
