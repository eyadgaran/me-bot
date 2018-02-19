'''
Module for helper classes
'''

__author__ = 'Elisha Yadgaran'

class Message(object):
    def __init__(self, timestamp, sender, message):
        self.timestamp = timestamp
        self.sender = sender
        self.message = message

    def __repr__(self):
        return '{}\t{}\t{}'.format(self.timestamp, self.sender, self.message)


class Stack(object):
    def __init__(self):
        self.data = []
        self.stack = []

    def pop_stack(self):
        if self.stack:
            self.data.append(' '.join(self.stack))
            self.stack = []

    def append_stack(self, row):
        self.stack.append(row)


class MessageStack(Stack):
    def pop_stack(self):
        if self.stack:
            first_timestamp = self.stack[0].timestamp
            first_sender = self.stack[0].sender
            message = ' '.join([i.message for i in self.stack])
            condensed_message = Message(first_timestamp, first_sender, message)
            self.data.append(condensed_message)
            self.stack = []
