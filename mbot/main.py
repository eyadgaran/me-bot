from flask import Flask
from mbot.instrumentation.endpoints import chat


__author__ = 'Elisha Yadgaran'


app = Flask(__name__)


if __name__ == '__main__':
    app.add_url_rule('/chat', 'chat', chat, methods=['POST'])
    app.run(host='0.0.0.0', port=2018)
