'''
Endpoint Module
'''


from flask import request, jsonify
from service import ChatService


def chat():
    '''
    Use this endpoint for form post requests
    '''
    input_phrase = request.form.get('input')
    service = ChatService()
    response = service.respond_to_phrase(input_phrase)

    return jsonify(response), 200
