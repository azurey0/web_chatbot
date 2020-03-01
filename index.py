from flask import Flask,request,jsonify,render_template
import os
import requests
import json
import sys
# from web_chatbot.transfer_learning_conv_ai.interact import Chatbot
from web_chatbot.information_retrieval.get_answer import IRbot

app = Flask(__name__)
# chat_bot = Chatbot()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/send_message', methods=['POST'])
def send_message():
    message = request.form['message']
    res = IRbot.chat(message)
    # res = chat_bot.chat(message)
    response_text = {"message": res}
    return jsonify(response_text)

if __name__=="__main__":
    app.run(port=80)
