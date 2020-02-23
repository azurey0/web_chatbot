from flask import Flask,request,jsonify,render_template
import os
import requests
import json
import pusher
import sys
from web_chatbot.transfer_learning_conv_ai.interact import act
from web_chatbot.transfer_learning_conv_ai.test22 import test22

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/send_message', methods=['POST'])
def send_message():
    message = request.form['message']
#     print(type(message))
#     print(test23(message)) >>correct
#     print(act(message)) >>error
    response_text = {"message": message}
    return jsonify(response_text)

if __name__=="__main__":
    app.run()
