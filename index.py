from flask import Flask,request,jsonify,render_template
import os
import requests
import json
import pusher
import sys
# from transfer_learning_conv_ai import interact

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/send_message', methods=['POST'])
def send_message():
    message = request.form['message']
    print(type(message))
    response_text = {"message": message}
    return jsonify(response_text)

if __name__=="__main__":
    app.run()