import redis
"""
# r = redis.Redis()
# r.mset({"Croatia": "Zagreb", "Bahamas": "Nassau"})
# print(type(r.get("Bahamas").decode("utf-8")))# convert redis return "byte" to "str"

import random

random.seed(444)
hats = {f"hat:{random.getrandbits(32)}": i for i in (
    {
        "color": "black",
        "price": 49.99,
        "style": "fitted",
        "quantity": 1000,
        "npurchased": 0,
    },
    {
        "color": "maroon",
        "price": 59.99,
        "style": "hipster",
        "quantity": 500,
        "npurchased": 0,
    },
    {
        "color": "green",
        "price": 99.99,
        "style": "baseball",
        "quantity": 200,
        "npurchased": 0,
    })
}

# r = redis.Redis(db=1)

# with r.pipeline() as pipe:#With a pipeline, all the commands are buffered on the client side and then sent at once
#     for h_id, hat in hats.items():
#         pipe.hmset(h_id, hat)
#     pipe.execute()
# print(r.bgsave())
# print(r.hgetall("hat:56854717"))
# print( r.keys() ) # Careful on a big DB. keys() is O(N)
# r.hincrby("hat:56854717", "quantity", -1) #reduce quantity by 1
# r.hget("hat:56854717", "quantity") # get quantity
# r.hincrby("hat:56854717", "npurchased", 1)# increase npurchased by 1


import json
import os

goal_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "dataset/chat_qa.json")
print(goal_dir)
print('openning file, please wait...')

with open(goal_dir, 'r', encoding='utf-8-sig') as f:
    data = json.load(f)

# print(data)
qa = {}
for i in range(len(data['qa'])):
    key=f"qa:{i}"
    value = {'question':data['qa'][i]['发送'],
             'answer':data['qa'][i]['接收']}
    qa[key]=value
    # print(qa)

r = redis.Redis(db=3)

with r.pipeline() as pipe:#With a pipeline, all the commands are buffered on the client side and then sent at once
    for qa_id, qa_content in qa.items():
        pipe.hmset(qa_id, qa_content)
    pipe.execute()

print(r.hget("qa:0",'answer').decode("utf-8"))
# print(e.shape)  # (67140, 768)
"""
import os
import pickle
import numpy as np
from scipy import spatial

r = redis.Redis(db=3)
goal_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "dataset/q_BallTree.pickle")

with open(goal_dir, "rb") as f:
    rawdata = f.read()
question_tree = pickle.loads(rawdata)

# embeddings = os.path.join(os.path.dirname(os.path.realpath(__file__)), "dataset/q_embedding_matrix.npy")
# e = np.load(embeddings)

from bert_serving.client import BertClient
bc = BertClient()

class IRbot:
    def chat(sentence):
        # encode input sentence
        embeddings = bc.encode(sentence.split())[0].reshape(1, -1)
        # print(embeddings.shape)
        # look for the index of similar embeddings in BallTree of questions
        _, index = question_tree.query(embeddings)
        # get answer according to the index in dataset
        # print(f'input: {sentence}','output: ', {r.hget(f"qa:{int(index[0][0])}", 'answer').decode("utf-8")})
        return r.hget(f"qa:{int(index[0][0])}", 'answer').decode("utf-8")

# if __name__ == "__main__":
#     chat('你好你好')
#     chat('今天天气真好')
