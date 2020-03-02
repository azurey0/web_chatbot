# import torch
# from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM
#-*- coding : utf-8-*-
# coding:unicode_escape
import json
# OPTIONAL: if you want to have more information on what's happening, activate the logger as follows
import logging
logging.basicConfig(level=logging.INFO)
import os
data_dir = os.path.dirname(os.path.realpath(__file__))

# def get_sentence_embedding(sentence):
#     # Load pre-trained model tokenizer (vocabulary)
#     tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
#
#     marked_text = "[CLS] " + sentence + " [SEP]"
#     # Tokenize our sentence with the BERT tokenizer.
#     tokenized_text = tokenizer.tokenize(marked_text)
#
#     # Map the token strings to their vocabulary indeces.
#     indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
#
#     # Mark each of the 22 tokens as belonging to sentence "1".
#     segments_ids = [1] * len(tokenized_text)
#
#     # Convert inputs to PyTorch tensors
#     tokens_tensor = torch.tensor([indexed_tokens])
#     segments_tensors = torch.tensor([segments_ids])
#
#     # Load pre-trained model (weights)
#     model = BertModel.from_pretrained('bert-base-uncased')
#
#     # Put the model in "evaluation" mode, meaning feed-forward operation.
#     model.eval()
#
#     # Predict hidden states features for each layer
#     with torch.no_grad():
#         encoded_layers, _ = model(tokens_tensor, segments_tensors)
#
#     # print ("Number of layers:", len(encoded_layers))
#     # layer_i = 0
#     # print ("Number of batches:", len(encoded_layers[layer_i]))
#     # batch_i = 0
#     # print ("Number of tokens:", len(encoded_layers[layer_i][batch_i]))
#     # token_i = 0
#     # print ("Number of hidden units:", len(encoded_layers[layer_i][batch_i][token_i]))
#
#     # Concatenate the tensors for all layers. We use `stack` here to. Could use different methods in later versions!
#     # create a new dimension in the tensor.
#     token_embeddings = torch.stack(encoded_layers, dim=0)
#     token_embeddings.size()
#
#     # Remove dimension 1, the "batches".
#     token_embeddings = torch.squeeze(token_embeddings, dim=1)
#     token_embeddings.size()
#     # Swap dimensions 0 and 1.
#     token_embeddings = token_embeddings.permute(1,0,2)
#     token_embeddings.size()
#
#     # Stores the token vectors, with shape [22 x 3,072]
#     token_vecs_cat = []
#     # `token_embeddings` is a [22 x 12 x 768] tensor.
#
#     # For each token in the sentence...
#     for token in token_embeddings:
#         # `token` is a [12 x 768] tensor
#
#         # Concatenate the vectors (that is, append them together) from the last
#         # four layers.
#         # Each layer vector is 768 values, so `cat_vec` is length 3,072.
#         cat_vec = torch.cat((token[-1], token[-2], token[-3], token[-4]), dim=0)
#
#         # Use `cat_vec` to represent `token`.
#         token_vecs_cat.append(cat_vec)
#
#     # `encoded_layers` has shape [12 x 1 x 22 x 768]
#     # `token_vecs` is a tensor with shape [22 x 768]
#     token_vecs = encoded_layers[11][0]
#
#     # Calculate the average of all token vectors.
#     sentence_embedding = torch.mean(token_vecs, dim=0)
#
#     print ("Our final sentence embedding vector of shape:", sentence_embedding.size())
#     return sentence_embedding





def raw_data_to_json(data_dir, json_dir):
    '''
    :param data_dir: dir and name of raw wechat dataset, for example,
            'C:\\Users\Ran\PycharmProjects\web_chatbot\information_retrieval\dataset\chat_short.txt'
            json_dir: save json file dir, for example,
            'C:\\Users\Ran\PycharmProjects\web_chatbot\information_retrieval\dataset\chat_short_2.json'
    :return: dataset in json file
    '''
    data_set = open(data_dir,'r',encoding='UTF-8')
    output_dict = dict()
    output_dict['chat'] = list()
    for line in data_set:
        line = line.split()
        if line[-1] != '你撤回了一条消息': #删掉‘你撤回了一条消息’ 行
            print(line)
            dic = {}
            dic['user'] = line[3]
            dic['status'] = line[4]
            dic['message_type'] = line[5]
            dic['text'] = line[7]
            output_dict['chat'].append(dic)

    with open(json_dir, 'w', encoding='UTF-8') as f:
        json.dump(output_dict, f, ensure_ascii=False)
    print('wrote to file: ',json_dir)

def json_to_qa(json_dir, qa_dir):
    '''
    :param json_dir: raw_data_to_json() generated files, for example
            'C:\\Users\Ran\PycharmProjects\web_chatbot\information_retrieval\dataset\chat_short_2.json'
    :param qa_dir: generates question-answer pairs in json file
    :return:
    '''

    print('openning file, please wait...')
    with open(json_dir, 'r', encoding='UTF-8') as f:
        data = json.load(f)
    print('successfully read file', json_dir)
    output_dict = dict()
    output_dict['qa'] = list()
    dic = {}
    dic['发送'] = data['chat'][0]['text']
    dic['接收'] = data['chat'][1]['text']

    for i in range(2, len(data['chat'])):
        message_type = data['chat'][i]['message_type']
        dic['id'] = i
        # if message_type is same to the last message_type, this message is uttered by the same person
        if message_type == (data['chat'][i - 1]['message_type']):
            message_type = data['chat'][i - 1]['message_type']
            dic[message_type] += ','+ data['chat'][i]['text']
        else:
            if message_type == '发送':
                output_dict['qa'].append(dic)
                dic = {}
                dic[message_type] = data['chat'][i]['text']

            if message_type == '接收':
                dic[message_type] = data['chat'][i]['text']
    print('complete QA pairs, writing to file...')
    with open(qa_dir, 'w', encoding='UTF-8') as f:
        json.dump(output_dict, f, ensure_ascii=False)
    print('wrote to file: ',qa_dir)
import numpy as np
import pickle
import sys

def get_embeddings(qa_dir, embedding_dir):
    '''
    :param qa_dir: qa dataset path, got from json_to_qa
    :param embedding_dir: generates embeddings, save in KDTree, save in .pickle
    :return: sentence level embeddings of Q pairs
    '''
    from bert_serving.client import BertClient
    bc = BertClient()

    import numpy as np
    from scipy import spatial
    import pickle

    output_dict = dict()
    output_dict['embeddings'] = list()

    print('openning file, please wait...')
    with open(qa_dir, 'r', encoding='utf-8-sig') as f:
        data = json.load(f)
    q_lst = []
    a_lst = []
    for i in range(len(data['qa'])):
        q_lst.append(data['qa'][i]['发送'])
        a_lst.append(data['qa'][i]['接收'])
    print('generating embeddings...')
    q_vec = bc.encode(q_lst)
    print('q_vec.shape: ', q_vec.shape)
    np.save('/root/projects/web_chatbot/information_retrieval/dataset/q_embedding_matrix',q_vec)
    print('wrote to file , q_embedding_matrix.pickle')


def generate_tree(embeddings_dir):
    embeddings = np.load(embeddings_dir)
    from sklearn.neighbors import BallTree

    print('constructing KDTree...')
    question_tree = BallTree(embeddings)
    print('finish embed process! ')
    pickle_out = open('/root/projects/web_chatbot/information_retrieval/dataset/q_BallTree.pickle', "wb")
    pickle.dump(question_tree, pickle_out)
    pickle_out.close()
    print('wrote to file , q_KDTree.pickle')

if __name__ == "__main__":

    dir = '/root/projects/web_chatbot/information_retrieval/dataset/chat_qa.json'

    generate_tree('/root/projects/web_chatbot/information_retrieval/dataset/q_embedding_matrix.npy')








