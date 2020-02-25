import torch
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM

# OPTIONAL: if you want to have more information on what's happening, activate the logger as follows
import logging
logging.basicConfig(level=logging.INFO)

def get_sentence_embedding(sentence):
    # Load pre-trained model tokenizer (vocabulary)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    marked_text = "[CLS] " + sentence + " [SEP]"
    # Tokenize our sentence with the BERT tokenizer.
    tokenized_text = tokenizer.tokenize(marked_text)

    # Map the token strings to their vocabulary indeces.
    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)

    # Mark each of the 22 tokens as belonging to sentence "1".
    segments_ids = [1] * len(tokenized_text)

    # Convert inputs to PyTorch tensors
    tokens_tensor = torch.tensor([indexed_tokens])
    segments_tensors = torch.tensor([segments_ids])

    # Load pre-trained model (weights)
    model = BertModel.from_pretrained('bert-base-uncased')

    # Put the model in "evaluation" mode, meaning feed-forward operation.
    model.eval()

    # Predict hidden states features for each layer
    with torch.no_grad():
        encoded_layers, _ = model(tokens_tensor, segments_tensors)

    # print ("Number of layers:", len(encoded_layers))
    # layer_i = 0
    # print ("Number of batches:", len(encoded_layers[layer_i]))
    # batch_i = 0
    # print ("Number of tokens:", len(encoded_layers[layer_i][batch_i]))
    # token_i = 0
    # print ("Number of hidden units:", len(encoded_layers[layer_i][batch_i][token_i]))

    # Concatenate the tensors for all layers. We use `stack` here to. Could use different methods in later versions!
    # create a new dimension in the tensor.
    token_embeddings = torch.stack(encoded_layers, dim=0)
    token_embeddings.size()

    # Remove dimension 1, the "batches".
    token_embeddings = torch.squeeze(token_embeddings, dim=1)
    token_embeddings.size()
    # Swap dimensions 0 and 1.
    token_embeddings = token_embeddings.permute(1,0,2)
    token_embeddings.size()

    # Stores the token vectors, with shape [22 x 3,072]
    token_vecs_cat = []
    # `token_embeddings` is a [22 x 12 x 768] tensor.

    # For each token in the sentence...
    for token in token_embeddings:
        # `token` is a [12 x 768] tensor

        # Concatenate the vectors (that is, append them together) from the last
        # four layers.
        # Each layer vector is 768 values, so `cat_vec` is length 3,072.
        cat_vec = torch.cat((token[-1], token[-2], token[-3], token[-4]), dim=0)

        # Use `cat_vec` to represent `token`.
        token_vecs_cat.append(cat_vec)

    # `encoded_layers` has shape [12 x 1 x 22 x 768]
    # `token_vecs` is a tensor with shape [22 x 768]
    token_vecs = encoded_layers[11][0]

    # Calculate the average of all token vectors.
    sentence_embedding = torch.mean(token_vecs, dim=0)

    print ("Our final sentence embedding vector of shape:", sentence_embedding.size())
    return sentence_embedding

if __name__ == "__main__":
    get_sentence_embedding('今天天气真好，一起去散步吧')