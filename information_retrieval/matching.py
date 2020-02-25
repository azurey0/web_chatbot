from information_retrieval.get_embeddings import get_sentence_embedding

# Corpus with example pairing sentences
corpus = ['A man is eating food.',
          'A man is eating a piece of bread.',
          'The girl is carrying a baby.',
          'A man is riding a horse.',
          'A woman is playing violin.',
          'Two men pushed carts through the woods.',
          'A man is riding a white horse on an enclosed ground.',
          'A monkey is playing drums.',
          'A cheetah is running behind its prey.']

queries = ['A man is eating pasta.',
           'Someone in a gorilla costume is playing a set of drums.',
           'A cheetah chases prey on across a field.']

def get_similarity(embedding_a, embedding_b):
    from scipy.spatial.distance import cosine
    cosine = cosine(embedding_a.unsqueeze(0), embedding_b.unsqueeze(0))
    return 1-cosine

if __name__ == "__main__":
    embedding_a = get_sentence_embedding('A man is eating food.')
    embedding_b = get_sentence_embedding('A man is eating pasta.')
    print(get_similarity(embedding_b, embedding_a))
