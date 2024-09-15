import numpy as np
import torch
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer, util


def calculate_similarity(text1, text2):
    """
    Calculate the cosine similarity between two input texts using Sentence-BERT embeddings.
    
    Args:
        text1 (str): The first text to compare.
        text2 (str): The second text to compare.

    Returns:
        float: The cosine similarity score between the two texts.
    """
    # Set device to GPU if available, otherwise use CPU
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Initialize SentenceTransformer model and set device
    model = SentenceTransformer('paraphrase-MiniLM-L6-v2', device=device)

    # Compute text embeddings directly on the device and convert them to NumPy arrays
    embedding1 = model.encode([text1], convert_to_tensor=True).cpu().numpy()
    embedding2 = model.encode([text2], convert_to_tensor=True).cpu().numpy()

    # Calculate cosine similarity between the embeddings
    similarity = cosine_similarity(embedding1, embedding2)

    # Output similarity
    print(f"Similarity between '{text1}' and '{text2}' is: {similarity[0][0]:.4f}")

    return similarity[0][0]


def semantic_search(corpus, queries, top_k=1):
    """
    Perform semantic search to determine how well a query matches the corpus.

    Args:
        corpus (list of str): A list of documents (answers) to search within.
        queries (list of str): A list of query strings to search for in the corpus.
        top_k (int): The number of top matches to return for each query (default is 1).

    Returns:
        float: The highest matching score for the first query.
    """
    # Set device to GPU if available, otherwise use CPU
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Initialize SentenceTransformer model and specify device
    model = SentenceTransformer('paraphrase-MiniLM-L6-v2', device=device)

    # Encode the corpus and queries into embeddings
    corpus_embeddings = model.encode(corpus, convert_to_tensor=True)
    queries_embeddings = model.encode(queries, convert_to_tensor=True)

    # Perform semantic search to find top_k most relevant matches
    hits = util.semantic_search(queries_embeddings, corpus_embeddings, top_k=top_k)

    # Display the first query and its top match
    print(f"Query: {queries[0]}")
    for hit in hits[0]:
        print(f"Match: {corpus[hit['corpus_id']]} (Score: {hit['score']:.4f})")
    
    # Return the highest matching score for the first query
    return hits[0][0]['score'] if hits[0] else None


if __name__ == '__main__':
    corpus = ['bookshelf']
    queries = ["What is the item that's too big for the program?"]
    semantic_search(corpus, queries)
    # calculate_similarity()
    # semantic_search()
