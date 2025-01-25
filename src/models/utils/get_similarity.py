
import numpy as np
import faiss

def find_most_similar_cosine(prediction, embeddings_dict):
    
    most_similar_id = None
    highest_similarity = -1 

    for product_id, embedding in embeddings_dict.items():
        similarity = np.dot(prediction, embedding)
        
        if similarity > highest_similarity:
            highest_similarity = similarity
            most_similar_id = product_id

    return most_similar_id, highest_similarity


def find_top_n_similar_cosine(prediction, embeddings_dict, top_n=3):
    
    similarities = []

    for product_id, embedding in embeddings_dict.items():
        
        similarity = np.dot(prediction, embedding)
        similarities.append((product_id, similarity))
    
    similarities = sorted(similarities, key=lambda x: x[1], reverse=True)[:top_n]
    return similarities


def find_top_n_faiss(prediction, embeddings_dict, top_n=3):
    product_ids = list(embeddings_dict.keys())
    embedding_matrix = np.vstack(list(embeddings_dict.values())).astype('float32')

    index = faiss.IndexFlatL2(embedding_matrix.shape[1])  # L2 distancia euclidea
    index.add(embedding_matrix)

    prediction = np.array(prediction, dtype='float32').reshape(1, -1)
    distances, indices = index.search(prediction, top_n)

    results = [(product_ids[i], distances[0][j]) for j, i in enumerate(indices[0])]
    return results