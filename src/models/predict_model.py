
import polars as pl
import json
from keras.models import load_model
import numpy as np
from get_sequences import get_sequences
from utils.get_similarity import find_most_similar_cosine, find_top_n_similar_cosine, find_top_n_faiss
from tqdm import tqdm

test = pl.read_csv("data/processed/test_index.csv")
unique_session_ids = test['session_id'].unique().to_list()
model = load_model('models/recommender.keras')
test_embeddings = np.load('data/processed/test_embeddings.npy')
predictions = {}
targets = {}

with open('data/processed/product_embedding.json', 'r') as file:
    product_embeddings = json.load(file)

for id in tqdm(unique_session_ids):
    id_str = str(int(id))
    df = test.filter(pl.col('session_id') == id)
    sequences, _ = get_sequences(df, test_embeddings) # obtenemos las secuencias de 3 interacciones en test
    preds = model.predict(sequences, verbose = 0) # se predice un producto por cada secuencia de 3 interacciones
    most_likely = find_top_n_faiss(preds[-1], product_embeddings, top_n=5)
    targets[id_str] = [int(x[0]) for x in most_likely]

predictions['target'] = targets

with open('predictions/predictions_3.json', 'w') as file_predictions:
    json.dump(predictions, file_predictions, indent=4)

    
