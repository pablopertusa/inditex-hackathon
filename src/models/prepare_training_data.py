import polars as pl
import numpy as np
from tqdm import tqdm
from utils.get_embeddings import get_product_embeddings
from get_sequences import get_sequences

train_embeddings = np.load('data/processed/train_embeddings.npy')
train = pl.read_csv('data/processed/train_index.csv')
products_embeddings = get_product_embeddings()

unique_sessions = train['session_id'].unique()

X_list = []
y_list = []
count = 0
for id in tqdm(unique_sessions):
    df = train.filter(pl.col('session_id') == id)
    sequences, products = get_sequences(df, train_embeddings)
    
    X_list.append(sequences)
    y_list.append(products)

X = np.concatenate(X_list, axis=0)
y_prods = np.concatenate(y_list, axis=0)
y = []
for p in y_prods:
    y.append(products_embeddings[str(p)])
y = np.array(y)

print(X.shape)
print(y.shape)

np.save('data/processed/training_sequences.npy', X)
np.save('data/processed/training_products.npy', y)

