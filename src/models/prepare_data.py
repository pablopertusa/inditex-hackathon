# Para el primer modelo lo que vamos a hacer es tener en cuenta solo los usuarios para hacer un modelo content-based,
# ya que con solo las sesiones no podemos tener un historial del usuario. Este modelo solo será usado para las 
# predicciones de las sesiones asociadas con un usuario.

import polars as pl
import pandas as pd
import json
from keras.models import Sequential
from keras.layers import Embedding
import tensorflow as tf
import numpy as np
import json


products_pandas = pd.read_pickle("data/raw/products.pkl")
products = pl.from_pandas(products_pandas)

# Products embeddings

products = products.select(pl.exclude('embedding')) # Esta columna venía ya predefinida pero no la voy a usar
products = (
    products
    .with_columns(
        pl.col("cod_section").cast(pl.Int32).alias('cod_int')
    )
    .fill_null(0)
)

color_vocab = products['color_id'].unique().to_list()
cod_vocab = products['cod_int'].unique().to_list()
family_vocab = products['family'].unique().to_list()

# Crear el modelo
model_color = Sequential([
    Embedding(input_dim=len(color_vocab) + 1, output_dim=64)
])

model_cod = Sequential([
    Embedding(input_dim=len(cod_vocab) + 1, output_dim=4)
])

model_family = Sequential([ 
    Embedding(input_dim=len(family_vocab) + 1, output_dim=24)
])

colores = np.array(products['color_id'].to_list())
embedding_colores = model_color(colores)
cods = np.array(products['cod_int'].to_list())
embedding_cods = model_cod(cods)
familias = np.array(products['family'].to_list())
embedding_familias = model_family(familias)
discounts = np.array(products['discount'].to_list(), dtype = float)

print('Embeddings creados')

dict_products_embeddings = {}
for i, p in enumerate(products['partnumber'].to_list()):
    discount = np.array([discounts[i]])
    embedding = np.concatenate((embedding_colores[i].numpy().tolist(), embedding_cods[i].numpy().tolist(), embedding_familias[i].numpy().tolist(), discount)).tolist()
    dict_products_embeddings[p] = embedding

print('Dict creado')

with open('data/processed/product_embedding.json', 'w') as file:
    json.dump(dict_products_embeddings, file, indent=4)


# clients embeddings

# clients = clients.select(pl.exclude('embedding'))
# clients = (
#     clients
#     .with_columns(
#         pl.col("cod_section").cast(pl.Int32).alias('cod_int')
#     )
#     .fill_null(0)
# )

# color_vocab = clients['color_id'].unique().to_list()
# cod_vocab = clients['cod_int'].unique().to_list()
# family_vocab = clients['family'].unique().to_list()

# # Crear el modelo
# model_color = Sequential([
#     Embedding(input_dim=len(color_vocab) + 1, output_dim=64)
# ])

# model_cod = Sequential([
#     Embedding(input_dim=len(cod_vocab) + 1, output_dim=4)
# ])

# model_family = Sequential([ 
#     Embedding(input_dim=len(family_vocab) + 1, output_dim=24)
# ])

# colores = np.array(clients['color_id'].to_list())
# embedding_colores = model_color(colores)
# cods = np.array(clients['cod_int'].to_list())
# embedding_cods = model_cod(cods)
# familias = np.array(clients['family'].to_list())
# embedding_familias = model_family(familias)

# print('Embeddings creados')

# dict_clients_embeddings = {}
# for i, p in enumerate(clients['partnumber'].to_list()):
#     embedding = np.concatenate((embedding_colores[i].numpy().tolist(), embedding_cods[i].numpy().tolist(), embedding_familias[i].numpy().tolist())).tolist()
#     dict_clients_embeddings[p] = embedding

# print('Dict creado')

# with open('data/processed/product_embedding.json', 'w') as file:
#     json.dump(dict_clients_embeddings, file, indent=4)



