# Para ejecutar este archivo si tienes configurado que tensorflow y keras usen la grafica, ten en cuenta que debes tener suficiente memoria en la grafica
# ya que puede ocurrir un error durante la ejecucion

import numpy as np
import json
import polars as pl
from keras.models import Sequential
from keras.layers import Embedding

schema = {
    "session_id": pl.Int64,
    "date": pl.Utf8,
    "timestamp_local": pl.Utf8,
    "add_to_cart": pl.Int32,
    "user_id": pl.Float64,
    "country": pl.Int64,
    "partnumber": pl.Int32,
    "device_type": pl.Int32,
    "pagetype": pl.Float64
}

train = pl.read_csv("data/raw/train.csv", schema=schema)
train_ex = train.select(pl.exclude(['date', 'user_id']))

train_dt = (
    train_ex
    .with_columns(
        pl.col('timestamp_local').str.strptime(pl.Datetime, format='%Y-%m-%d %H:%M:%S%.f').alias('timestamp')
    )
    .with_columns(
        pl.col('timestamp').dt.timestamp().alias('unix_timestamp'),
        pl.col('timestamp').dt.hour().alias('hour'),
        pl.col('timestamp').dt.week().alias('week'),
        pl.col('timestamp').dt.weekday().alias('weekday')
    )
    .select(pl.exclude('timestamp_local'))
    .with_columns(
        ((pl.col("unix_timestamp") - pl.col("unix_timestamp").min())
        / (pl.col("unix_timestamp").max() - pl.col("unix_timestamp").min())
    ).alias("unix_timestamp_normalized"))
    .select(pl.exclude(['timestamp', 'unix_timestamp']))
)

train_dt = (
    train_dt
    .fill_null(0)
    .with_columns(
        pl.col('pagetype').cast(pl.Int32)
    )
    .with_columns([
    (pl.col("hour") - pl.col("hour").min()) / (pl.col("hour").max() - pl.col("hour").min()).alias("hour_normalized"),
    (pl.col("week") - pl.col("week").min()) / (pl.col("week").max() - pl.col("week").min()).alias("week_normalized"),
    (pl.col("weekday") - pl.col("weekday").min()) / (pl.col("weekday").max() - pl.col("weekday").min()).alias("weekday_normalized"),
    ])
)

pagetype_vocab = train_dt['pagetype'].unique().to_list()
country_vocab = train_dt['country'].unique().to_list()
device_vocab = train_dt['device_type'].unique().to_list()

model_pagetype = Sequential([
    Embedding(input_dim=len(pagetype_vocab) + 1, output_dim=16)
])

model_country = Sequential([
    Embedding(input_dim=len(country_vocab) + 1, output_dim=3)
])

model_device = Sequential([
    Embedding(input_dim=len(device_vocab) + 1, output_dim=2)
])

device_embedding = model_device(np.array(train_dt['device_type'].to_list()))
# Crear un índice del vocabulario (necesario para la capa Embedding)
vocab_to_index = {val: idx for idx, val in enumerate(country_vocab)}
data_indices = [vocab_to_index[val] for val in train_dt['country'].to_list()]

country_embedding = model_country(np.array(data_indices))
pagetype_embedding = model_pagetype(np.array(train_dt['pagetype'].to_list()))


with open('data/processed/product_embedding.json', 'r') as file:
        product_embeddings = json.load(file)

train_product_embeddings = [product_embeddings[str(p)] for p in train_dt['partnumber'].to_list()]

add_to_cart = np.array(train_dt['add_to_cart'].to_list())
hour = np.array(train_dt['hour'].to_list())
week = np.array(train_dt['week'].to_list())
weekday = np.array(train_dt['weekday'].to_list())
timestamp = np.array(train_dt['unix_timestamp_normalized'].to_list())
device = device_embedding.numpy()
country = country_embedding.numpy()
pagetype = pagetype_embedding.numpy()
train_product_embeddings = np.array(train_product_embeddings)

arrays = [
    np.expand_dims(hour, axis=1) if hour.ndim == 1 else hour,
    np.expand_dims(week, axis=1) if week.ndim == 1 else week,
    np.expand_dims(weekday, axis=1) if weekday.ndim == 1 else weekday,
    np.expand_dims(timestamp, axis=1) if timestamp.ndim == 1 else timestamp,
    np.expand_dims(device, axis=1) if device.ndim == 1 else device, 
    np.expand_dims(country, axis=1) if country.ndim == 1 else country,
    np.expand_dims(pagetype, axis=1) if pagetype.ndim == 1 else pagetype,
    train_product_embeddings
    ]

combined_embeddings = np.hstack(arrays)

np.save('data/processed/train_embeddings.npy', combined_embeddings) # Esto es la matriz de embeddings de train

train = (
    train
    .with_columns(
        pl.Series(range(10**6)).alias('index') # Para luego acceder a la matriz de embeddings
    )
)
train.write_csv('data/processed/train_index.csv')

# Ahora test

test = pl.read_csv("data/raw/test.csv")
test_ex = test.select(pl.exclude(['date', 'user_id']))

test_dt = (
    test_ex
    .with_columns(
        pl.col('timestamp_local').str.strptime(pl.Datetime, format='%Y-%m-%d %H:%M:%S%.f').alias('timestamp')
    )
    .with_columns(
        pl.col('timestamp').dt.timestamp().alias('unix_timestamp'),
        pl.col('timestamp').dt.hour().alias('hour'),
        pl.col('timestamp').dt.week().alias('week'),
        pl.col('timestamp').dt.weekday().alias('weekday')
    )
    .select(pl.exclude('timestamp_local'))
    .with_columns(
        ((pl.col("unix_timestamp") - pl.col("unix_timestamp").min())
        / (pl.col("unix_timestamp").max() - pl.col("unix_timestamp").min())
    ).alias("unix_timestamp_normalized"))
    .select(pl.exclude(['timestamp', 'unix_timestamp']))
)

test_dt = (
    test_dt
    .fill_null(0)
    .with_columns(
        pl.col('pagetype').cast(pl.Int32)
    )
    .with_columns([
    (pl.col("hour") - pl.col("hour").min()) / (pl.col("hour").max() - pl.col("hour").min()),
    (pl.col("week") - pl.col("week").min()),
    (pl.col("weekday") - pl.col("weekday").min()) / (pl.col("weekday").max() - pl.col("weekday").min()),
    ])
)
device_embedding = model_device(np.array(test_dt['device_type'].to_list()))
# Crear un índice del vocabulario (necesario para la capa Embedding)
vocab_to_index = {val: idx for idx, val in enumerate(country_vocab)}
data_indices = [vocab_to_index[val] for val in test_dt['country'].to_list()]

country_embedding = model_country(np.array(data_indices))

vocab_to_index = {val: idx for idx, val in enumerate(pagetype_vocab)}
data_indices = [vocab_to_index[val] for val in test_dt['pagetype'].to_list()]

pagetype_embedding = model_pagetype(np.array(data_indices))

test_product_embeddings = [product_embeddings[str(p)] for p in test_dt['partnumber'].to_list()]

test_hour = np.array(test_dt['hour'].to_list())
test_week = np.array(test_dt['week'].to_list())
test_weekday = np.array(test_dt['weekday'].to_list())
test_timestamp = np.array(test_dt['unix_timestamp_normalized'].to_list())
test_device = device_embedding.numpy()
test_country = country_embedding.numpy()
test_pagetype = pagetype_embedding.numpy()
test_product_embeddings = np.array(test_product_embeddings)

test_arrays = [
    np.expand_dims(test_hour, axis=1) if test_hour.ndim == 1 else test_hour,
    np.expand_dims(test_week, axis=1) if test_week.ndim == 1 else test_week,
    np.expand_dims(test_weekday, axis=1) if test_weekday.ndim == 1 else test_weekday,
    np.expand_dims(test_timestamp, axis=1) if test_timestamp.ndim == 1 else test_timestamp,
    np.expand_dims(test_device, axis=1) if test_device.ndim == 1 else test_device, 
    np.expand_dims(test_country, axis=1) if test_country.ndim == 1 else test_country,
    np.expand_dims(test_pagetype, axis=1) if test_pagetype.ndim == 1 else test_pagetype,
    test_product_embeddings
]

test_combined_embeddings = np.hstack(test_arrays)

np.save('data/processed/test_embeddings.npy', test_combined_embeddings)

test = (
    test
    .with_columns(
        pl.Series(range(len(test))).alias('index')
    )
)
test.write_csv('data/processed/test_index.csv')