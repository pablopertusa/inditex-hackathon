
import polars as pl
import pandas as pd
import json
import json


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

schema_clients = {
    "user_id": pl.Float64,
    "country": pl.Int64,
    "R": pl.Int64,
    "F": pl.Int64,
    "M": pl.Float64
}

train = pl.read_csv("data/raw/train.csv", schema=schema)
clients = pl.read_csv("data/raw/users_data.csv", schema=schema_clients)
products_pandas = pd.read_pickle("data/raw/products.pkl")
products = pl.from_pandas(products_pandas)

# Solo nos vamos a quedar con las interacciones de sesionnes con usuarios
train_users_not_null = train.filter(pl.col('user_id').is_not_null())

products = products.select(['discount', 'partnumber', 'color_id', 'cod_section', 'family'])
products = products.with_columns(
    products["cod_section"].cast(pl.Int32).alias("cod_section")
)
clients = clients.rename({'country' : 'user_country'})

train_with_users = train_users_not_null.join(clients, on = 'user_id', how = 'left')
train_total = train_with_users.join(products, on = 'partnumber', how = 'left')
train_total = train_total.select(pl.exclude(['session_id', 'date']))

train_total.write_csv('data/processed/train_users.csv')

# Ahora obtemos los productos más populares para recomendarlos a las sesiones que no tienen interacciones previas

populares = (
    train
    .group_by('partnumber')
    .agg(pl.len().alias('visitas'), pl.col('add_to_cart').sum().alias('compras'))
    .sort(by = ['compras','visitas'], descending=[True, True])
    .head(5)
    .select('partnumber')
)
most_populars = populares['partnumber'].to_list()

with open('data/processed/most_populars.json', 'w') as file:
    json.dump(most_populars, file)

print('Most populars creados')