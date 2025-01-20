# Naive predictions

import polars as pl
import json

test = pl.read_csv("data/raw/test.csv")
unique_session_ids = test['session_id'].unique().to_list()

predictions = {}
targets = {}

with open('data/processed/most_populars.json', 'r') as file:
    most_populars = json.load(file)

for id in unique_session_ids:
    id = str(int(id))
    targets[id] = most_populars

predictions['target'] = targets

with open('predictions/predictions_3.json', 'w') as file_predictions:
    json.dump(predictions, file_predictions, indent=4)

    
