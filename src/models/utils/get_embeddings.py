import json
import numpy as np
from typing import Dict, List
from keras import Model


def get_product_embeddings() -> Dict[int, np.ndarray]:
    with open('data/processed/product_embedding.json', 'r') as file:
        embeddings = json.load(file)
    return embeddings