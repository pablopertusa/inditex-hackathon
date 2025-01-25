import numpy as np
from keras.preprocessing.sequence import pad_sequences

def get_sequences(df, train_embeddings,  length = 3): # length es la cantidad de interacciones previas que se usaran para predecir el siguiente producto
    
    resul_x = []
    resul_y = []
    l = len(df)
    
    for i in range(0, l, length):
        subset = df.slice(i, length + 1)
        if len(subset) == 1:
            sub_embeddings = train_embeddings[subset['index'].to_list()]
        else:
            sub_embeddings = train_embeddings[subset['index'].to_list()[:-1]]
        np_embeddings = np.array([sub_embeddings])
        partnumber = subset['partnumber'].last()
        padded_embeddings = pad_sequences(np_embeddings, maxlen = 3, padding = 'pre', truncating = 'pre', dtype='float32')
        padded_embeddings = np.squeeze(padded_embeddings, axis=0)
        resul_x.append(padded_embeddings)
        resul_y.append(partnumber)

    resul_x = np.array(resul_x)
    resul_y = np.array(resul_y)
    return resul_x, resul_y