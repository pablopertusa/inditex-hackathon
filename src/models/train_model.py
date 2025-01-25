import tensorflow as tf
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, Input
from keras.callbacks import EarlyStopping
import numpy as np

X = np.load('data/processed/training_sequences.npy')
y = np.load('data/processed/training_products.npy')

model = Sequential([
    Input(shape = (3, 110)),
    LSTM(128, return_sequences=False, activation='tanh'),
    Dense(64, activation='relu'),
    Dropout(0.1),
    Dense(93, activation='linear')
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    loss='mean_squared_error',
    metrics=['mae']
)

early_stopping = EarlyStopping(
    monitor='loss', 
    patience=2, 
    restore_best_weights=True,
    verbose=1
)

history = model.fit(
    X, 
    y, 
    validation_split=0.15,
    batch_size=256,
    epochs=10,
    verbose=1,
    callbacks = [early_stopping]
)

model.save('models/recommender.keras')
