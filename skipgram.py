from keras.models import Sequential
from keras.layers.embeddings import Embedding
from keras.la
models = []
feature_dim = 300
vocab = 1e4


model_word = Sequential()

model_word.add(Embedding(vocab, feature_dim ))
model_word.add(
