from keras.models import Sequential
from keras.layers.embeddings import Embedding
from keras.layers.core import Reshape
from keras.preprocesing.sequence import * 
models = []
feature_dim = 300
vocab = 1e4


model_word = Sequential()

model_word.add(Embedding(vocab, feature_dim ))
model.add(Reshape(dims=(feature_dim,)))
models.append(model_word)

model_context = Sequential()
model_context.add(Embedding(vocab, feature_dim, input_length=1))
model_context.add(Reshape(dims=(feature_dim,)))
models.append(model_context)

model = Sequential()
model.add(Merge(models, mode='dot'))
model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss='mse', optimizer=Adam(lr=0.001))

sampling_table = make_sampling_table(max_feature)

for e in xrange(num_epochs):
    print("-"*40)
    print("epoch#", e)
    print("-"*40)
    samples_seen = 0
    losses = []
    for i, seq in enumerate(tokenizer.texts_to_sequences_generator(text_generator())):
        couples, labels = skipgrams(seq, max_feature, window_size=4, negative_samples=1., sampling_table=sampling_table)
        if couples:
            X = np.array(couples,dtype="int32")
            loss = model.train_on_batch(X, labels)
            losses.append(loss)
            samples_seen +=1
    

