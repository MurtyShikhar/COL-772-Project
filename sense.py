from keras import backend as K
from keras.engine.topology import Layer
import theano
import theano.tensor as T
class SenseEmbedding(Layer):
    ''' 
        Sense embeddings for NLP Project.
        Assumes K senses per word, and a global vector along with it.

    '''

    def __init__(self, input_dim, vector_dim, num_senses, init = 'uniform', activation = 'sigmoid'):
        self.input_dim = input_dim
        self.vector_dim = vector_dim 
        self.init = initializations.get(init)
        self.activation = activations.get(activation)
        self.W_g = self.init((input_dim, vector_dim))
        self.W_s = self.init((input_dim, vector_dim, K))
        self.params = [self.W_g, self.W_s]

    def call(self, x):
        sum_context = T.sum(self.W_g[x[1]])
        scores, ignore = T.scan(lambda w: T.dot(w, sum_context), sequences = [self.W_s[x[0]]], outputs = None)
        curr_sense = T.argmax(scores)


    def get_config(self):
         return {"name":self.__class__.__name__,
                    "input_dim":self.input_dim,
                    "proj_dim":self.proj_dim,
                    "init":self.init.__name__,
                    "activation":self.activation.__name__}
