from keras import backend as K
from keras.models import Sequential
from keras.engine.topology import Layer
from keras.engine import InputSpec
from keras import initializations, activations
import theano
import theano.tensor as T
class WordEmbedding(Layer):
    ''' 
        Sense embeddings for NLP Project.
        Assumes K senses per word, and a global vector along with it.

    '''

    def __init__(self, input_dim, vector_dim, context_size, input_length = None, init = 'uniform', activation = 'sigmoid', **kwargs):
        self.input_dim = input_dim
        self.vector_dim = vector_dim 
        self.init = initializations.get(init)
        self.activation = activations.get(activation)
        self.input_length = input_length
        kwargs['input_dtype'] = 'int32'
        kwargs['input_shape'] = (self.input_length, ) 
        super(WordEmbedding, self).__init__(**kwargs)

    def build(self, input_shape):
        self.W_g = self.init((self.input_dim, self.vector_dim))
        self.trainable_weights = [self.W_g]

    def call(self, x, mask = None):
        W_g = self.W_g
        dot_prod = T.nlinalg.diag(T.dot(W_g[x[:,0]], W_g[x[:,1]].T))
        return self.activation(dot_prod)

    # def get_output_shape_for(self, input_shape):
    #     assert input_shape and len(input_shape) == 2
    #     return (input_shape[0], 1)

    def get_config(self):
         return {"name":self.__class__.__name__,
                    "input_dim":self.input_dim,
                    "proj_dim":self.vector_dim,
                    "init":self.init.__name__,
                    "activation":self.activation.__name__}