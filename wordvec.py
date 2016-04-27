from keras import backend as K
from keras.models import Sequential
from keras.engine.topology import Layer
from keras.engine import InputSpec
from keras import initializations, activations
import inspect
class WordEmbedding(Layer):
    ''' 
        Simple word embeddings for NLP Project.

    '''
# TODO: CHOOSE BETTER INITIALIZATION
    def __init__(self, vocab_dim, vector_dim, context_size = 0, input_dim = 1, output_dim = 1, init = 'uniform', activation = 'sigmoid', **kwargs):
        self.init = initializations.get(init)
        self.activation = activations.get(activation)
        self.output_dim = output_dim
        self.input_dim = input_dim + context_size
        self.vector_dim = vector_dim
        self.vocab_dim = vocab_dim
        kwargs['input_dtype'] = 'int32'
        if self.input_dim:
            kwargs['input_shape'] = (self.input_dim, ) 
        super(WordEmbedding, self).__init__(**kwargs)

    def build(self, input_shape):
        self.W_g = self.init((self.vocab_dim, self.vector_dim))
        self.trainable_weights = [self.W_g]


    def call(self, x, mask = None):
        W_g = self.W_g
        dot_prod = K.batch_dot(W_g[x[:,0]], W_g[x[:,1]], axes = 1)
        return self.activation(dot_prod)

    def get_output_shape_for(self, input_shape):
        assert input_shape and len(input_shape) == 2
        return (input_shape[0], 1)

    def get_config(self):
         return {"name":self.__class__.__name__,
                    "input_dim":self.input_dim,
                    "proj_dim":self.vector_dim,
                    "init":self.init.__name__,
                    "activation":self.activation.__name__}