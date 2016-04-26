from keras import backend as K
from keras.models import Sequential
from keras.engine.topology import Layer
from keras.engine import InputSpec
from keras import initializations, activations
class WordEmbedding(Layer):
    ''' 
        Sense embeddings for NLP Project.
        Assumes K senses per word, and a global vector along with it.

    '''

    def __init__(self, output_dim, context_size, input_dim = None, init = 'uniform', activation = 'sigmoid', **kwargs):
        self.init = initializations.get(init)
        self.activation = activations.get(activation)
        self.output_dim = output_dim
        self.input_dim = input_dim
        kwargs['input_dtype'] = 'int32'
        if self.input_dim:
            kwargs['input_shape'] = (self.input_dim, ) 
        super(WordEmbedding, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 2
        input_dim = input_shape[1]
        self.W_g = self.init((input_dim, self.output_dim))
        self.trainable_weights = [self.W_g]

    def call(self, x, mask = None):
        W_g = self.W_g
        dot_prod = K.batch_dot(W_g[x[:,0]], W_g[x[:,1]], axes = 1)
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