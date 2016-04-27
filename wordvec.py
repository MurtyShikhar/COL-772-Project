from keras import backend as K
from keras.models import Sequential
from keras.engine.topology import Layer
from keras.engine import InputSpec
from keras import initializations, activations
class WordEmbedding(Layer):
    ''' 
        Word embeddings for NLP Project.

    '''

    def __init__(self, features, vocab_size, context_size, input_dim = None, init = 'uniform', activation = 'sigmoid', **kwargs):
        self.input_dim = input_dim
        self.features = features
        self.vocab_size = vocab_size
        self.init = initializations.get(init)
        self.activation = activations.get(activation)
        kwargs['input_dtype'] = 'int32'
        if self.input_dim:
            kwargs['input_shape'] = (self.input_dim, ) 
        super(WordEmbedding, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 2
        #input_dim = input_shape[1]
        # input_shape is going to be (None, self.input_dim) in any case
        self.W_g = self.init((self.vocab_size, self.features))
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