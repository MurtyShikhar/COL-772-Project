from keras import backend as K
from keras.models import Sequential
from keras.engine.topology import Layer
from keras.engine import InputSpec
from keras import initializations, activations
import theano
import theano.tensor as T
class SenseEmbedding(Layer):
    ''' 
        Sense embeddings for NLP Project.
        Assumes K senses per word, and a global vector along with it.

    '''

    def __init__(self, input_dim, vector_dim, num_senses, context_size, input_length = None, init = 'uniform', activation = 'sigmoid', **kwargs):
        self.input_dim = input_dim
        self.vector_dim = vector_dim 
        self.init = initializations.get(init)
        self.activation = activations.get(activation)
        #self.input_spec = [InputSpec(ndim=2, dtype = 'int32'), InputSpec(ndim=context_size, 'int32')] 
        self.num_senses = num_senses
        self.input_length = input_length
        kwargs['input_dtype'] = 'int32'
        kwargs['input_shape'] = (self.input_length, ) 
        super(SenseEmbedding, self).__init__(**kwargs)

    def build(self, input_shape):
        self.W_g = self.init((self.input_dim, self.vector_dim))
        self.W_s = self.init((self.input_dim, self.vector_dim, self.num_senses))
        self.trainable_weights = [self.W_g, self.W_s]

    def call(self, x, mask = None):
        sum_context = T.sum(self.W_g[x[:,2:]])
        scores, ignore = theano.scan(lambda w: T.dot(w, sum_context), sequences = [self.W_s[x[:,0]]], outputs_info = None)
        right_sense = T.argmax(scores)
        dot_prod = T.dot(self.W_s[x[:,0]][right_sense], self.W_g[x[:,1]])
        return self.activation(dot_prod)

    def get_output_shape_for(self, input_shape):
        assert input_shape and len(input_shape) == 2
        return (input_shape[0], 1)

    def get_config(self):
         return {"name":self.__class__.__name__,
                    "input_dim":self.input_dim,
                    "proj_dim":self.proj_dim,
                    "init":self.init.__name__,
                    "activation":self.activation.__name__}


#if __name__ == "__main__":
#    model = Sequential()
#    vocab_size = 1e4
#    dim = 200
#    num_senses = 3
#    context_size = 4
#    model.add(SenseEmbedding(vocab_size, dim, num_senses, context_size))

