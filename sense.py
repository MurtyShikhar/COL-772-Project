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
        self.W_s = self.init((self.input_dim, self.num_senses, self.vector_dim))
        self.trainable_weights = [self.W_g, self.W_s]

    def call(self, x, mask = None):
        print("shape: ", x.shape)
        W_g = self.W_g
        W_s = self.W_s
        nb = x.shape[0]
        # sum up the global vectors for all the context words, sum_context = nb x self.vector_dim
        sum_context = T.sum(W_g[x[:,2:]] , axis = 1)
        # sequence_vectors is a num_senses x nb x self.vector_dim
        sequence_vectors = W_s[x[:,0]].dimshuffle(1,0,2)
        # scores is a matrix of size num_senses x nb
        scores, ignore = theano.scan(lambda w: T.nlinalg.diag(T.dot(w, sum_context.T)), sequences = [sequence_vectors], outputs_info = None)

        # right_senses is a vector of size nb
        right_senses = T.argmax(scores, axis = 0)
        # context_sense_vectors is a matrix of size nb x self.vector_dim
        context_sense_vectors = W_s[x[:,0]][:,right_senses, :][T.arange(nb), 0]
        dot_prod = T.nlinalg.diag(T.dot( context_sense_vectors, W_g[x[:,1]].T ))

        return self.activation(dot_prod)

    def get_output_shape_for(self, input_shape):
        assert input_shape and len(input_shape) == 2
        return (input_shape[0], 1)

    def get_config(self):
         return {"name":self.__class__.__name__,
                    "input_dim":self.vector_dim,
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

