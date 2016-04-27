from keras import backend as K
from keras.models import Sequential
from keras.engine.topology import Layer
from keras.engine import InputSpec
from keras import initializations, activations
import theano
def cos_sim(vector1, vector2):
    return K.dot(vector1,vector2)
class SenseEmbedding(Layer):
    ''' 
        Sense embeddings for NLP Project.
        Assumes K senses per word, and a global vector along with it.

    '''

    def __init__(self, num_senses, vocab_dim, vector_dim, context_size = 0, input_dim = 1, output_dim = 1, init = 'uniform', activation = 'sigmoid', **kwargs):
        self.init = initializations.get(init)
        self.activation = activations.get(activation)
        self.output_dim = output_dim
        self.input_dim = input_dim + context_size
        self.vector_dim = vector_dim
        self.vocab_dim = vocab_dim
        self.num_senses = num_senses
        kwargs['input_dtype'] = 'int32'
        if self.input_dim:
            kwargs['input_shape'] = (self.input_dim, ) 
        super(SenseEmbedding, self).__init__(**kwargs)

    def build(self, input_shape):
        self.W_g = self.init((self.vocab_dim, self.vector_dim))
        self.W_s = self.init((self.vocab_dim, self.num_senses, self.vector_dim))
        self.trainable_weights = [self.W_g, self.W_s]

    def call(self, x, mask = None):
        W_g = self.W_g
        W_s = self.W_s
        nb = x.shape[0]
        # sum up the global vectors for all the context words, sum_context = nb x self.vector_dim
        sum_context = K.sum(W_g[x[:,2:]] , axis = 1)
        # sequence_vectors is a num_senses x nb x self.vector_dim
        sequence_vectors = W_s[x[:,0]].dimshuffle(1,0,2)
        # scores is a matrix of size num_senses x nb
        scores, ignore = theano.scan(lambda w: K.batch_dot(w, sum_context, axes = 1), sequences = [sequence_vectors], outputs_info = None)

        # right_senses is a vector of size nb
        right_senses = K.argmax(scores, axis = 0)
        # context_sense_vectors is a matrix of size nb x self.vector_dim
        context_sense_vectors = W_s[x[:,0]][x[:,0], right_senses]
        dot_prod = K.batch_dot(context_sense_vectors, W_g[x[:,1]], axes = 1)

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

