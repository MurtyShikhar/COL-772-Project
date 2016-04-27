from keras import backend as K
from keras.models import Sequential
from keras.engine.topology import Layer
from keras.engine import InputSpec
from keras import initializations, activations
import theano
import theano.tensor as T

theano.config.optimizer = 'None'
theano.config.exception_verbosity ='high'
theano.optimizer='fast_compile'

class SenseEmbedding(Layer):
    ''' 
        Sense embeddings for NLP Project.
        Assumes K senses per word, and a global vector along with it.

    '''

    def __init__(self, features, vocab_size,  num_senses, context_size, input_dim = None, init = 'uniform', activation = 'sigmoid', **kwargs):
        self.input_dim = input_dim
        self.features = features 
        self.vocab_size = vocab_size
        self.init = initializations.get(init)
        self.activation = activations.get(activation)
        self.num_senses = num_senses
        kwargs['input_dtype'] = 'int32'

        if self.input_dim:
            kwargs['input_shape'] = (self.input_dim, ) 
        super(SenseEmbedding, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 2
        self.W_g = self.init((self.vocab_size, self.features))
        self.W_s = self.init((self.vocab_size, self.num_senses, self.features))
        self.trainable_weights = [self.W_g]

    def call(self, x, mask = None):
        W_g = self.W_g
        W_s = self.W_s
        nb = x.shape[0]

        # sum up the global vectors for all the context words, sum_context = nb x self.features
        sum_context = K.sum(W_g[x[:,2:]] , axis = 1)
        # sequence_vectors is a num_senses x nb x self.features
        sequence_vectors = W_s[x[:,0]].dimshuffle(1,0,2)
        # scores is a matrix of size num_senses x nb
        scores, ignore = theano.scan(lambda w: K.batch_dot(w, sum_context, axes = 1), sequences = [sequence_vectors], outputs_info = None)

        # right_senses is a vector of size nb
        right_senses = K.argmax(scores, axis = 0)
        # context_sense_vectors is a matrix of size nb x self.features
        correct_sense_vectors = W_s[x[:,0], right_senses]
        context_global_vectors = W_g[x[:,1]]
        dot_prod = K.batch_dot(correct_sense_vectors, context_global_vectors, axes = 1)
        #dot_prod  = T.nlinalg.diag(T.dot( context_sense_vectors, W_g[x[:,1]].T ))
        #return self.activation(T.sum(W_g[x[:,1]], axis = 0)) 
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



