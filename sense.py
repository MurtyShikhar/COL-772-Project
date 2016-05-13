from keras import backend as K
from keras.models import Sequential
from keras.engine.topology import Layer
from keras.engine import InputSpec
from keras import initializations, activations
import theano
import theano.tensor as T


class MatrixFactorization(Layer):
    
        # Embeddings for matrix factorization 
        # 

    def __init__(self,  , vector_dim, input_dim, output_dim = 1, init = 'uniform', activation = 'sigmoid', **kwargs):
        self.input_dim = input_dim
        self.vector_dim = vector_dim 
        self.init = initializations.get(init)
        self.activation = activations.get(activation)
        self.output_dim = output_dim

        kwargs['input_dtype'] = 'int32'
        if self.input_dim:
            kwargs['input_shape'] = (self.input_dim, ) 
        super(SenseEmbedding, self).__init__(**kwargs)

    def build(self, input_shape):
        E = self.num_entities
        R = self.num_relations
        self.tupleEmbed   = self.init((E, E, self.vector_dim))
        self.relationEmbed   = self.init((R, self.vector_dim))
        self.trainable_weights = [self.W_g, self.W_s]

    def call(self, x, mask = None):
        tupleEmbed    = self.tupleEmbed
        relationEmbed = self.relationEmbed
        nb = x.shape[0]
        entity_embeddings   = tupleEmbed[x[:, 0], x[:, 2]]
        relation_embeddings = relationEmbed[x[:, 1]]
        dot_prod = K.batch_dot(entity_embeddings, relation_embeddings, axes = 1)
        return self.activation(dot_prod)


    def get_output_shape_for(self, input_shape):
        assert input_shape and len(input_shape) == 2
        return (input_shape[0], 1)

    def get_config(self):
         return {"name":self.__class__.__name__,
                    "input_dim":self.input_dim,
                    "vector_dim":self.vector_dim,
                    "vocab_dim" :self.vocab_dim,
                    "init":self.init.__name__,
                    "activation":self.activation.__name__}



