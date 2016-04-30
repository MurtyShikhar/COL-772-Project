from keras import backend as K
from keras.models import Sequential
from keras.engine.topology import Layer
from keras.engine import InputSpec
from keras import initializations, activations
import theano
import theano.tensor as T
import theano.printing as printing
from random import sample
from .common import _EPSILON
epsilon = 0.1
p = printing.Print('x')

def get_vector(curr_word, new_sense, W_g, W_s):
    cond = T.eq(new_sense, -1)
    return T.switch(cond, W_g[curr_word], W_s[curr_word, new_sense])

# update the sense of a word in the context vector
def change_context_vec(vect, new_sense, prev_sense, curr_word, W_g, W_s):
    return vect - get_vector(curr_word, prev_sense, W_g, W_s) + get_vector(curr_word, new_sense, W_g, W_s)

# Gets the best sense for word according to context_vector
def get_best_sense(l2c, word, curr_sense, context_vector):
    scores_all_senses = T.dot(context_vector, W_s[curr_word].T)
    sorted_senses = T.argsort(scores_all_senses)
    score_best = scores_all_senses[sorted_senses[-1]]
    score_second_best = scores_all_senses[sorted_senses[-2]]
    new_sense = T.switch(T.gt(score_best-score_second_best, epsilon), sorted_senses[-1], curr_sense)
    l2c.sense_vocab[word].add(new_sense)
    return new_sense

# updates the context vector with the best sense of the curr_word with (prev) sense curr_senses[i]
def update_context_vec_with_best_sense(curr_word, i, curr_senses, context_vector, W_g, W_s, l2c):
    prev_sense  = curr_senses[i]
    new_sense = get_best_sense(l2c, curr_word,prev_sense,context_vector)
    context_vector =   change_context_vec(context_vector, new_sense, prev_sense, curr_word, W_g, W_s)
    new_senses = T.set_subtensor(curr_senses[i], new_sense)
    return [new_senses, context_vector]



# Perform word sense disambiguation and learn better context vector
# ASSUME THAT WORD IS THE MOD OF CONTEXT
def disambiguate_context(word, context, W_g, W_s, l2c):
    # sum up the global vectors of the context
    context_vector = T.sum(W_g[context], axis = 0)
    # start with -1 with none of the words disambiguated
    start_senses = -1*T.ones_like(context)
    output_alg, updates = theano.scan(update_context_vec_with_best_sense, sequences = [context, T.arange(4)], outputs_info = [start_senses, context_vector], non_sequences = [W_g, W_s, l2c])
    disambiguated_senses = output_alg[0][-1]
    word_sense = disambiguated_senses[len(disambiguated_senses)/2]
    return disambiguated_senses, word_sense

def sample_context(word, context, W_g, W_s, l2c):
    context_senses = theano.scan(lambda context_word: sample(l2c.sense_vocab[context_word],1), sequences=[context])
    context_vector = K.sum(theano.scan(get_vector, sequences=[context, context_senses] non_sequences=[W_g, W_s]))
    sense_of_word = get_best_sense(word, -1, context_vector)
    return context_senses, sense_of_word

# Get loss for word and context
# context words are 2*context_size+1
def loss_percontext(word, word_sense, context, context_senses):    
    def calc_score(context_word, context_word_sense):
        return K.sigmoid(K.dot(get_vector(word, word_sense),get_vector(context_word, context_word_sense)))
    mid_index = len(context)/2
    return K.mean(theano.scan(calc_score , sequences = [context[:mid_index]+context[mid_index+1:], context_senses[:mid_index]+context_senses[mid_index+1:]]))


class L2CFastEmbedding(Layer):
    # Creates multiple emdeddings per sense of the word.
    # Also keeps track of the senses of the context words and uses their embeddings.

    def __init__(self, num_senses, vocab_dim, vector_dim, input_dim, output_dim = 1, init = 'uniform', activation = 'linear', **kwargs):
        self.input_dim = input_dim
        self.vector_dim = vector_dim 
        self.vocab_dim = vocab_dim
        self.init = initializations.get(init)
        self.activation = activations.get(activation)
        self.output_dim = output_dim
        self.num_senses = num_senses
        # self.sense_vocab[word] contains all the possible senses which the model is CONFIDENT about. -1 means W_b[word] else i means W_s[word,i]
        self.sense_vocab = [{-1} for i in range(vocab_dim)]
        kwargs['input_dtype'] = 'int32'
        if self.input_dim:
            kwargs['input_shape'] = (self.input_dim, ) 
        super(L2CEmbedding, self).__init__(**kwargs)

    def build(self, input_shape):
        self.W_g = self.init((self.vocab_dim, self.vector_dim))
        self.W_s = self.init((self.vocab_dim, self.num_senses, self.vector_dim))
        self.trainable_weights = [self.W_g, self.W_s]

    def call(self, x, mask = None):
        # x is of dimension nb x (2*context_size + 2) where x[:,0] are the words, x[:,2:] is the context, x[:,1] indicates positive or negative example
        W_g = self.W_g
        W_s = self.W_s
        nb = x.shape[0]
        context_length = self.input_dim - 2
        def final_loss(word, is_pos, context):
            right_senses, word_sense = theano.switch(is_pos ==1, disambiguate_context(word, context, W_g, W_s, self), sample_contexts(word, context, W_g, W_s, self))
            return loss_percontext(word, word_sense, context, context_senses)
        return theano.scan(final_loss, sequences=[x[:,0], x[:,1], x[:,2:]])

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
