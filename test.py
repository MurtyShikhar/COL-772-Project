import numpy as np
import theano
import theano.tensor as T
import theano.printing as printing
theano.config.optimizer = 'None'
theano.config.exception_verbosity ='high'
theano.optimizer='fast_compile'

vocab  = 10
senses = 5
epsilon = 0.1
vect_dim = 6



W_g = theano.shared(np.random.randint(5, size = (vocab, vect_dim)))
W_s = theano.shared(np.random.randint(5, size=  (vocab, senses, vect_dim)))

def get_vector(curr_word, new_sense):
	cond = T.eq(new_sense, -1)
	return T.switch(cond, W_g[curr_word], W_s[curr_word][new_sense])


def change_context_vec(vect, new_sense, prev_sense, curr_word):
	return vect - get_vector(curr_word, prev_sense) + get_vector(curr_word, new_sense)


def l2C(curr_word, i, curr_senses, context_vector):
	# theano vector of size (num_senses,)
	scores_all_senses = T.dot(context_vector, W_s[curr_word].T)

	sorted_senses = T.argsort(scores_all_senses)
	score_best = scores_all_senses[sorted_senses[-1]]
	score_second_best = scores_all_senses[sorted_senses[-2]]

	prev_sense  = curr_senses[i]
	context_vector = T.switch(T.gt(score_best-score_second_best, epsilon),  change_context_vec(context_vector, sorted_senses[-1], prev_sense, curr_word), context_vector )
	new_senses = T.set_subtensor(curr_senses[i], sorted_senses[-1])
	return [new_senses, context_vector]


# context words are 2*context_size+1
def loss_fn_per_context(word_position,context):
	# sum up the global vectors of the context
	context_vector = T.sum(W_g[context], axis = 0)
	#return context_vector
	# start with -1 with none of the words disambiguated
	start = -1*T.ones_like(context)

	output_alg, updates = theano.scan(l2C, sequences = [context, T.arange(4)], outputs_info = [start, context_vector])

	disambiguated_senses = output_alg[0][-1]
	augmented_context_vector = output_alg[1][-1]


	sense_of_actual_word = disambiguated_senses[word_position]
	#return T.argsort(T.dot(context_vector, W_s[actual_word].T)), T.dot(context_vector, W_s[actual_word].T)

	actual_word = context[word_position]
	# #scores, ignore_updates = theano.scan(lambda s: T.dot(W_s[])   , sequences = [disambiguated_senses])

	def score(i):
		return T.switch(T.eq(i, actual_word), 0, T.log(T.nnet.sigmoid(T.dot(W_g[actual_word], W_g[i]))))

	scores, ignore_updates  = theano.scan(score, sequences = [context])

	def calc_score(context_word, sense_of_context_word):
	 	return T.switch(T.eq(context_word, actual_word), 0, T.log(T.nnet.sigmoid(T.dot(W_s[actual_word][sense_of_actual_word], W_s[context_word][sense_of_context_word] ))))

	sense_scores, ignore_updates_ = theano.scan(calc_score, sequences = [context, disambiguated_senses])
	loss_this_example = T.sum(scores, axis = 0) + T.sum(sense_scores, axis = 0)
	return loss_this_example



x = theano.shared(np.array([ [0,1,1,2], [1,2,2,1], [2,3,1,0]]))
nb = x.get_value().shape[0]
window = x.get_value().shape[1]
loss, updates = theano.scan(loss_fn_per_context, sequences = [ x[:,0] ,x], outputs_info = None)
#total_loss = T.sum(loss)/nb

score = loss[-1]
f = theano.function([], loss)
print(f())