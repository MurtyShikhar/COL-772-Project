import numpy as np
from scipy import spatial, stats


class Evaluate(model, tokenizer):

    def __init__(self,tokenizer, words, context, average_scores):
        self.tokenizer = tokenizer
        self.words = words
        self.context = context
        self.word_index = tokenizer.word_index
        self.average_scores = average_scores

    def get_scores(self,global_vectors, sense_vectors):
        words = self.words

        global_sim_vect = np.zeros(len(words))
        avgSimC_vect    = np.zeros(len(words))

        for i in xrange(len(words)):
            f_word, s_word = words[i] 
            global_sim[i]  = self.global_similarity(f_word , s_word)
            avgSimC[i]     = self.average_similarity(f_word, s_word)

        avgSimC_score    = stats.spearmanr(self.average_scores, avgSimC_vect)[0]
        global_sim_score = stats.spearmanr(self.average_scores, global_sim_vect)[0]
        return avgSimC_score, global_sim_score  

    def global_word_vector(self, w, global_vectors):
        i = self.word_index.get(w)
        if (not i):
            return None
        return global_vectors[i]

    def sense_word_vector(self, i, sense, sense_vectors):
        assert (sense < 3)
        return sense_vectors[i][sense]

    def global_similarity(self,f_word, s_word, global_vectors):
      v1 = global_word_vector(f_word)
      v2 = global_word_vector(s_word)
      if (not v1 or not v2): return -1
      else:
         return 1.0- spatial.distance.cosine(v1, v2)  

    def average_similarity(self, f_word, s_word, sense_vectors):
        id_f = self.word_index.get(f_word)
        id_s = self.word_index.get(s_word)
        if (id_f is None or id_s is None):
            return -1
        else:
            score = 0.0
            for sense_f in xrange(3):
                for sense_s in xrange(3)
                    v1 = sense_word_vector(id_f, sense_f)
                    v2 = sense_word_vector(id_s, sense_s)
                    score += (1.0 - spatial.distance.cosine(v1, v2))

            return score/9.0