## # # # # # # # # # # # # # ## # # # # # # ## # # # # # # # # # # # # # # # # # # # ## # # # # # # # # # # #
#README:                                                                                                    #
#    1) change fin to the word vector file you want to evaluate.                                            #
#    2) pass --avgSimC as second argument to calculate avgSimC score, or --avgSim to calculate avgSim score.#
#    3) pass --globalSim as second argument to script to calculate the globalSim score.                     #
# # # # # # ## ## # # # # # # # # # # # # ## # # # # # # # # # # # # # # # ## # # # # # # # # # # # # # # # #
from __future__ import unicode_literals
import numpy as np
from scipy import stats
import gzip
from scipy import spatial
import sys
import time
fin = gzip.open('/Users/apple/Desktop/Spring-2016/NLP_IITD/jeevan_shankar-multi-sense-skipgram-74aeafd22528/wordvectors-L2R.gz')
#fin = gzip.open('/Users/apple/Desktop/Spring-2016/NLP_IITD/withoutmodif/vectors-MSSG.gz')

#fin = gzip.open('/Users/apple/Desktop/Spring-2016/NLP_IITD/jeevan_shankar-multi-sense-skipgram-74aeafd22528/sense2vec.gz')
fin = gzip.open('wordvectors-MSSG.gz')
lines = fin.readlines()
fin.close()



print ('initializing...')
import codecs
import os
import spacy.en
from spacy.tokens.doc import Doc
import re, string



LABELS = {
    'ENT': 'ENT',
    'PERSON': 'ENT',
    'NORP': 'ENT',
    'FAC': 'ENT',
    'ORG': 'ENT',
    'GPE': 'ENT',
    'LOC': 'ENT',
    'LAW': 'ENT',
    'PRODUCT': 'ENT',
    'EVENT': 'ENT',
    'WORK_OF_ART': 'ENT',
    'LANGUAGE': 'ENT',
    'DATE': 'DATE',
    'TIME': 'TIME',
    'PERCENT': 'PERCENT',
    'MONEY': 'MONEY',
    'QUANTITY': 'QUANTITY',
    'ORDINAL': 'ORDINAL',
    'CARDINAL': 'CARDINAL'
}


year_pattern = re.compile('^\d{4}')
punc_pattern = re.compile('[%s]' % re.escape(string.punctuation))
nlp = spacy.en.English()




def represent_word_pos(word):
    if word.like_url:
        return 'URL|X'
    elif year_pattern.match(word.text):
        return 'YEAR|X'
    text = re.sub(r'\s', '_', word.text)
    tag = LABELS.get(word.ent_type_, word.pos_)
    if not tag:
        tag = '?'
    return text + '|' + tag



class read_model:

    def __init__(self, read_vanilla_vec ):
        self.vocab = {}
        self.inverse_vocab = {}
        self.nCluster = {}
        self.weights = {}
        self.nClusterCount = {}
        self.nClusterCenter = {}
        self.read_vanilla = read_vanilla_vec
    def read_vanilla_vec(self):

        assert(self.read_vanilla == 1)
        V,D = map(lambda i: int(i), lines[0].strip().split(" "))
        j = 1
        for i in xrange(V):
            line = lines[j].strip().split(" ")
            j+=1
            self.vocab[i] = line[0].lower()
            self.inverse_vocab[self.vocab[i]] = i
            self.weights[i] = {}
            self.weights[i][0] = np.zeros(D)
            line = lines[j].strip().split(" ")
            for d in xrange(D): self.weights[i][0][d] = float(line[d])
            j+=1

    def read_l2r(self):
       V,D,S, _ = map(lambda i: int(i)  ,lines[0].strip().split(" "))
       j = 1

       for i in xrange(V):
        line = lines[j].strip().split(" ")
        j+=1
        self.vocab[i] = line[0].lower()
        self.inverse_vocab[self.vocab[i]] = i
        
        self.nCluster[i] = int(line[1]) if (len(line) > 1) else S
        if (len(line) > 2):
            self.nClusterCount[i] = []
            for k in xrange(self.nCluster[i]):
                self.nClusterCount[i].append( int(line(k+2)))
        self.weights[i] = {}
        self.nClusterCenter[i] = {}
        self.weights[i][0] = np.zeros(D)
        line = lines[j].strip().split(" ")
        for d in xrange(D): self.weights[i][0][d] = float(line[d])

        j+=1
        for s in xrange(1, self.nCluster[i] + 1):
            line = lines[j].strip().split(" ")
            j+=1
            self.weights[i][s] = np.zeros(D)
            for d in xrange(D): self.weights[i][s][d] = float(line[d])
            self.weights[i][s] /= np.linalg.norm(self.weights[i][s])
 


    def read_pretrained(self):
        V, D = map(lambda i: int(i)  ,lines[0].strip().split(" "))
        j = 1
        S = 3
        maxoutMethod = 0
        print(' # words in vocab := %d, size := %d, senses = %d, maxoutMethod = %d' %(V, D, S, maxoutMethod))
        for i in xrange(V):
            line = lines[j].strip().split(" ")
            j+=1
            self.vocab[i] = line[0].lower()
            self.inverse_vocab[self.vocab[i]] = i
        
            self.nCluster[i] = int(line[1]) if (len(line) > 1) else S
            if (len(line) > 2):
                self.nClusterCount[i] = []
                for k in xrange(self.nCluster[i]):
                    self.nClusterCount[i].append( int(line(k+2)))
            self.weights[i] = {}
            self.nClusterCenter[i] = {}
            self.weights[i][0] = np.zeros(D)
            line = lines[j].strip().split(" ")
            for d in xrange(D): self.weights[i][0][d] = float(line[d])

            j+=1
            for s in xrange(1, self.nCluster[i] + 1):
                line = lines[j].strip().split(" ")
                j+=1
                self.weights[i][s] = np.zeros(D)
                for d in xrange(D): self.weights[i][s][d] = float(line[d])
                self.weights[i][s] /= np.linalg.norm(self.weights[i][s])
                if (maxoutMethod == 0):
                    line = lines[j].strip().split(" ")
                    j+=1
                    self.nClusterCenter[i][s] = np.zeros(D)


dataset = "/Users/apple/Desktop/Spring-2016/NLP_IITD/Project/multi-sense/MSSG/resources/ratings.txt"
f = open(dataset, "r")
words = []
context = []
i = 0
average_scores = []
for line in f:
    ele = line.split("\t")

    words.append((ele[1].lower(), ele[3].lower()))
    context.append((ele[5], ele[6]))
    i+=1
    average_scores.append(float(ele[7]))


# for idx, context_pair in enumerate(context):
#     sent1 = nlp(context_pair[0])
#     sent2 = nlp(context_pair[1])
#     word1 = sent1[sent1.text.split().index('>')+1]
#     word2 = sent2[sent2.text.split().index('>')+1]
#     words.append((represent_word_pos(word1), represent_word_pos(word2)))



# f2 = open("output_sense.txt", "w")
# f1 = open("./SCWS/ratings_transformed.txt")
# l = f1.readlines();
# l = map(lambda i : i.strip(), l)
# i = 0
# words = []
# while (i+1 < len(l)):
#     words.append((l[i], l[i+1]))
#     f2.write(l[i] + "\t" + l[i+1] + "\n")
#     i+=2

# print(len(words))
f.close()
# f2.close()
# f1.close()
average_scores = np.array(average_scores)
Model = read_model(0) 
Model.read_l2r()
D = 300
maxoutMethod = 0

def getID(w1):
   # print w1
    return -1 if w1 not in Model.inverse_vocab else Model.inverse_vocab[w1] 



def get_embedding_context_all(id, cont):
    l = map(lambda i: i.lower().strip(), cont.split(" "))
    j =map(lambda i: getID(i), l)
    contextIDs = filter(lambda i : i != -1, j)
    senseContext = getSense(id, contextIDs)        
    return Model.weights[id][senseContext] if maxoutMethod != 0 else Model.nClusterCenter[id][senseContext]

def get_embedding_context(id, cont):

    l = map(lambda i: i.lower().strip(), cont.split(" "))
    word = Model.vocab[id]

    for i in xrange(len(l)):
        if l[i] == word:
            break


    contextIDs = []    
    j = i+1
    p = 0
    
    while p != 5 and j < len(l):
        id_cont = getID(l[j])
        if (id_cont != -1):
            contextIDs.append(id_cont)
            p+=1
        j+=1

    p = 0
    j = i-1

    while p != 5 and j > 0:
        id_cont = getID(l[j])
        if (id_cont != -1):
            contextIDs.append(id_cont)
            p+=1
        j-=1
    
    senseContext = getSense(id, contextIDs)
    return Model.weights[id][senseContext] if maxoutMethod == 0 else Model.nClusterCenter[id][senseContext]

def prob(id, contextEmbedding, currSense):
    currEmbedding = Model.weights[id][currSense]
    return 1.0- spatial.distance.cosine(contextEmbedding, currEmbedding)


opt = int(sys.argv[2]) if (len(sys.argv) > 2) else 1
def avgSimC(w1, w2, c):
    id1 = getID(w1); id2 = getID(w2);
    if (id1 == -1 or id2 == -1):
        return 0

    score = 0.0
    contextEmbedding_w1 = get_embedding_context(id1, context[c][0]) if opt == 1 else get_embedding_context_all(id1, context[c][0])
    contextEmbedding_w2 = get_embedding_context(id2, context[c][1]) if opt == 1 else get_embedding_context_all(id2, context[c][1])
    for i in xrange(1, Model.nCluster[id1]+1):
        for j in xrange(1,Model.nCluster[id2]+1):
            embedding_w1 = Model.weights[id1][i]
            embedding_w2 = Model.weights[id2][j]
            score_i = 1.0 -spatial.distance.cosine(embedding_w1, embedding_w2)
            #print(score_i)
            score+= score_i*prob(id1, contextEmbedding_w1, i)*prob(id2, contextEmbedding_w2, j) 
    #print("\t %s \t\t %s: %f" %(w1, w2, score))
    return score

def avgSim(w1, w2):
    id1 = getID(w1); id2 = getID(w2);
    if (id1 == -1 or id2 == -1):
        return 0
    
    score = 0.0
    for i in xrange(1, Model.nCluster[id1] +1):
        for j in xrange(1, Model.nCluster[id2] + 1):
            embedding_w1 = Model.weights[id1][i]
            embedding_w2 = Model.weights[id2][j]
            score += 1.0 - spatial.distance.cosine(embedding_w1, embedding_w2)
    return score/(Model.nCluster[id1]*Model.nCluster[id2])        


def global_sim(w1, w2):
    id1 = getID(w1); id2 = getID(w2);
    if (id1 == -1 or id2 == -1):
      #  print(w1, w2) 
        return -1

    embedding_w1 = Model.weights[id1][0]
    embedding_w2 = Model.weights[id2][0]
    return 1.0 - spatial.distance.cosine(embedding_w1, embedding_w2)

def getSense(wordId, contextIDs):
    contextEmbedding = np.zeros(D)
    for i in xrange(len(contextIDs)):
       contextEmbedding += Model.weights[contextIDs[i]][0]

    correctSense = 0
    max_score = -1*float('inf')
    for i in xrange(1, Model.nCluster[wordId]+1):
        score = np.dot(contextEmbedding, Model.weights[wordId][i])
        if (score > max_score): 
            max_score = score
            correctSense = i
    #print("correct sense: %d" % correctSense)

    return correctSense

perform_avgSimC = (sys.argv[1]  == '--avgSimC')
perform_avgSim  = (sys.argv[1] ==  '--avgSim') 
def calc_scores():

    if (perform_avgSimC): print("calculating the avgSimC score...")
    else: print("calculating global sim score")
    score = np.zeros(len(words)) 
    for i in xrange(len(words)):
        w1, w2 = words[i]
        w1 = w1.encode('ascii')
        w2 = w2.encode('ascii')
        print(w1, w2)
        if (perform_avgSimC == 1):
            score[i] = avgSimC(w1, w2, i)
        elif perform_avgSim == 1:
            score[i] = avgSim(w1, w2)
        else:
            score[i] = global_sim(w1, w2)
            print(score[i], i)

    return stats.spearmanr(score, average_scores)[0]#100.0*score/float(len(words))





if __name__ == "__main__":

    avgSim_score = calc_scores()
    print("Avg Sim score :=  %f" %( avgSim_score))


