## # # # # # # # # # # # # # ## # # # # # # ## # # # # # # # # # # # # # # # # # # # ## # # # # # # # # # # #
#README:                                                                                                    #
#    1) change fin to the word vector file you want to evaluate.                                            #
#    2) pass --avgSimC as second argument to calculate avgSimC score, or --avgSim to calculate avgSim score.#
#    3) pass --globalSim as second argument to script to calculate the globalSim score.                     #
# # # # # # ## ## # # # # # # # # # # # # ## # # # # # # # # # # # # # # # ## # # # # # # # # # # # # # # # #

import numpy as np
from scipy import stats
import gzip
from scipy import spatial
import sys
import time
#fin = gzip.open('/Users/apple/Desktop/Spring-2016/NLP_IITD/jeevan_shankar-multi-sense-skipgram-74aeafd22528/vectors-KMeans-neg15-MSSG.gz')
fin = gzip.open('/Users/apple/Desktop/Spring-2016/NLP_IITD/release/vectors.MSSG.300D.30K.gz')
lines = fin.readlines()
fin.close()

#V, D, S, maxoutMethod
V, D = map(lambda i: int(i)  ,lines[0].strip().split(" "))
S = 3
maxoutMethod = 0
print(' # words in vocab := %d, size := %d, senses = %d, maxoutMethod = %d' %(V, D, S, maxoutMethod))
vocab = {}
inverse_vocab = {}
nCluster = {}
weights = {}
nClusterCount = {}
nClusterCenter = {}
j = 1
for i in xrange(V):
    line = lines[j].strip().split(" ")
    j+=1
    vocab[i] = line[0].lower()
    inverse_vocab[vocab[i]] = i
    
    nCluster[i] = int(line[1]) if (len(line) > 1) else S
    if (len(line) > 2):
        nClusterCount[v] = []
        for k in xrange(nCluster[i]):
            nClusterCount[v].append( int(line(k+2)))
    weights[i] = {}
    nClusterCenter[i] = {}
    weights[i][0] = np.zeros(D)
    line = lines[j].strip().split(" ")
    for d in xrange(D): weights[i][0][d] = float(line[d])

    j+=1
    for s in xrange(1, nCluster[i] + 1):
        line = lines[j].strip().split(" ")
        j+=1
        weights[i][s] = np.zeros(D)
        for d in xrange(D): weights[i][s][d] = float(line[d])
        weights[i][s] /= np.linalg.norm(weights[i][s])
        if (maxoutMethod == 0):
            line = lines[j].strip().split(" ")
            j+=1
            nClusterCenter[i][s] = np.zeros(D)
            for d in xrange(D): nClusterCenter[i][s][d] = float(line[d])


    print '\r>> Done with %dth iteration' %i,
    sys.stdout.flush()
    #time.sleep(0.1)
fin.close()


dataset = "/Users/apple/Desktop/Spring-2016/NLP_IITD/Project/multi-sense/MSSG/resources/ratings.txt"
f = open(dataset, "r")
words = []
context = []
average_scores = []
for line in f:
    ele = line.split("\t")
    words.append((ele[1].lower(), ele[3].lower()))
    context.append((ele[5], ele[6]))
    average_scores.append(float(ele[7]))

print(len(words))
f.close()

average_scores = np.array(average_scores)

def getID(w1):
   # print w1
    return -1 if w1 not in inverse_vocab else inverse_vocab[w1] 



def get_embedding_context_all(id, cont):
    l = map(lambda i: i.lower().strip(), cont.split(" "))
    j =map(lambda i: getID(i), l)
    contextIDs = filter(lambda i : i != -1, j)
    senseContext = getSense(id, contextIDs)        
    return weights[id][senseContext] if maxoutMethod != 0 else nClusterCenter[id][senseContext]

def get_embedding_context(id, cont):

    l = map(lambda i: i.lower().strip(), cont.split(" "))
    word = vocab[id]

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
    
    print vocab[id],

    for i in contextIDs:
        print vocab[i],
        print " ",

    print("\n")    
    senseContext = getSense(id, contextIDs)
    #print(len(contextIDs))
    return weights[id][senseContext] if maxoutMethod != 0 else nClusterCenter[id][senseContext]

def prob(id, contextEmbedding, currSense):
    currEmbedding = weights[id][currSense]
    return 1.0- spatial.distance.cosine(contextEmbedding, currEmbedding)


opt = int(sys.argv[2]) if (len(sys.argv) > 2) else 1
def avgSimC(w1, w2, c):
    id1 = getID(w1); id2 = getID(w2);
    if (id1 == -1 or id2 == -1):
        return 0

    score = 0.0
    contextEmbedding_w1 = get_embedding_context(id1, context[c][0]) if opt == 1 else get_embedding_context_all(id1, context[c][0])
    contextEmbedding_w2 = get_embedding_context(id2, context[c][1]) if opt == 1 else get_embedding_context_all(id2, context[c][1])
    for i in xrange(1, nCluster[id1]+1):
        for j in xrange(1,nCluster[id2]+1):
            embedding_w1 = weights[id1][i]
            embedding_w2 = weights[id2][j]
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
    for i in xrange(1, nCluster[id1] +1):
        for j in xrange(1, nCluster[id2] + 1):
            embedding_w1 = weights[id1][i]
            embedding_w2 = weights[id2][j]
            score += 1.0 - spatial.distance.cosine(embedding_w1, embedding_w2)
    return score/(nCluster[id1]*nCluster[id2])        


def global_sim(w1, w2):
    id1 = getID(w1); id2 = getID(w2);
    if (id1 == -1 or id2 == -1): return -1

    embedding_w1 = weights[id1][0]
    embedding_w2 = weights[id2][0]
    return 1.0 - spatial.distance.cosine(embedding_w1, embedding_w2)

def getSense(wordId, contextIDs):
    contextEmbedding = np.zeros(D)
    for i in xrange(len(contextIDs)):
       contextEmbedding += weights[contextIDs[i]][0]

    correctSense = 0
    max_score = -1*float('inf')
    for i in xrange(1, nCluster[wordId]+1):
        score = np.dot(contextEmbedding, weights[wordId][i])
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
        if (perform_avgSimC == 1):
            score[i] = avgSimC(w1, w2, i)
        elif perform_avgSim == 1:
            score[i] = avgSim(w1, w2)
        else:
            score[i] = global_sim(w1, w2)

    return stats.spearmanr(score, average_scores)[0]#100.0*score/float(len(words))

if __name__ == "__main__":

    avgSim_score = calc_scores()
    print("Avg Sim score :=  %f" %( avgSim_score))


