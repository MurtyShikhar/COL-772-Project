## # # # # # # # # # # # # # ## # # # # # # ## # # # # # # # # # # # # # # # # # # # ## # # # # # # # # # # #
#README:                                                                                                    #
#    1) change fin to the word vector file you want to evaluate.                                            #
#    2) pass --avgSimC as second argument to calculate avgSimC score, or --avgSim to calculate avgSim score.#
#    3) pass --globalSim as second argument to script to calculate the globalSim score.                     #
# # # # # # ## ## # # # # # # # # # # # # ## # # # # # # # # # # # # # # # ## # # # # # # # # # # # # # # # #
from __future__ import unicode_literals
from __future__ import print_function
print ('initializing...')
import codecs
import sys
import os
import spacy.en
from spacy.tokens.doc import Doc
import re, string
from gensim.models import word2vec
from scipy import stats

model_name = 'sense2vec_cluster-False_pos-True'
test_dataset = '../SCWS/ratings.txt'

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


print ('loading model')
model = word2vec.Word2Vec.load(model_name)
print ('loading test data')
f = open(test_dataset, "r")
words = []
context = []
average_scores = []
for line in f:
    ele = line.split("\t")
    words.append((ele[1].lower(), ele[3].lower()))
    context.append((ele[5], ele[6]))
    average_scores.append(float(ele[7]))

average_pred_scores = [0 for i in range(len(average_scores))]
print(len(words))
f.close()


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


def calc_scores():
    count = 0
    for idx, context_pair in enumerate(context):
        sent1 = nlp(context_pair[0])
        sent2 = nlp(context_pair[1])
        word1 = sent1[sent1.text.split().index('>')+1]
        word2 = sent2[sent2.text.split().index('>')+1]
        sim = 0
        try:
            sim = 5*model.similarity(represent_word_pos(word1),represent_word_pos(word2)) + 5
        except Exception, e:
            count += 1
            print ('ignoring pair number '+str(count))
        average_pred_scores[idx] = sim
    return stats.spearmanr(average_pred_scores, average_scores)[0]#100.0*score/float(len(words))

if __name__ == "__main__":

    avgSim_score = calc_scores()
    print("Avg Sim score :=  %f" %( avgSim_score))


