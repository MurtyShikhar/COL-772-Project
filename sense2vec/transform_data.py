from __future__ import unicode_literals
from __future__ import print_function
import codecs
import os
import spacy.en
from spacy.tokens.doc import Doc
import re, string
from nltk.corpus import stopwords
# from gensim.models import word2vec


class Parameters:
    def __init__(self):
        self.file_name = '../wikipedia-dump/text8'
        self.out_file = '../wikipedia-dump/text9'
        self.CLUSTER = False
        self.POS = True
        self.n_workers = 4
        self.vec_dimension_size = 128
        self.window = 5
        self.min_count = 10
        self.negative = 5 
        self.n_iterations = 2
    def model_name(self):
        return 'sense2vec_cluster-'+str(self.CLUSTER)+'_pos-'+str(self.POS)


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


cachedStopWords = stopwords.words("english")
year_pattern = re.compile('^\d{4}')
punc_pattern = re.compile('[%s]' % re.escape(string.punctuation))
nlp = spacy.en.English()


def invalid_word(word):
    return word.is_space or punc_pattern.match(word.text) or word.text in cachedStopWords


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

param = Parameters()
in_file_path = param.file_name 
CLUSTER = param.CLUSTER
POS = param.POS
count = 0

file = codecs.open(in_file_path, 'r', 'utf-8')
out_file = codecs.open(param.out_file, 'w', 'utf-8')
print ('open')
for text in file:
    if count % 100 == 0:
        print (count)
    doc = nlp(text)
    doc.is_parsed = True
    for ent in doc.ents:
        ent.merge(ent.root.tag_, ent.text, LABELS[ent.label_])
    for np in doc.noun_chunks:
        while len(np) > 1 and np[0].dep_ not in ('advmod', 'amod', 'compound'):
            np = np[1:]
        np.merge(np.root.tag_, np.text, np.root.ent_type_)
    new_line = ' '.join([represent_word_pos(w) for w in doc if not invalid_word(w)])+'\n'
    out_file.write(new_line)
    count += 1

