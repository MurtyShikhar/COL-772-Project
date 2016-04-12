from __future__ import unicode_literals
from __future__ import print_function
import codecs
import os
import spacy.en
from spacy.tokens.doc import Doc
import re, string
from nltk.corpus import stopwords
from gensim.models import word2vec


class Parameters:
    def __init__(self):
        self.data_directory = '../../WestburyLab.Wikipedia.Corpus/'
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


def represent_word_both(word):
    if word.like_url:
        return 'URL|X'
    elif year_pattern.match(word.text):
        return 'YEAR|X'
    text = re.sub(r'\s', '_', word.text)
    tag = LABELS.get(word.ent_type_, word.pos_) + str(word.cluster)
    if not tag:
        tag = '?'
    return text + '|' + tag


def represent_word_cluster(word):
    if word.like_url:
        return 'URL|X'
    elif year_pattern.match(word.text):
        return 'YEAR|X'
    text = re.sub(r'\s', '_', word.text)
    tag = str(word.cluster)
    if not tag:
        tag = '?'
    return text + '|' + tag


class MyCorpus(object):
    def __init__(self, Parameters):
        self.Parameters = Parameters
    
    def __iter__(self):
        Parameters = self.Parameters
        in_directory = Parameters.data_directory 
        CLUSTER = Parameters.CLUSTER
        POS = Parameters.POS
        count = 0
        for data_file in os.listdir(in_directory):
            if count >= 1000:
                count = 0
                raise StopIteration
                break
            in_file_path = os.path.join(in_directory, data_file)
            print (in_file_path)
            text = codecs.open(in_file_path, 'r', 'utf-8').read()
            print ('read')
            doc = nlp(text)
            print ('parsed')
            doc.is_parsed = True
            if POS and not CLUSTER:
                for ent in doc.ents:
                    ent.merge(ent.root.tag_, ent.text, LABELS[ent.label_])
                print ('done ents')
                for np in doc.noun_chunks:
                    while len(np) > 1 and np[0].dep_ not in ('advmod', 'amod', 'compound'):
                        np = np[1:]
                    np.merge(np.root.tag_, np.text, np.root.ent_type_)
                print ('done np')
                for sent in doc.sents:
                    if sent.text.strip():
                        yield [represent_word_pos(w) for w in sent if not invalid_word(w)]
            elif CLUSTER and not POS:
                for sent in doc.sents:
                    if sent.text.strip():
                        yield [represent_word_cluster(w) for w in sent if not invalid_word(w)]
            else:
                for ent in doc.ents:
                    ent.merge(ent.root.tag_, ent.text, LABELS[ent.label_])
                for np in doc.noun_chunks:
                    while len(np) > 1 and np[0].dep_ not in ('advmod', 'amod', 'compound'):
                        np = np[1:]
                    np.merge(np.root.tag_, np.text, np.root.ent_type_)
                for sent in doc.sents:
                    if sent.text.strip():
                        yield [represent_word_both(w) for w in sent if not invalid_word(w)]
            count += 1


def get_vectors(CLUSTER, POS):
    param = Parameters()
    param.CLUSTER = CLUSTER
    param.POS = POS
    corpus = MyCorpus(param)
    model = word2vec.Word2Vec(
        corpus,
        size=param.vec_dimension_size,
        window=param.window,
        min_count=param.min_count,
        workers=param.n_workers,
        sample=1e-5,
        negative=param.negative
    )
    # model = word2vec.Word2Vec(corpus, min_count =1)
    model.init_sims(replace=True)
    model.save(param.model_name())


get_vectors(False,True)
print ('*******************************************************************************************************')
get_vectors(True,False)
print ('*******************************************************************************************************')
# get_vectors(True,True)