import random
import numpy as np
from keras.preprocessing import text, sequence


    # Take a sequence (list of indexes of words),
    # returns couples of [word_index, other_word index] and labels (1s or 0s),
    # where label = 1 if 'other_word' belongs to the context of 'word',
    # and label=0 if 'other_word' is ramdomly sampled
    # # Arguments
    #     vocabulary_size: int. maximum possible word index + 1
    #     window_size: int. actually half-window.
    #         The window of a word wi will be [i-window_size, i+window_size+1]
    #     negative_samples: float >= 0. 0 for no negative (=random) samples.
    #         1 for same number as positive samples. etc.
    #     categorical: bool. if False, labels will be
    #         integers (eg. [0, 1, 1 .. ]),
    #         if True labels will be categorical eg. [[1,0],[0,1],[0,1] .. ]
    # # Returns
    #     couples, lables: where `couples` are int pairs and
    #         `labels` are either 0 or 1.
    # # Notes
    #     By convention, index 0 in the vocabulary is
    #     a non-word and will be skipped.
    

def skipgrams_wordvec(sequence, vocabulary_size,
              window_size=4, negative_samples=1., shuffle=True,
              categorical=False, sampling_table=None):
    couples = []
    labels = []

    for i, wi in enumerate(sequence):
        if not wi:
            continue
        if sampling_table is not None:
            if sampling_table[wi] < random.random():
                continue

# TODO: ASSUMES EACH WORD OCCURS ATMOST ONCE PER SENTENCE
               
        window_start = max(0, i-window_size)
        window_end = min(len(sequence), i+window_size+1)
        for j in range(window_start, window_end):
            if j != i:
                wj = sequence[j]
                if not wj:
                    continue
                couples.append([wi, wj, j- window_start])
                if categorical:
                    labels.append([0,1])
                else:
                    labels.append(1)

# TODO: SAMPLING OF NEGATIVE EXAMPLES SHOULD BE FROM A DIFFERENT DISTRIUBTION D^{3/4} WHERE D IS THE WORD DISTRIBUTION
    if negative_samples > 0:
        nb_negative_samples = int(len(labels) * negative_samples)
        words = [c[0] for c in couples]
        random.shuffle(words)
        couples += [[words[i %len(words)], random.randint(1, vocabulary_size-1)] for i in range(nb_negative_samples)]
        if categorical:
            labels += [[1,0]]*nb_negative_samples
        else:
            labels += [0]*nb_negative_samples

    if shuffle:
        seed = random.randint(0,10e6)
        random.seed(seed)
        random.shuffle(couples)
        random.seed(seed)
        random.shuffle(labels)
    return couples, labels

def skipgrams_sense(sequence, vocabulary_size, num_senses = 3,
              window_size=4, negative_samples=1., shuffle=True,
              categorical=False, sampling_table=None):
    couples = []
    labels = []
    dict_of_contexts = {}

    for i, wi in enumerate(sequence):
        if not wi:
            continue
        if sampling_table is not None:
            if sampling_table[wi] < random.random():
                continue

# TODO: ASSUMES EACH WORD OCCURS ATMOST ONCE PER SENTENCE
               
        window_start = max(0, i-window_size)
        window_end = min(len(sequence), i+window_size+1)
        to_add = sequence[window_start: window_end]
        while (len(to_add) != 2*window_size + 1):
            to_add.append(0)

        dict_of_contexts[wi] = to_add
        for j in range(window_start, window_end):
            if j != i:
                wj = sequence[j]
                if not wj:
                    continue
                couples.append([wi, wj])
                if categorical:
                    labels.append([0,1])
                else:
                    labels.append(1)

# TODO: SAMPLING OF NEGATIVE EXAMPLES SHOULD BE FROM A DIFFERENT DISTRIUBTION D^{3/4} WHERE D IS THE WORD DISTRIBUTION
    if negative_samples > 0:
        nb_negative_samples = int(len(labels) * negative_samples)
        words = [c[0] for c in couples]
        random.shuffle(words)
        # FOR NEGATIVE SAMPLES, -i INDICATES SENSE A NEATIVE SAMPLE OF SENSE i
        couples += [[words[i %len(words)], random.randint(1, vocabulary_size-1)] for i in range(nb_negative_samples)]
        if categorical:
            labels += [[1,0]]*nb_negative_samples
        else:
            labels += [0]*nb_negative_samples

    if shuffle:
        seed = random.randint(0,10e6)
        random.seed(seed)
        random.shuffle(couples)
        random.seed(seed)
        random.shuffle(labels)
    couples_augmented = [[x,y]+dict_of_contexts[x] for (x, y) in couples]
    return couples_augmented, labels

def skipgrams_l2c(sequence, vocabulary_size, num_senses = 3,
              window_size=4, negative_samples=1., shuffle=True,
              categorical=False, sampling_table=None):
    triples = []
    labels = []
    dict_of_contexts = {}

    for i, wi in enumerate(sequence):
        if not wi:
            continue
        if sampling_table is not None:
            if sampling_table[wi] < random.random():
                continue

# TODO: ASSUMES EACH WORD OCCURS ATMOST ONCE PER SENTENCE
               
        window_start = max(0, i-window_size)
        window_end = min(len(sequence), i+window_size+1)
        to_add = sequence[window_start: window_end]
        while (len(to_add) != 2*window_size + 1):
            to_add.append(0)

        dict_of_contexts[wi] = to_add
        for j in range(window_start, window_end):
            if j != i:
                wj = sequence[j]
                if not wj:
                    continue
                triples.append([wi, wj, j- window_start])
                if categorical:
                    labels.append([0,1])
                else:
                    labels.append(1)

# TODO: SAMPLING OF NEGATIVE EXAMPLES SHOULD BE FROM A DIFFERENT DISTRIUBTION D^{3/4} WHERE D IS THE WORD DISTRIBUTION
    if negative_samples > 0:
        nb_negative_samples = int(len(labels) * negative_samples)
        words = [c[0] for c in triples]
        random.shuffle(words)
        # FOR NEGATIVE SAMPLES, -i INDICATES SENSE A NEATIVE SAMPLE OF SENSE i
        triples += [[words[i %len(words)], random.randint(1, vocabulary_size-1), -1*np.random.randint(num_senses+1)] for i in range(nb_negative_samples)]
        if categorical:
            labels += [[1,0]]*nb_negative_samples
        else:
            labels += [0]*nb_negative_samples

    if shuffle:
        seed = random.randint(0,10e6)
        random.seed(seed)
        random.shuffle(triples)
        random.seed(seed)
        random.shuffle(labels)
    triples_augmented = [[x,y,z]+dict_of_contexts[x] for (x, y, z) in triples]
    return triples_augmented, labels