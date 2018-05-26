"""
Purpose: simple 2-3 line helper methods that deeper classes will use.
"""
import sys
import numpy as np
import collections
from nltk.corpus import stopwords

reload(sys)
sys.setdefaultencoding('utf-8')

ECB = 'ecb'
ECB_PLUS = 'ecbplus'
CONLL_FILES_DIR = '../conll_corefs/'
NPY_DATA_DIR = '/home/rldata/event_coreference_data/npy_data/'
VOCAB_DIR = '../vocab_files/'

VOCAB_FILE    = VOCAB_DIR + 'ecb_plus_vocabulary.txt'
POS_FILE      = VOCAB_DIR + 'ecb_plus_pos_tags.txt'
NER_FILE      = VOCAB_DIR + 'ecb_plus_ners.txt'
LEMMA_FILE    = VOCAB_DIR + 'ecb_plus_lemmas.txt'
WORD2VEC_FILE = VOCAB_DIR + 'ecb_plus_word_embeddings.txt'
TF_IDF_DISTRIBUTION_FILE = NPY_DATA_DIR + 'document_tf_idf_distribution.npy'
DOCUMENT_DATASET_FILE = NPY_DATA_DIR + 'document_train_test_set.npy'

TRAIN = range(1,36) + map(str, range(1,36))
TEST = range(36,46) + map(str, range(36,46))
_val = [2, 5, 12, 18, 21, 23, 34, 35]
VAL = set(_val).union(set(map(str, _val)))
ALL_TOPICS = set(TRAIN+TEST)

DOC_REPRESENTATION_SIZE = 100
LEMMA_ONE_HOT_SIZE = 500
STOPS = stopwords.words('english')

class EmptyClass:
    def __init__(self):
        pass

MOST_COMMON_LEMMAS = None

def is_stop_word(word):
    return unicode(word.lower().decode('utf-8')) in STOPS

def get_singletons_from_mlst(all_mentions, as_indexes=False):
    singletons = {m.coref_chain_id:0 for m in all_mentions}
    for m in all_mentions:
        if m.coref_chain_id in singletons:
            singletons[m.coref_chain_id] += 1
            if singletons[m.coref_chain_id] > 1:
                del singletons[m.coref_chain_id]
    singletons = {cid for cid in singletons.keys()} # set
    if as_indexes:
        idxs = set()
        for i,m in enumerate(all_mentions):
            if m.coref_chain_id in singletons:
                idxs.add(i)
        return idxs
    else:
        return singletons

def normalize_fname(fname):
    return extensionless(fname.replace('.txt',''))

def extensionless(fname):
    return fname[:-4]

def is_ecb(fname):
    return extensionless(fname).endswith(ECB)

def is_ecbplus(fname):
    return extensionless(fname).endswith(ECB_PLUS)

def get_category(fname):
    return ECB if is_ecb(fname) else ECB_PLUS

def get_topic(fname):
    return fname.split('_')[0]

def documents_in_same_topic(pair):
    return get_topic(pair[0]) == get_topic(pair[1])

def update_list_dict(dictionary, key, value):
    if dictionary.has_key(key):
        dictionary[key].append(value)
    else:
        dictionary[key] = [value]

def list_to_string(lst):
    s = ''
    for val in lst:
        s += val + ' '
    return s

def update_str_dict(dictionary, key, string):
    if dictionary.has_key(key):
        dictionary[key] += ' '+string
    else:
        dictionary[key] = string

def combine_dicts(d1, d2):
    newd = {}
    for k1 in d1:
        newd[k1] = d1[k1] + (d2[k1] if k1 in d2 else [])
    for k2 in d2:
        newd[k2] = d2[k2] if k2 not in d1 else newd[k2]
    return newd

def count_tag_vals(d):
    newd = {'ACTION':[]}
    for k in d:
        assert type(d[k]) is list
        if k.startswith('ACT') or k.startswith('NEG'):
            newd['ACTION'] += d[k]
        else:
            key = k.split('_')[0]
            if key not in newd:
                newd[key] = d[k]
            else:
                newd[key] += d[k]

    for key in newd:
        print key, ':', len(newd[key])

def all_in_list(test, main):
    for val in test:
        if val not in main:
            return False
    return True

def divide(top, bot):
    try:
        return top / bot
    except ZeroDivisionError:
        return 0.0

def get_intra_doc_iid(mid, fname):
    return 'INTRA_%s_%s'%(mid,extensionless(fname))

def load_vocabulary(fname=VOCAB_FILE):
    ret = []
    try:
        with open(fname, 'r') as f:
            for line in f:
                l = line[0:-1] # Don't include \n
                if '\r' in l:
                    l = l[0:-1]
                ret.append(l)
    except IOError:
        print 'Couldnt load file %s'%fname
    return ret

def load_word2vecs():
    f = open(WORD2VEC_FILE, 'r')
    dic = {}
    for line in f:
        str_vec = line.split(',')
        word = str_vec[0]
        str_vec = str_vec[1:len(str_vec)]
        dic[word] = []
        for string in str_vec:
            dic[word].append(round(float(string),8))
        dic[word] = np.array(dic[word])
    return dic

def build_most_common_lemmas_dict(tokens):
    global MOST_COMMON_LEMMAS
    tf_dict = collections.defaultdict(int)
    for tok in tokens:
        tf_dict[tok.lemma] += 1
    common_lems = dict(collections.Counter(tf_dict).most_common(LEMMA_ONE_HOT_SIZE-1))
    common_lems[''] = 0
    lemmas = common_lems.keys()
    MOST_COMMON_LEMMAS = {lemma:np.array([1.0 if lem==lemma else 0.0 for lem in lemmas]) for lemma in lemmas}

word2vec_dict = load_word2vecs()
empty_embedding = np.random.uniform(-1./600, 1./600, 300)
zeros_embedding = np.zeros(300)


