import numpy as np
import time
from simple_helpers import load_vocabulary,WORD2VEC_FILE

def load_bin_vec(fname, vocab, vocab_length):
    """
    Loads 300x1 word vecs from Google (Mikolov) word2vec
    """
    word_vecs = {}
    with open(fname, "rb") as f:
        header = f.readline()
        vocab_size, layer1_size = map(int, header.split())
        binary_len = np.dtype('float32').itemsize * layer1_size
        count = 0
        for line in xrange(vocab_size):
            if count >= vocab_length:
                return word_vecs
            word = []
            while True:
                ch = f.read(1)
                if ch == ' ':
                    word = ''.join(word)
                    break
                if ch != '\n':
                    word.append(ch)
            if word in vocab:
                count += 1
                if count%100 == 0:
                    print count,word
                vocab.remove(word)
                word_vecs[word] = np.fromstring(f.read(binary_len), dtype='float32')
            else:
                f.read(binary_len)

    return word_vecs

def get_vectors(vocab):
    # Mac Bin location
    bin_location = '/Users/kian/Documents/HONOR/project/word2vec/GoogleNews-vectors-negative300.bin'

    # PC Bin location
    #bin_location = 'C:\Users\Kian\Sura\project\word2vec\GoogleNews-vectors-negative300.bin'
    return load_bin_vec(bin_location, vocab, len(vocab))#extractdatafromfile(EACH_UNIQUE_WORD_FILE))

start = time.clock()
print 'Getting vectors...'
vocab = load_vocabulary()
print 'Vocabulary size',len(vocab)
dics = get_vectors(vocab)
print 'Got vectors!'
print 'Time taken: ',(time.clock()-start)
print 'Writing new file...'
f = open(WORD2VEC_FILE, 'w')
for k in dics:
    f.write('%s'%k)
    for val in dics[k]:
        f.write(',%f'%val)
    f.write('\n')
f.close()
print 'Finished writing new file!'
