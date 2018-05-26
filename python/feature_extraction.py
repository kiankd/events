from ecb_doc_classes import Mention
from corenlp_classes import Token,empty_token
from sklearn.decomposition import PCA
import simple_helpers as helpers
import numpy as np
import collections

def get_word_embedding(token):
    try:
        return helpers.word2vec_dict[token.word]
    except KeyError:
        return helpers.empty_embedding

def get_lemma_one_hot(token):
    try:
        return helpers.MOST_COMMON_LEMMAS[token.lemma]
    except KeyError:
        return helpers.MOST_COMMON_LEMMAS['']


# stores doc features since we will get doc features multiple times
document_features = {}
_train_doc_indexes = {}
_val_doc_indexes = {}
_train_doc_reps = None
_val_doc_reps = None
_all_lemmas_for_tfidf = None
PCA_COMPRESSOR = None

def _tf_log_normalization(raw_freq):
    if raw_freq == 0:
        return 0.0
    return np.log(raw_freq) + 1.0

def build_tfidf(tokens_by_doc, all_tokens):
    """
    :param tokens_by_doc: Dict
    :param all_tokens: List
    """
    global _train_doc_reps, _val_doc_reps, _all_lemmas_for_tfidf, PCA_COMPRESSOR

    # set the reference for setting the indices
    tfidf_doc_indexes = _train_doc_indexes if PCA_COMPRESSOR is None else _val_doc_indexes

    # intialize lemma idf counts
    if _all_lemmas_for_tfidf is None:
        _all_lemmas_for_tfidf = {tok.lemma:0 for tok in all_tokens} # set of all lemmas by num docs containing them
    else: # reset the counts to zeros - we have to use same lemmas as train set to have consistent dimensionality
        for lem in _all_lemmas_for_tfidf:
            _all_lemmas_for_tfidf[lem] = 0

    # index them
    tfidf_lem_indexes = {}
    for i,lem in enumerate(_all_lemmas_for_tfidf.keys()):
        tfidf_lem_indexes[lem] = i

    # build the term frequency matrix (log-normalized)
    tf_mat = []
    for i,doc in enumerate(tokens_by_doc):
        doc_lemms = collections.defaultdict(int) # if key (the lemma) isn't there, gives 0. i.e., d['owqieruew'] = 0

        # count lemmas in doc
        for tok in tokens_by_doc[doc]:
            if tok.lemma in _all_lemmas_for_tfidf:
                doc_lemms[tok.lemma] += 1

        # increment lemma counts by doc, i.e. if a doc contains a lemma then increment
        for lemma_in_doc in doc_lemms:
            if doc_lemms[lemma_in_doc] > 0:
                _all_lemmas_for_tfidf[lemma_in_doc] += 1


        # update matrix with log-normalized term frequencies
        tf_mat.append([_tf_log_normalization(doc_lemms[lemma]) for lemma in _all_lemmas_for_tfidf.keys()])
        tfidf_doc_indexes[doc] = i

    tf_mat = np.array(tf_mat)

    # build idf matrix (+1 smoothed)
    total_docs = len(tokens_by_doc)
    idf_dict = {lem : np.log(total_docs/(1.+_all_lemmas_for_tfidf[lem])) for lem in _all_lemmas_for_tfidf.keys()}

    # multiply tf and idf
    for lem in _all_lemmas_for_tfidf.keys():
        tf_mat[:,tfidf_lem_indexes[lem]] *= idf_dict[lem] # lemmas are COLUMNS

    # now we have to use PCA to compress them to create doc representations and lemma representations
    if PCA_COMPRESSOR is None:
        PCA_COMPRESSOR = PCA(n_components=helpers.DOC_REPRESENTATION_SIZE)
        _train_doc_reps = PCA_COMPRESSOR.fit_transform(tf_mat)
    else:
        _val_doc_reps = PCA_COMPRESSOR.transform(tf_mat)

def get_doc_tfidf_rep(docname):
    """
    :param docname: String
    :return: A numpy array representing the (compressed) TFIDF representation of the document.
    """
    if _train_doc_indexes.has_key(docname):
        return _train_doc_reps[_train_doc_indexes[docname]]
    return _val_doc_reps[_val_doc_indexes[docname]]

def extract_mention_relational_features(mention, doc_mentions, all_mentions):
    sorted_mentions = sorted(doc_mentions, key=lambda m: int(m.get_token_id()))
    idx = 0
    for m in sorted_mentions:
        if m is mention:
            break
        else:
            idx += 1

    # represent the position of mention in document, specifically noting if it is first or last with separate features
    vec = [1. if idx==0 else 0., 1. if idx==len(sorted_mentions) else 0., float(idx)/len(sorted_mentions)]

    # count words and lemmas in common with each of the other mentions in doc based on prec/rec percent
    n = len(mention.tokens)
    words = [token.word for token in mention.tokens]
    lemmas = [token.lemma for token in mention.tokens]
    for mlist in (doc_mentions, all_mentions,):
        common_words_proportions = []
        common_lemmas_proportions = []
        for m in mlist:
            if mention is not m:
                words_in_common, lemmas_in_common = 0., 0.
                for tok in m.tokens:
                    if tok.word in words:
                        words_in_common += 1
                    if tok.lemma in lemmas:
                        lemmas_in_common += 1
                wprec = words_in_common / n
                wrec = words_in_common / len(m.tokens)
                lprec = lemmas_in_common / n
                lrec = lemmas_in_common / len(m.tokens)
                common_words_proportions.append( 0. if wprec==0 or wrec==0 else 2*wprec*wrec/(wprec+wrec) )
                common_lemmas_proportions.append( 0. if lprec==0 or lrec==0 else 2*lprec*lrec/(lprec+lrec) )
        vec.append(1. if len(mlist) > 0 else 0.)
        vec.append(0. if len(common_words_proportions)==0 else np.mean(common_words_proportions))
        vec.append(0. if len(common_lemmas_proportions)==0 else np.mean(common_lemmas_proportions))
    return np.array(vec)

# The global function here - get all features describing a mention, including surrounding mentions.
def extract_mention_features(mention, document_tokens, doc_mentions, all_mentions):
    emb_vec = extract_embedding_features(mention, document_tokens)
    doc_vec = get_doc_tfidf_rep(mention.fname)
    relational_vec = extract_mention_relational_features(mention, doc_mentions, all_mentions)
    return np.concatenate((emb_vec, doc_vec, relational_vec, ))

def extract_embedding_features(mention, doc_tokens):
    """
    :param mention: Mention
    :param doc_tokens: List
    """
    all_feats = np.array([])
    for extractor in (get_word_embedding, get_lemma_one_hot): # iterating over extraction functions, wemb and one-hots
        first_word_emb = extractor(mention.tokens[0])
        last_word_emb = extractor(mention.tokens[-1])
        mention_avg_emb = get_token_embeddings(mention.tokens, extractor=extractor)

        # get embeddings for the 2 preceding and 2 following tokens
        surrounding_emb = np.array([])
        start, end = mention.get_start_end_token_ids()
        start, end = int(start), int(end)
        for index in (start-2,start-1,end+1,end+2,):
            if 0 <= index < len(doc_tokens):
                surrounding_emb = np.concatenate((surrounding_emb, extractor(doc_tokens[index]),))
            else:
                surrounding_emb = np.concatenate((surrounding_emb, extractor(empty_token),))

        # avg embeddings for the 5 preceding and 5 following tokens
        preceding_emb = get_token_embeddings(doc_tokens[max(start-5,0) : start], extractor=extractor)
        following_emb = get_token_embeddings(doc_tokens[end+1 : min(len(doc_tokens)-1,end+6)], extractor=extractor)

        # avg embedding for the mention's sentence
        sentence_emb = get_token_embeddings(get_sent_tokens(mention.get_sentnum(), doc_tokens), extractor=extractor)

        all_feats = np.concatenate((all_feats, first_word_emb, last_word_emb, mention_avg_emb, surrounding_emb,
                                    sentence_emb, preceding_emb, following_emb,), axis=0)
    return all_feats


# gets embeddings for a set of tokens
def get_token_embeddings(tokens, key=None, extractor=get_word_embedding):
    if key in document_features:
        return document_features[key]
    try:
        return np.sum((extractor(t) for t in tokens), axis=0) / float(len(tokens))
    except ZeroDivisionError:
        return extractor(empty_token)


# uses binary search to find tokens corresponding to a certain sentence number
def get_sent_tokens(sentnum, doc_tokens):
    tokens = []

    s_bound, e_bound = 0, len(doc_tokens)
    mid = (s_bound + e_bound)/2
    crt_token = doc_tokens[mid]
    limit = np.log2(len(doc_tokens))
    count = 0
    while crt_token.sentnum != sentnum and s_bound < e_bound:
        if crt_token.sentnum > sentnum:
            e_bound = mid
        else:
            s_bound = mid
        mid = (s_bound + e_bound)/2
        crt_token = doc_tokens[mid]

        count += 1
        if count > limit + 15: # recursion limit, if gone past then something is wrong, do linear alg.
            break

    # linear alg that is called if binary search fucks up
    if count > limit + 15:
        for i in xrange(len(doc_tokens)):
            if doc_tokens[i].sentnum == sentnum:
                mid = i
                break

    # assuming the search found what we want...
    # left search
    for i in reversed(xrange(0, mid)):
        if doc_tokens[i].sentnum == sentnum:
            tokens.append(doc_tokens[i])
        else:
            break

    tokens = [t for t in reversed(tokens)] # re-reverse to be in sequential order

    # right search
    for i in xrange(mid, len(doc_tokens)):
        if doc_tokens[i].sentnum == sentnum:
            tokens.append(doc_tokens[i])
        else:
            break

    return tokens
