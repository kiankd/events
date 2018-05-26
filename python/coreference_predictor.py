import datetime
import simple_helpers as helpers
import numpy as np
import collections
from abc import ABCMeta, abstractmethod
from general_classes import MentionCoreferenceClusters, convert_to_conll
from feature_extraction import extract_mention_features, build_tfidf, get_doc_tfidf_rep
from ecb_doc_classes import Mention
from clustering import AgglomerativeClusterer
from corenlp_classes import empty_token
from clustering_evaluation import bcubed
from sklearn.metrics import adjusted_rand_score
from sklearn.metrics.pairwise import cosine_similarity

NEG_SAMPLE_MULTIPLIER = 5
SENTENCE_SEARCH_DISTANCE = 2

np.random.seed(1917)
_cos_sim_hash = {}
def cos_sim(hash1, hash2, v1, v2):
    # assume v1 and v2 are numpy arrays 1d
    key = tuple(sorted([hash1, hash2]))
    if not key in _cos_sim_hash:
        _cos_sim_hash[key] = 0.5 * (1 + cosine_similarity(v1.reshape(1,-1), v2.reshape(1,-1)))
    return _cos_sim_hash[key]

class CoreferencePredictor(object):
    __metaclass__ = ABCMeta

    def __init__(self, mention_coref_clusters, all_tokens, events_only=False, data_set='all', with_topics=False, topics=helpers.ALL_TOPICS):
        """
        :type mention_coref_clusters: MentionCoreferenceClusters
        :type all_tokens: list
        """
        self.mentions_dict = mention_coref_clusters.get_mentions_by_class(topics=topics)
        self.mentions_by_doc_dict = mention_coref_clusters.get_mentions_by_doc(topics=topics)
        self.gold_clusters = mention_coref_clusters.get_clusters(topics=topics)
        self.tokens = all_tokens
        self.token_hash_table = {(t.fname.replace('.txt',''), int(t.tid),):t for t in self.tokens} # hash a token table with filename and token id for quick access
        self.tokens_by_doc = {}
        for t in self.tokens:
            helpers.update_list_dict(self.tokens_by_doc, t.fname.replace('.txt',''), t)

        self.events_only = events_only
        self.data_set = data_set
        self.predictor_name = ''
        self.document_pairs = None
        self.positive_mention_pairs = None
        self.negative_mention_pairs = None
        self.with_topics = with_topics

        self.set_name()

        # only resetting token coreference values, NOT mentions
        for token in self.tokens:
            token.reset_coreference()

        for mention in self.itermentions():
            for token in mention.tokens:
                token.reset_coreference()

        self.new_coref_id = 1

    @abstractmethod
    def predict(self):
        pass

    @abstractmethod
    def set_name(self):
        pass

    def sort_tokens_into_docs(self, word_lists=False):
        d = {}
        for token in self.tokens:
            if not word_lists:
                helpers.update_list_dict(d, token.fname.replace('.txt',''), token)
            else:
                helpers.update_str_dict(d, token.fname.replace('.txt',''), token.word.decode('utf-8'))
        return d

    def itermentions(self):
        for mclass in self.mentions_dict:
            for mention in self.mentions_dict[mclass]:
                yield mention

    def get_mentions_by_topic(self, events_only=False, split_into_topics=True):
        """
        Returns a dictionary according to the topics. If split_into_topics=False, then
         just return all mentions in a single-entry dictionary: d['all'] = [...]
        """
        d = {}
        for doc in self.mentions_by_doc_dict:
            topic = helpers.get_topic(doc)
            if topic in d:
                d[topic] += [m for m in self.mentions_by_doc_dict[doc] if not events_only or m.is_event()]
            else:
                d[topic] = [m for m in self.mentions_by_doc_dict[doc] if not events_only or m.is_event()]

        if not split_into_topics: # if we aren't topic splitting, put all mentions in one set
            newd = {'all':[]}
            for mention_list in d.itervalues():
                newd['all'] += mention_list
            d = newd

        for key in d:
            d[key] = sorted(d[key], key=Mention.get_comparator_function())

        return d

    # get a single token
    def get_token(self, fname, tid):
        return self.token_hash_table[(fname, tid)]

    # get tokens between a certain index range
    def get_token_range(self, fname, start, end):
        tokens = []
        for i in xrange(start, end+1):
            try:
                tokens.append(self.get_token(fname, i))
            except KeyError:
                tokens.append(empty_token)
        return tokens

    # saves our model's predictions with token-based conll format
    def save_predictions_token_based(self, save=True):
        print '%s predictions:'%self.predictor_name
        fname = self.predictor_name + str(datetime.datetime.now())[5:].split('.')[0].replace(':','_').replace(' ','_')
        convert_to_conll(self.tokens,
                         helpers.CONLL_FILES_DIR+fname,
                         isgold=False,
                         events_only=self.events_only,
                         data_set=self.data_set,
                         save=save,
                         )

    # saves our model's predictions with mention-based conll format
    def save_predictions_mention_based(self, predicted_list, add_name='', gold_list=None):
        for lst in predicted_list, gold_list:
            fname = self.predictor_name + add_name + \
                    str(datetime.datetime.now())[5:].split('.')[0].replace(':', '_').replace(' ', '_')+'.response_conll'
            if lst is gold_list:
                fname = 'ecb_plus_events_test_mention_based.key_conll'
            if lst is not None:
                print fname+' --> Number of mentions: %d. Number of clusters: %d'%(len(lst), len(set(lst)))
                with open(helpers.CONLL_FILES_DIR+fname, 'w') as f:
                    file_name = 'ECB+/ecbplus_all'
                    f.write('#begin document (ECB+/ecbplus_all); part 000\n')
                    for val in lst:
                        f.write('%s (%s)\n'%(file_name, val))
                    f.write('\n#end document\n')

    # changes object id values to be consistent (?)
    def set_tokens_to_clusters(self, clusters, coref_class=''):
        for cluster in clusters.itervalues():
            for mention in cluster:
                mention.set_coref_id(self.new_coref_id, reset=True)
            self.new_coref_id += 1

    # delta filtering
    def delta_filter(self, clusters, delta, build_test_comparison=False):
        m_topics = self.get_mentions_by_topic(events_only=self.events_only, split_into_topics=False)
        all_mentions = []
        delta_clusters = collections.defaultdict(list)
        clust_key_idxs = collections.defaultdict(int)
        
        # iterate
        for mlst in m_topics.itervalues():
            mentions = sorted(mlst, key=Mention.get_comparator_function())
            mentions = [m for m in mentions if m.is_event() or not self.events_only]
            
            for i,mention in enumerate(mentions):
                # delta stuff
                clusters_to_check = clust_key_idxs[clusters[i]]
                
                for j in range(0, clusters_to_check+1):
                    clust_key = (clusters[i], j,) # define key for this mention
                    if clust_key in delta_clusters: # compare tfidf for lemma delta
                        my_tfidf = get_doc_tfidf_rep(mention.fname)
                        broken = False
                        for m in delta_clusters[clust_key]:
                            check_tfidf = get_doc_tfidf_rep(m.fname)
                            sim = cos_sim(mention.fname, m.fname, my_tfidf, check_tfidf)
                            if sim < delta:
                                broken = True
                                break
                        if not broken: # then it is similar enough, add it!
                            helpers.update_list_dict(delta_clusters, clust_key, mention)
                            break
                    
                    # then this was too different, so add a new cluster!
                    # this must only be reached if it wasn't added to a cluster
                    # otherwise it would have broken out of the loop
                    # also it cannot be in clusters if we are at end since
                    # otherwise we would not be at the end
                    if j == clusters_to_check:
                        delta_clusters[clust_key].append(mention)
                        clust_key_idxs[clusters[i]] += 1

            all_mentions += mentions

        test_comp = None
        if build_test_comparison:
            test_comp = [m.coref_chain_id for m in all_mentions]

        self.set_tokens_to_clusters({i:cluster for i,cluster in enumerate(delta_clusters.values())}) # this updates the coref_ids on all mentions
        return test_comp, [m.coref_chain_id for m in all_mentions]


class BaselineLemmaPredictor(CoreferencePredictor):
    def predict(self, build_test_comparison=True, delta=0.0):
        super(BaselineLemmaPredictor, self).predict()
        m_topics = self.get_mentions_by_topic(events_only=self.events_only,
                                             split_into_topics=False)
        all_mentions = []
        clusters = collections.defaultdict(list)
        lemma_keys_idxs = collections.defaultdict(int)
        for mlst in m_topics.itervalues():
            mentions = sorted(mlst, key=Mention.get_comparator_function())
            mentions = [m for m in mentions if m.is_event() or not self.events_only]
            for i,mention in enumerate(mentions):
                best_lemmas = []

                # set best_lemmas - for now it is by tuples, may be too restrictive..
                for token in mention.tokens:
                    if not helpers.is_stop_word(token.word):
                        best_lemmas.append(token.lemma)
                if len(best_lemmas) == 0:
                    best_lemmas = [t.lemma for t in mention.tokens]

                # HEAD LEMMA ONLY!!!! 
                best_lemmas = [mention.tokens[0].lemma]

                lemmas = tuple(best_lemmas+[mention.topic()])
                if not self.with_topics: # remove topic if we aint doin topics
                    lemmas = lemmas[:-1]

                # delta stuff
                clusters_to_check = lemma_keys_idxs[lemmas]
                
                for i in range(0, clusters_to_check+1):
                    lemmas_key = (lemmas, i,)
                    if lemmas_key in clusters: # compare tfidf for lemma delta
                        my_tfidf = get_doc_tfidf_rep(mention.fname)
                        broken = False
                        for m in clusters[lemmas_key]:
                            check_tfidf = get_doc_tfidf_rep(m.fname)
                            sim = cos_sim(mention.fname, m.fname, my_tfidf, check_tfidf)
                            if sim < delta:
                                broken = True
                                break
                        if not broken: # then it is similar enough, add it!
                            helpers.update_list_dict(clusters, lemmas_key, mention)
                            break
                    
                    # then this was too different, so add a new cluster!
                    # this must only be reached if it wasn't added to a cluster
                    # otherwise it would have broken out of the loop
                    # also it cannot be in clusters if we are at end since
                    # otherwise we would not be at the end
                    if i == clusters_to_check:
                        clusters[lemmas_key].append(mention)
                        lemma_keys_idxs[lemmas] += 1

            all_mentions += mentions

        test_comp = None
        if build_test_comparison:
            test_comp = [m.coref_chain_id for m in all_mentions]

        self.set_tokens_to_clusters({i:cluster for i,cluster in enumerate(clusters.values())}) # this updates the coref_ids on all mentions
        return test_comp, [m.coref_chain_id for m in all_mentions]

    def set_name(self, with_topics=False):
        self.predictor_name = 'BaselineLemma' + ('WithTopics' if self.with_topics else '')


class BaselineSingletonPredictor(CoreferencePredictor):

    def predict(self):
        super(BaselineSingletonPredictor, self).predict()
        class_clusters = {}
        singleton_index = 1
        for mention_class in self.mentions_dict:
            class_clusters[mention_class] = {}
            if (not self.events_only) or mention_class == 'ACTION':
                for mention in self.mentions_dict[mention_class]:
                    class_clusters[mention_class][singleton_index] = [mention]
                    singleton_index += 1
            self.set_tokens_to_clusters(class_clusters[mention_class], mention_class)

    def set_name(self):
        self.predictor_name = 'BaselineSingleton'


class BaselineAllCoreferentialPredictor(CoreferencePredictor):

    def predict(self):
        super(BaselineAllCoreferentialPredictor, self).predict()
        class_clusters = {}
        coreference_idx = 100 # use same idx for everything in this baseline
        for mention_class in self.mentions_dict:
            class_clusters[mention_class] = {coreference_idx:[]}
            if (not self.events_only) or mention_class == 'ACTION':
                for mention in self.mentions_dict[mention_class]:
                    class_clusters[mention_class][coreference_idx].append(mention)
            self.set_tokens_to_clusters(class_clusters[mention_class], mention_class)

    def set_name(self):
        self.predictor_name = 'BaselineAllCoreferential'


class ClusteringPredictor(CoreferencePredictor):
    def extract_features(self, mentions_by_topic):
        """
        :param mentions_by_topic: Dictionary of lists of mentions with keys as the topics
        :return: Returns feature matrix x and all of the mentions sorted
        """
        print 'Performing feature extraction...'
        # first get all mentions
        all_mentions = []
        for mlst in mentions_by_topic.itervalues():
            all_mentions += mlst

        # then we build TF-IDF matrix for the data set
        build_tfidf(self.tokens_by_doc, self.tokens)

        x = []
        for i, m in enumerate(all_mentions):
            if i % 180 ==0:
                print '%d out of %d...'%(i,len(all_mentions))
            if not self.events_only or m.is_event():
                x.append(extract_mention_features(m, self.tokens_by_doc[m.fname], self.mentions_by_doc_dict[m.fname], all_mentions))
        return x, all_mentions

    def predict(self,
                x = None,
                all_mentions = None,
                mentions_by_topic=None,
                threshold=0.35,
                metric='euclidean',
                link_function='single',
                build_test_comparison=False,
                split_into_topics=False,
                oracle_topics=False,
                use_lemma=False,
                lemma_pred=None,
                lemma_delta=0
                ):

        super(ClusteringPredictor, self).predict()
        clusters = []
        cluster_idxs = []

        if mentions_by_topic is None:
            mentions_by_topic = self.get_mentions_by_topic(events_only=self.events_only, split_into_topics=split_into_topics)
        if x is None:
            x, all_mentions = self.extract_features(mentions_by_topic)

        i = 0
        for mlst in mentions_by_topic.itervalues():
            if use_lemma:
                test_list, lemma_preds = lemma_pred.predict(build_test_comparison=True, delta=lemma_delta)
            else:
                lemma_preds = None

            c = AgglomerativeClusterer(x[i:len(mlst)+i], distance_metric=metric)
            clust_idxs = c.cluster(threshold, linktype=link_function, preset_predictions=lemma_preds)
            clusters += [[mlst[i] for i in clust] for clust in clust_idxs]
            cluster_idxs += [list(arr) for arr in clust_idxs]
            i = len(mlst)

        if build_test_comparison and not use_lemma:
            test_list = [m.coref_chain_id for m in all_mentions]

        if oracle_topics:
            new_clusts = []
            for cluster in clusters:
                cluster_divided_into_topics = collections.defaultdict(list)
                for mention in cluster:
                    cluster_divided_into_topics[mention.topic()].append(mention)
                for mlst in cluster_divided_into_topics.itervalues():
                    new_clusts.append(mlst)
            clusters = new_clusts

        self.set_tokens_to_clusters({i:cluster for i,cluster in enumerate(clusters)})
        predicted_list = [m.coref_chain_id for m in all_mentions]

        if build_test_comparison:
            return test_list, predicted_list
        else:
            return None, predicted_list

    # TODO: implement multi-processing for the threshold space search.
    def neural_predict(self, x, y, threshold_range, metric='cosine', link_function='single', rand_score=False, train_data=None, delta_filter=False, lemma_init=False, lemma_predictor=None):
        c = AgglomerativeClusterer(x, distance_metric=metric, train_data=train_data)
        best_score = (0., 0., 0.,)
        best_thresh = 0.
        best_delta = 0
        best_clusters = None
        all_scores = {}
        if (type(threshold_range) != float):
            for threshold in np.linspace(0.65, 1.0, threshold_range)[1:-1]:
                clust_idxs = c.cluster(threshold, linktype=link_function)
                clusters = np.zeros(len(x))
                for i,cluster in enumerate(clust_idxs):
                    clusters[cluster] = i

                # recall,precision,f1
                rpf1 = bcubed(y, clusters)
                all_scores[threshold] = rpf1

                # delta filtering for each threshold, sloooowww
                print "delta filter for threshold",threshold
                print "result with no delta",rpf1
                if delta_filter: 
                    for delta in np.linspace(0, 1, 101):
                        _, new_clusters = self.delta_filter(clusters, delta)
                        drpf1 = bcubed(y, new_clusters)
                        print delta, drpf1
                        if drpf1[2] > best_score[2]:
                            best_score = drpf1
                            best_thresh = threshold
                            best_delta = delta
                    print "best delta & thresh so far",best_delta,best_thresh
                    print "best result so far",best_score

        else:
            if not lemma_init:
                clust_idxs = c.cluster(threshold_range, linktype=link_function)
                clusters = np.zeros(len(x))
                for i,cluster in enumerate(clust_idxs):
                    clusters[cluster] = i
                best_score = bcubed(y, clusters)
                best_clusters = clusters

            # use lemma initialization and tune to a certain value for delta!
            if lemma_init:
                print('Doing lemma initialization tests!')
                best_score = 0
                best_params = None
                for delta in np.linspace(0.5, 1, 11):
                    tmp, lemma_preds = lemma_predictor.predict(build_test_comparison=False, delta=delta)
                    for thresh in np.linspace(0.6, 1, 21):
                        for mkt in np.linspace(0,0,1):#0.5, 1, 26): # optimize min keep thresh
                            clust_idxs = c.cluster(thresh, linktype=link_function, preset_predictions=lemma_preds, minimum_keeping_threshold=mkt)
                            clusters = np.zeros(len(x))
                            for i,cluster in enumerate(clust_idxs):
                                clusters[cluster] = i

                            # recall,precision,f1
                            score = bcubed(y, clusters)
                            print delta, thresh, mkt, score
                            if score[2] > best_score:
                                best_score = score[2]
                                best_params = (thresh, delta, mkt,)
                
                print 'Best score and best params:'
                print best_score, ': with - d=%0.2f, t=%0.2f'%(best_params[1], best_params[0])
                print 'with minimum keeping threshold: %0.2f'%best_params[2]
                best_thresh = best_params # return best_thresh as a tuple

            # delta filtering       
            if delta_filter:
                new_best_score = best_score
                best_delta = 0
                for delta in np.linspace(0, 1, 101):
                    _, new_clusters = self.delta_filter(best_clusters, delta)
                    rpf1 = bcubed(y, new_clusters)
                    print delta,rpf1
                    if rpf1[2] > new_best_score[2]:
                        new_best_score = rpf1
                        best_delta = delta

                print "BEST DELTA ",best_delta
                print "NEW BEST SCORE ",new_best_score


        # print "OLD BEST SCORE ",best_score
        return best_score, best_thresh, all_scores

    def set_name(self):
        self.predictor_name = 'ClusteringBasedPredictor'
