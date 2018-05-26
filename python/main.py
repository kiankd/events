from extract_ecbplus import get_ecb_data,convert_docs_to_txt
from extract_corenlp import get_core_data
from general_classes import Documents
from coreference_predictor import *
from document_predictor import PairwiseDocumentPredictor
from clustering_evaluation import bcubed
import simple_helpers as helpers

TOPICS = range(1,46)

def build(tops, events_only=False):
    ecbdoc_list, clusters = get_ecb_data(tops, get_all=False, clean=True, events_only=events_only)
    coredoc_list = get_core_data(tops)
    coredocs = Documents(coredoc_list)
    coredocs.convert(Documents(ecbdoc_list), clusters)
    # clusters.analyze()
    return coredocs

def analyze_orig_data(tops, get_all=False, clean=False) :
    ecbdoc_list, clusters = get_ecb_data(tops, get_all=get_all)
    docs = Documents(ecbdoc_list)
    docs.gold_mention_clusters = clusters
    docs.analyze(clean=clean)
    print

def load(events_only=False, data_set='all'):
    coredocs = Documents.load(events_only=events_only)
    helpers.build_most_common_lemmas_dict(coredocs.get_all_tokens(topics=helpers.TRAIN))
    coredocs.filter_for_data_set(data_set)
    if data_set != 'all':
        print 'WARNING THERE IS A SERIOUS BUG WHEN USING THIS FUNCTION TO FILTER DATA.\n' \
              'PLEASE DO IT THE OTHER WAY BY DECLARING THE topics VARIABLE IN A COREFERENCE-PREDICTORS __INIT__'
    return coredocs

def get_final_test_set(events_only):
    cd = load(events_only=events_only, data_set='all')
    pred = ClusteringPredictor(cd.gold_mention_clusters,
                               cd.get_all_tokens(topics=helpers.TEST),
                               events_only=events_only,
                               data_set='test',
                               topics=helpers.TEST)
    mentions_by_topic = pred.get_mentions_by_topic(events_only=events_only, split_into_topics=False)
    test_x, test_mentions = pred.extract_features(mentions_by_topic)
    return pred, test_x, test_mentions

def baseline_test_set_results(events_only=False):
    cd = load(events_only=events_only, data_set='test')
    # cd.convert_to_conll(helpers.CONLL_FILES_DIR, True, events_only=EVENTS_ONLY, data_set='test')
    for predictor_class in [BaselineSingletonPredictor, BaselineLemmaPredictor, BaselineAllCoreferentialPredictor]:
        print
        bslp = predictor_class(cd.get_clusters(), cd.get_all_tokens(), events_only=EVENTS_ONLY, data_set='test')
        # bslp.predict()
        # bslp.save_predictions(save=True)

def save_vocabulary():
    cd = load(events_only=False)
    words, pos, ner, lemmas = set(), set(), set(), set()
    for token in cd.get_all_tokens():
        words.add(token.word)
        pos.add(token.pos)
        ner.add(token.ner)
        lemmas.add(token.lemma)

    for fname,lst in [(helpers.VOCAB_FILE, words), (helpers.POS_FILE, pos), (helpers.NER_FILE, ner), (helpers.LEMMA_FILE, lemmas)]:
        with open(fname, 'w') as f:
            for val in lst:
                try:
                    f.write('%s\n'%val)
                except UnicodeEncodeError:
                    print 'DIDNT WRITE:',val


def test_doc_clusterer():
    train_cd = load(EVENTS_ONLY, data_set='train')
    test_cd  = load(EVENTS_ONLY, data_set='test')
    pdp = PairwiseDocumentPredictor(train_cd.get_doc_to_tokens_dict(), test_cd.get_doc_to_tokens_dict())
    pdp.build_features()

if __name__ == '__main__':
    EVENTS_ONLY = True
    # cd = build(TOPICS, events_only=EVENTS_ONLY)
    # cd.serialize(events_only=EVENTS_ONLY)
    # exit(0)

    # cd = load(data_set='all')

    # pred = BaselineLemmaPredictor(cd.gold_mention_clusters,
    #                               cd.get_all_tokens(topics=helpers.TEST),
    #                               topics=helpers.TEST,
    #                               events_only=True,
    #                               data_set='test',
    #                               with_topics=False)
    # predicted = pred.predict()
    # pred.save_predictions_mention_based(predicted)
    cd = load(events_only=False)
    gold = None
    for met in ['cosine']:
        for thresh in [0.7]:
            pred = ClusteringPredictor(cd.gold_mention_clusters, cd.get_all_tokens(topics=helpers.VAL), events_only=True, data_set='val', topics=helpers.VAL)
            print 'Predicting...'
            g, predicted = pred.predict(threshold=thresh,
                                        metric=met,
                                        link_function='single',
                                        build_test_comparison=gold is None,
                                        split_into_topics=False
                                        )
            if gold is None:
                gold = g

            print thresh, ':', bcubed(gold, predicted)

            # pred.save_predictions_mention_based(predicted, gold_list=gold)

    print
