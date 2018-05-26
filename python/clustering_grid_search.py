from itertools import product, combinations
from multiprocessing import Process
from coreference_predictor import ClusteringPredictor
from main import load
from clustering_evaluation import bcubed
import simple_helpers as helpers
import numpy as np
import copy

GOLD = 'GOLD'

def all_combinations(lst):
    ret = set()
    for combo_size in range(1, len(lst) + 1):
        for subset in combinations(lst, combo_size):
            ret.add(tuple(subset))
    return ret

def grid_search(save_file, d, pred, x, mentions_by_topic, all_mentions):
    predictions = {}
    params = set([combo for combo in product(*d.values())])

    print 'We have %d combinations to try for the grid search.'%len(params)
    gold = None
    for combo in params:
        thresh, met, link = combo[0], combo[1], combo[2]
        g, predicted = pred.predict(x=x, all_mentions=all_mentions,
                                    mentions_by_topic=mentions_by_topic,
                                    threshold=thresh,
                                    metric=met,
                                    build_test_comparison=gold is None,
                                    split_into_topics=False)
        if g is not None:
            gold = g
            predictions[GOLD] = gold

        predictions[combo] = predicted
        print combo

    np.save(helpers.NPY_DATA_DIR+save_file, np.array([predictions]))

def hash_d(d):
    return tuple(d.items())

def load_results(fname):
    try:
        data = np.load(fname)
    except IOError:
        data = np.load(helpers.NPY_DATA_DIR+fname)
    return data[0]

def combine_results(lst_of_file_names, save_file=None):
    d = {fname:load_results(fname) for fname in lst_of_file_names}
    combined_predictions = {}
    for predictions_dict in d.itervalues():
        for combo in predictions_dict.iterkeys():
            if not combined_predictions.has_key(combo):
                combined_predictions[combo] = predictions_dict[combo]

    file_additive_name = '_plus_'
    new_name = helpers.NPY_DATA_DIR
    for s in lst_of_file_names:
        s = s.lstrip(helpers.NPY_DATA_DIR)
        new_name += s.rstrip('.npy') + file_additive_name

    if save_file:
        new_name = helpers.NPY_DATA_DIR+save_file
    else:
        new_name = new_name[:-1*len(file_additive_name)]+'.npy'

    print 'Combined into file: ' + new_name
    np.save(new_name, np.array([combined_predictions]))

def analyze_results(predictions):
    # recalls = {key:value for key,value in recalls.items() if key[2] == 'cosine'}
    # precisions = {key:value for key,value in precisions.items() if key[2] == 'cosine'}
    recalls, precisions, fscores = {}, {}, {}
    for key in predictions:
        # if key[3] != ('lemma',):
        #     continue
        r,p,f1 = bcubed(predictions[GOLD], predictions[key])
        recalls[key] = r
        precisions[key] = p
        fscores[key] = f1
        print key,r,p,f1

    list_names = {hash_d(recalls):'Best Recall Score',
                  hash_d(precisions): 'Best Precision Score',
                  hash_d(fscores):'Best F1 Score'}

    for d in (recalls, precisions, fscores):
        di = copy.deepcopy(d)
        print '\nTop 10 param sets for %s:'%list_names[hash_d(d)]
        for i in range(25):
            params = key_with_max_value(di)
            print 'Number %d: Params = %s Scores = %0.5f %0.5f %0.5f'\
                  %(i+1, str(params), recalls[params], precisions[params], fscores[params])
            del di[params]

def key_with_max_value(d):
    values = list(d.values())
    return list(d.keys())[values.index(max(values))]

def multi_processing_gs(n_threads=8, base_name='results'):
    threshs = [0.025*i for i in xrange(5,36)]
    threshs_split = np.array_split(np.array(threshs), n_threads) # multi-threading by splitting the thresholds as evenly as possible.

    # initializing our model and features
    topics = helpers.VAL
    coredocs = load(data_set='all')
    pred = ClusteringPredictor(coredocs.gold_mention_clusters, coredocs.get_all_tokens(topics), events_only=True,
                               data_set='val', topics=topics)
    mentions_by_topic = pred.get_mentions_by_topic(events_only=True, split_into_topics=False)
    x, all_mentions = pred.extract_features(mentions_by_topic)

    # setting up multiproc
    my_processes = []
    for i,thresh_array in enumerate(threshs_split):
        fname = base_name+str(i)+'.npy'
        d = {
            'threshold':list(thresh_array),
            'metric': ['cosine','euclidean','manhattan'],
            'link_function': ['single','complete'],
            }
        my_processes.append(Process(target=grid_search, args=(fname, d, pred, x, mentions_by_topic, all_mentions, )))

    for p in my_processes:
        p.start()

    for p in my_processes:
        p.join()

    # combine_results(['npy_data/%s%s.npy'%(base_name, i) for i in xrange(len(threshs_split))])

if __name__ == '__main__':
    nthreads = 15
    base = 'embedding_features_agglom_grid_search_NOTOPICS_split'

    # multi_processing_gs(n_threads=nthreads, base_name='embedding_features_agglom_grid_search_NOTOPICS_split')
    # files = [base + str(i) + '.npy' for i in xrange(nthreads)]
    # combine_results(files, save_file='embedding_features_agglom_grid_search_NOTOPICS.npy')
    analyze_results(load_results('embedding_features_agglom_grid_search_NOTOPICS.npy'))


    # combine_results(['npy_data/grid_search_results2.npy','npy_data/grid_search_results3.npy'])
    # rr, rm = load_results('npy_data/grid_search_results2_plus_grid_search_results3.npy')
    # analyze_results(rr, rm)
    # for fname in ('npy_data/grid_search_results1.npy', 'npy_data/grid_search_results2.npy'):
    #     print '\n----------------------------'
    #     print fname
    #     data = np.load(fname)
    #     rr, rm = data[0], data[1]
    #     analyze_results(rr, rm)
