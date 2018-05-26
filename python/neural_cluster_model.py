from main import load, get_final_test_set
from coreference_predictor import ClusteringPredictor,BaselineLemmaPredictor
from clustering_evaluation import bcubed
from results_analysis import build_results_graphs, build_clustering_graph
from singleton_predictor import SingletonPredictor
from sklearn.decomposition import PCA
from sklearn.metrics import f1_score, precision_score, recall_score
from copy import deepcopy
from scipy.misc import comb as combinations
import simple_helpers as helpers
import numpy as np
import random
import time
import theano
import theano.tensor as T
import lasagne
import argparse

HOURGLASS = 'hourglass'
SINGLE_LAYER = 'single_layer'

parser = argparse.ArgumentParser(description='Do neural-network based clustering for event coreference.')
parser.add_argument('results_file', help='The file we save the results into.')
parser.add_argument('-cl', help='Use this argument if you want to put singletons into one class for training.', action='store_true')
parser.add_argument('-val_cl', action='store_true')
parser.add_argument('-lr', default=0.0001, help='Set learning rate with this option.')
parser.add_argument('-lam1', default=0.0, help='Set the lambda 1 regularization parameter.')
parser.add_argument('-lam2', default=0.0, help='Set the lambda 2 regularization parameter.')
parser.add_argument('-tr', default=10, help='Set threshold range to search for clustering learned mention representations.')
parser.add_argument('-epochs', default=100, help='Set number of epochs.')
parser.add_argument('-hl', default=100, help='Set number of hidden neurons in the middle layer.')
parser.add_argument('-opt_thresh', default=0.0)
parser.add_argument('-analyze', action='store_true')
parser.add_argument('-test_pred', action='store_true')
parser.add_argument('-s', action='store_true')
parser.add_argument('-baseline', action='store_true')
parser.add_argument('-pca', action='store_true')
parser.add_argument('-pca_tests', action='store_true')
parser.add_argument('-permute', action='store_true')
parser.add_argument('-rms', action='store_true')
parser.add_argument('-sing_pred', action='store_true')
parser.add_argument('-network', default=HOURGLASS)
parser.add_argument('-z', action='store_true')
parser.add_argument('-T', action='store_true')
parser.add_argument('-t', action='store_true')
parser.add_argument('-LEMMA', action='store_true')
args = parser.parse_args()

SINGLETON_CLASS = 0
BASELINE_THRESH = 0.657
BATCHSIZE = 272
THRESH_SEARCH_RANGE = 101
LAMBDA1 = float(args.lam1)
LAMBDA2 = float(args.lam2)

random.seed(1917)
LEARNING_RATE = float(args.lr)
EPOCHS = int(args.epochs)
THRESH_RANGE = int(args.tr)
HOURGLASS_LAYER_SIZE = int(args.hl)
RESULTS_FILE = args.results_file.rstrip('.npy') + '.npy'
PUT_SINGLETONS = args.cl
PUT_VAL_SINGELTONS = args.val_cl
REMOVE_TRAIN_SINGLETONS = args.rms
ANALYZE_RESULTS = args.analyze
LEMMA_PRED = args.LEMMA
GET_TEST_PREDICTIONS = args.test_pred
OPTIMAL_THRESHOLD = float(args.opt_thresh)
MAKE_BASELINE = args.baseline
MAKE_BASELINE_PCA = args.pca
PERMUTE_Y = args.permute
SINGLETON_PREDICTOR = args.sing_pred
NETWORK = args.network
SAVE_PRED = args.s
PCA_TESTS = args.pca_tests
ZERO_CCE = args.z
ORACLE_TOPICS = args.T
TOPIC_SPLIT = False #args.t
DELTA_FILTER = False
LEMMA_INIT = True

# basic neural network builder
def build_single_layer_nn(x_shape, num_classes, batchsize=None, input_var=None):
    print 'Building single layer network...'
    l_in = lasagne.layers.InputLayer(shape=(None, x_shape[0]), input_var=input_var) # set to None for now
    l_hid1 = lasagne.layers.DenseLayer(
        l_in, num_units=HOURGLASS_LAYER_SIZE,
        nonlinearity=lasagne.nonlinearities.rectify,
        W=lasagne.init.GlorotUniform())
    l_out = lasagne.layers.DenseLayer(
        l_hid1, num_units=num_classes,
        nonlinearity=lasagne.nonlinearities.softmax)
    return l_hid1, l_out

# deeper network
def build_multi_layer_nn(x_shape, num_classes, batchsize=None, input_var=None):
    l_in = lasagne.layers.InputLayer(shape=(None, x_shape[0]), input_var=input_var)  # set to None for now
    l_in_drop = lasagne.layers.DropoutLayer(l_in, p=0.5)  # dropout between input and hidden
    l_hid1 = lasagne.layers.DenseLayer(
        l_in_drop, num_units=2500,
        nonlinearity=lasagne.nonlinearities.rectify,
        W=lasagne.init.GlorotUniform())

    l_hid1_drop = lasagne.layers.DropoutLayer(l_hid1, p=0.5)  # dropout between hidden layers
    l_hid2 = lasagne.layers.DenseLayer(
        l_hid1_drop, num_units=1500,
        nonlinearity=lasagne.nonlinearities.rectify,
        W=lasagne.init.GlorotUniform())

    l_hid2_drop = lasagne.layers.DropoutLayer(l_hid2, p=0.5) # dropout between hidden layers layers
    l_hid3 = lasagne.layers.DenseLayer(
        l_hid2_drop, num_units=1000,
        nonlinearity=lasagne.nonlinearities.rectify,
        W=lasagne.init.GlorotUniform())

    l_out = lasagne.layers.DenseLayer(
        l_hid3, num_units=num_classes,
        nonlinearity=lasagne.nonlinearities.softmax)

    return l_hid3, l_out

# hourglass network
def build_hourglass_nn(x_shape, num_classes, batchsize=None, input_var=None):
    l_in = lasagne.layers.InputLayer(shape=(None, x_shape[0]), input_var=input_var)  # set to None for now
    l_in_drop = lasagne.layers.DropoutLayer(l_in, p=0.25)  # dropout between input and hidden
    l_hid1 = lasagne.layers.DenseLayer(
        l_in_drop, num_units=1000,
        nonlinearity=lasagne.nonlinearities.rectify,
        W=lasagne.init.GlorotUniform())

    l_hid1_drop = lasagne.layers.DropoutLayer(l_hid1, p=0.25)  # dropout between hidden layers
    hourglass_layer = lasagne.layers.DenseLayer(
        l_hid1_drop, num_units=HOURGLASS_LAYER_SIZE,
        nonlinearity=lasagne.nonlinearities.rectify,
        W=lasagne.init.GlorotUniform())

    hourglass_drop = lasagne.layers.DropoutLayer(hourglass_layer, p=0.25) # dropout between hidden layers layers
    l_hid3 = lasagne.layers.DenseLayer(
        hourglass_drop, num_units=1000,
        nonlinearity=lasagne.nonlinearities.rectify,
        W=lasagne.init.GlorotUniform())

    l_out = lasagne.layers.DenseLayer(
        l_hid3, num_units=num_classes,
        nonlinearity=lasagne.nonlinearities.softmax)

    return hourglass_layer, l_out

def build_args_nn(x_train, y_train, batchsize, input_var):
    if NETWORK==SINGLE_LAYER:
        return build_single_layer_nn(x_train[0].shape, max(y_train) + 1, batchsize=batchsize, input_var=input_var)
    else:
        return build_hourglass_nn(x_train[0].shape, max(y_train) + 1, batchsize=batchsize, input_var=input_var)

####################### ######################
def prepare_model(x_train, y_train, batchsize, params=None):
    input_var = T.matrix('inputs')
    target_var = T.ivector('targets')
    same_cluster_indices_matrix = T.matrix('same_clusters')
    diff_cluster_indices_matrix = T.matrix('diff_clusters')

    # prepare network
    print '\nPreparing the model with primary hidden layer size %d...'%HOURGLASS_LAYER_SIZE
    print 'X-shape = %d, Num_classes = %d, num_samples = %d'%(x_train[0].shape[0], max(y_train), len(x_train))
    representation_layer, network = build_args_nn(x_train, y_train, batchsize, input_var)

    # loss stuff
    prediction = lasagne.layers.get_output(network)
    get_representations = lasagne.layers.get_output(representation_layer, inputs=input_var, deterministic=True)
    loss = lasagne.objectives.categorical_crossentropy(prediction, target_var)

    if LAMBDA1 == LAMBDA2 == 0.0:
        loss = loss.mean()
    else:
        representations = get_representations
        dot_prods = T.dot(representations, representations.T) # X times X.T
        diag = T.sqrt(T.diagonal(dot_prods)) # sqrt(||ri||^2) = ||ri||

        norms = T.outer(diag, diag.T)
        distances = 0.5*(1 -(dot_prods * (1./norms))) # d(a,b) = 1/2 (1 - dot(a,b) / (||a||*||b||))

        # we want the first sum to be as close to zero as possible, so we add it to the loss.
        # we want the second sum to be as close to 1 as possible, so we want LAMBDA2 * (1 - sum2)
        # to be as close to zero as possible, thus adding that difference to the overall loss.
        loss = loss.mean() \
               + (LAMBDA1 * T.sum(same_cluster_indices_matrix * distances)) \
               + (LAMBDA2 * (1.0 - T.sum(diff_cluster_indices_matrix * distances)))

    # for loading/building the parameters
    if not params:
        params = lasagne.layers.get_all_params(network, trainable=True)
    else:
        lasagne.layers.set_all_param_values(network, params)
        params = lasagne.layers.get_all_params(network, trainable=True)

    updates = lasagne.updates.adam(loss, params, learning_rate=LEARNING_RATE)

    # the final keys
    train_function = theano.function([input_var, target_var, same_cluster_indices_matrix, diff_cluster_indices_matrix],
                                     loss, updates=updates, allow_input_downcast=True, on_unused_input='ignore')
    convert_to_numpy_function = theano.function([input_var], get_representations, allow_input_downcast=True)

    # theano.printing.debugprint(train_function.maker.fgraph.outputs[0])

    return network, train_function, convert_to_numpy_function

# iterator
def iterate_minibatches(inputs, targets, batchsize, shuffle=False):
    assert len(inputs) == len(targets)
    if shuffle:
        c = list(zip(inputs, targets))
        random.shuffle(c)
        inputs, targets = zip(*c)

    for i in xrange(len(inputs)/batchsize): # cut off the elements that don't fit by rounding down
        start, end = i*batchsize, (i+1)*batchsize
        targs = np.array(targets[start:end])
        n = len(targs)
        if not (LAMBDA1 == LAMBDA2 == 0.0):
            targs = targs.reshape((n,1))
            bool_mat = targs == targs.T # pairwise class equality boolean matrix, nxn
            bool_mat *= (targs != SINGLETON_CLASS)  # make singletons != to each other (since otherwise they have same class)
            np.fill_diagonal(bool_mat, True) # above will make the diagonal False for singletons, so reset to True
            total_same_class_pairs = (bool_mat.sum() - n)/2.
            total_diff_class_pairs = combinations(n, 2) - total_same_class_pairs

            if total_same_class_pairs == 0:
                print 'No pairs of same class!'
            if total_diff_class_pairs == 0:
                print 'No pairs from different classes!'

            # need these to do proper summations over the distances
            same_clust_mat = bool_mat*(1.0/total_same_class_pairs)
            diff_clust_mat = (bool_mat==False)*(1.0/total_diff_class_pairs)
            np.fill_diagonal(same_clust_mat, 0)
            np.fill_diagonal(diff_clust_mat, 0)
            same_clust_mat = np.triu(same_clust_mat)
            diff_clust_mat = np.triu(diff_clust_mat)
        else:
            same_clust_mat = np.zeros((n,n))
            diff_clust_mat = np.zeros((n,n))

        targs = targs.reshape(n)
        yield inputs[start:end], tuple(targs), same_clust_mat, diff_clust_mat

# permute the y-data class names, preserves order
def permute_y(ydata):
    new_classes = range(max(ydata)+1)
    random.shuffle(new_classes) # now class j becomes new_classes[j]
    permuted_y = []
    for i in xrange(len(ydata)):
        permuted_y.append(new_classes[ydata[i]]) # ydata[i] is the class of sample i
    return permuted_y

#################### main function #####################
def prepare_and_run_model(train_pred, val_pred,
                          threshold_range,
                          x_train, y_train,
                          x_val, y_val,
                          batchsize=1, epochs=50):

    network, train_function, representation_building_function = prepare_model(x_train, y_train, batchsize)

    # running the model
    results = [[LEARNING_RATE, EPOCHS]]
    best_representation = None
    best_val_f1 = 0.0
    print '\nBeginning training...'
    for epoch in xrange(epochs):
        train_err = 0.0
        train_batches = 0.0
        start_time = time.time()

        if PERMUTE_Y:
            y_train = permute_y(y_train)

        for inputs_batch, targets_batch, same_clust_mat, diff_clust_mat in iterate_minibatches(x_train, y_train, batchsize, shuffle=True):
            train_err += train_function(inputs_batch, targets_batch, same_clust_mat, diff_clust_mat)
            train_batches += 1

        # now time to evaluate with our agglomerative clusterer on the representations
        new_val_x = representation_building_function(x_val)
        # new_train_x = representation_building_function(x_train)

        # print 'Clustering validation...'
        val_accuracy, threshold = cluster(val_pred, new_val_x, y_val, threshold_range, train_data=None)

        # print 'Clustering train...'
        train_accuracy = (0.0, 0.0, 0.0,)
        # train_accuracy = cluster(train_pred, new_train_x, y_train)

        print("Epoch {} of {} took {:.3f}s".format(
            epoch + 1, epochs, time.time() - start_time))
        print "Validation threshold was found to be: ",str(threshold)
        print("  training loss:\t\t{:.6f}".format(train_err / train_batches))

        if train_accuracy != (0.0, 0.0, 0.0,):
            print("  train recall:\t\t\t{:.6f}".format(train_accuracy[0]))
            print("  train precision:\t\t{:.6f}".format(train_accuracy[1]))
            print("  train f1:\t\t\t{:.6f}".format(train_accuracy[2]))

        print("  validation recall:\t\t{:.6f}".format(val_accuracy[0]))
        print("  validation precision:\t\t{:.6f}".format(val_accuracy[1]))
        print("  validation f1:\t\t{:.6f}".format(val_accuracy[2]))
        print
        results.append([epoch, train_err/train_batches, train_accuracy, val_accuracy])

        if val_accuracy[2] > best_val_f1:
            best_val_f1 = val_accuracy[2]
            best_representation = lasagne.layers.get_all_param_values(network)

    print 'Best acc:',best_val_f1
    return results, best_representation, representation_building_function

### clustering ###
def cluster(predictor, x, y, thresh_range, get_all_scores=False, train_data=None, delta_filter=False, lemma_init=False):
    if thresh_range <= 0:
        return (0., 0., 0.,), 0.

    print 'Clustering...'
    score, threshold, all_scores = predictor.neural_predict(x, y, thresh_range, metric='cosine',
                                                            link_function='single', rand_score=False, 
                                                            train_data=train_data, delta_filter=delta_filter,
                                                            lemma_init=LEMMA_INIT, lemma_predictor=get_lemma_predictor(helpers.VAL))

    if not get_all_scores:
        return score, threshold
    else:
        return score, threshold, all_scores

###################   results analysis   ###################
def analyze_neural_results(fname, x_train, y_train, x_val, y_val, clusterer, get_test_set_predictions=False, oracle_topics=False):
    data = np.load(fname)
    if len(data) == 3:
        results_list, best_model_params, rep_function = data[0], data[1], data[2]
    else:
        results_list, best_model_params = data[0], data[1]
        rep_function = None

    lr, num_epochs = results_list[0]

    epochs = []
    train_losses = []
    validation_accs = []
    for i in xrange(1,len(results_list)):
        epoch, train_loss, train_acc_NULL, val_acc = results_list[i]
        epochs.append(epoch)
        train_losses.append(train_loss)
        validation_accs.append(val_acc[2]) #for now just getting the F1s

    maxidx = 0
    best = 0
    for i in xrange(len(validation_accs)):
        if validation_accs[i] > best:
            best = validation_accs[i]
            maxidx = i
    print 'Epoch with optimal validation accuracy: ',epochs[maxidx]
    print 'The accuracy was:',best

    save_name = '%s_%s_%s_optimalEpoch%s_results' \
                % (fname.rstrip('.npy'), str(lr).replace('.', ''),
                   'clustered_singletons' if PUT_SINGLETONS else '',
                   str(epochs[maxidx]))

    # build_results_graphs(save_name, epochs, train_losses, validation_accs)

    if get_test_set_predictions:
        test_set_predictions(x_train, x_val, y_val, clusterer, save_name, rep_function, best_model_params, oracle_topics=oracle_topics)


def analyze_singleton_accuracy(gold, predictions):
    assert len(gold) == len(predictions)
    ### comparing singleton detection ###
    gold_sing_idxs = set()
    pred_sing_idxs = set()
    for lst in [gold, predictions, ]:
        sing_idxs = set()
        nonsing_idxs = set()
        for i in xrange(len(lst)):
            for j in xrange(len(lst)):
                if i != j:
                    if lst[i] == lst[j]:  # not singleton
                        nonsing_idxs.add(i)
                        nonsing_idxs.add(j)
            if i not in nonsing_idxs:
                sing_idxs.add(i)

        if len(gold_sing_idxs) == 0:
            gold_sing_idxs = sing_idxs.copy()
        else:
            pred_sing_idxs = sing_idxs.copy()

    print "PERCENT OF SINGLETONS: %0.3f"%(len(gold_sing_idxs) / float(len(gold)))

    gold_sing = [1 if i in gold_sing_idxs else 0 for i in xrange(len(gold))]
    pred_sing = [1 if i in pred_sing_idxs else 0 for i in xrange(len(predictions))]

    print 'Accuracy with respect to the correct identification of mentions as singletons:'
    print 'R: %0.4f,   P: %0.4f,   F1: %0.4f' % (
    recall_score(gold_sing, pred_sing), precision_score(gold_sing, pred_sing), f1_score(gold_sing, pred_sing))

    gold_non_sing = [0 if i in gold_sing_idxs else 1 for i in xrange(len(gold))]
    pred_non_sing = [0 if i in pred_sing_idxs else 1 for i in xrange(len(predictions))]

    print 'Accuracy with respect to the correct identification of mentions as NOT being singletons:'
    print 'R: %0.4f,   P: %0.4f,   F1: %0.4f' % (
    recall_score(gold_non_sing, pred_non_sing), precision_score(gold_non_sing, pred_non_sing), f1_score(gold_non_sing, pred_non_sing))

    # make coref chains without singletons
    non_sing_gold = []
    non_sing_pred = []
    for i,val in enumerate(gold_non_sing):
        if val == 1:
            non_sing_gold.append(gold[i])
            non_sing_pred.append(predictions[i])
    results = bcubed(non_sing_gold, non_sing_pred)
    print 'B3 results obtained after removing all GOLD singletons:'
    print results


def analyze_within_vs_cross_accuracy(gold, predictions, mentions):
    assert len(gold) == len(predictions) == len(mentions)
    
    # see what results are when singletons are removed
    for i, mention in enumerate(mentions):
        pass

def test_set_predictions(x_train, x_val, y_val, clusterer, save_name, rep_function, model_params, oracle_topics=False):
    global BASELINE_THRESH
    input_var = T.matrix('inputs')
    representation_layer, network = build_args_nn(x_train, y_train, batchsize=None, input_var=input_var)
    lasagne.layers.set_all_param_values(network, model_params)
    rep_function = theano.function([input_var], lasagne.layers.get_output(representation_layer, inputs=input_var, deterministic=True),
                                   allow_input_downcast=True)


    THRESHOLD = None#0.84#0.84
    DELTA = None#0.66#0.89


    if THRESHOLD is None or DELTA is None:
        new_val_x = rep_function(x_val)    
        if OPTIMAL_THRESHOLD <= 0.0:
            score, threshold, all_scores = cluster(clusterer, new_val_x, y_val, THRESH_SEARCH_RANGE, get_all_scores=True, delta_filter=DELTA_FILTER, lemma_init=LEMMA_INIT)
            print score, threshold
        else:
            threshold = OPTIMAL_THRESHOLD
            score, threshold, all_scores = cluster(clusterer, new_val_x, y_val, OPTIMAL_THRESHOLD, get_all_scores=True, delta_filter=DELTA_FILTER, lemma_init=LEMMA_INIT)
            print "\nBEST SCORE AND THRESHOLD:"
            print score, threshold
        THRESHOLD = threshold[0]
        print('USING THRESHOLD %0.2f, DELTA %0.2f'%(THRESHOLD, DELTA))
        exit(0) 


    print '\nSaving and clustering for test set results!'
    test_pred, test_x, test_mentions = get_final_test_set(events_only)
    new_test_x = rep_function(test_x)
    print('USING THRESHOLD %0.2f, DELTA %0.2f'%(THRESHOLD, DELTA))
    gold, predictions = test_pred.predict(new_test_x, test_mentions, threshold=THRESHOLD, metric='cosine',
                                          link_function='single', oracle_topics=oracle_topics, split_into_topics=TOPIC_SPLIT,
                                          build_test_comparison=True, use_lemma=LEMMA_INIT, lemma_pred=get_lemma_predictor(helpers.TEST),
                                          lemma_delta=DELTA)

    analyze_singleton_accuracy(gold, predictions)
    # analyze_within_vs_cross_accuracy(gold, predictions, test_mentions)


    if SAVE_PRED:
        try:
            test_pred.save_predictions_mention_based(predictions, add_name=save_name.split('/')[-1])
        except IOError:
            print('IOError, not saving results!')

    if MAKE_BASELINE:
        print '\nSaving baseline predictions!'
        # score, threshold, all_scores = cluster(clusterer, x_val, y_val, THRESH_SEARCH_RANGE, get_all_scores=True)
        # print 'Baseline score and threshold:',score,threshold
        if BASELINE_THRESH <= 0.0:
            print 'Clustering the baseline validation data to find optimal threshold...'
            score, threshold, all_scores = cluster(clusterer, x_val, y_val, THRESH_SEARCH_RANGE,
                                                   get_all_scores=True)
            print 'Baseline score and threshold:', score, threshold
            BASELINE_THRESH = threshold

        gold, predictions = test_pred.predict(test_x, test_mentions, threshold=BASELINE_THRESH, metric='cosine', build_test_comparison=True)
        # test_pred.save_predictions_mention_based(predictions, add_name='baseline')
        analyze_singleton_accuracy(gold, predictions)

    if MAKE_BASELINE_PCA:
        print '\nPerforming PCA to do PCA-Baseline predictions...'
        pca = PCA(n_components=HOURGLASS_LAYER_SIZE)
        pca.fit(x_train)
        new_val_x = pca.transform(x_val)
        new_test_x = pca.transform(test_x)

        print 'Clustering the new PCA-based validation data to find optimal threshold...'
        score, threshold, all_scores = cluster(clusterer, new_val_x, y_val, THRESH_SEARCH_RANGE, get_all_scores=True)
        print 'PCA-Baseline score and threshold:', score, threshold

        gold, predictions = test_pred.predict(new_test_x, test_mentions, threshold=threshold, metric='cosine')
        test_pred.save_predictions_mention_based(predictions, add_name='PCA-baseline')

# TODO: analyze the vectors  create!
def analyze_representations(r):
    pass

################### data initializiation ###################
def build_y_data(x, all_mentions, cluster_singletons=False, remove_train_singletons=False):
    """
    Turns cluster ids of the mentions into integer values.
    The cluster_singletons argument puts all singleton mentions into their own
        class, as opposed to having them all in separate classes.
    """
    singleton_cids = helpers.get_singletons_from_mlst(all_mentions)

    coref_class = 0 if cluster_singletons else -1
    y = []
    idxs_to_remove = set()
    y_dic = {}
    for i,m in enumerate(all_mentions):
        if not (m.coref_chain_id in singleton_cids and remove_train_singletons):
            mid = m.coref_chain_id
            if mid in singleton_cids and cluster_singletons:
                y.append(SINGLETON_CLASS)
            else:
                if mid not in y_dic:
                    coref_class += 1
                    y_dic[mid] = coref_class
                y.append(y_dic[mid])
        else:
            idxs_to_remove.add(i)

    newx = []
    for i in xrange(len(x)):
        if i not in idxs_to_remove:
            newx.append(x[i])

    assert set(y) == set(range(coref_class + 1)) # asserts that every class value between 0 and max is in there
    return np.array(newx), y

def get_lemma_predictor(data):
    cd = load(events_only=events_only, data_set='all')
    lpred = BaselineLemmaPredictor(cd.gold_mention_clusters,
                                   cd.get_all_tokens(topics=data),
                                   events_only=True,
                                   data_set='val' if data is helpers.VAL else 'test',
                                   topics=data)
    return lpred

def initialize_data_sets(events_only, test_set, split_into_topics, cluster_singletons=False, remove_train_singletons=False):
    x_train, y_train, train_mentions, train_pred = None, None, None, None
    x_val, y_val, val_mentions, val_pred = None, None, None, None

    cd = load(events_only=events_only, data_set='all')
    for data in (train_set, test_set,):
        pred = ClusteringPredictor(cd.gold_mention_clusters,
                                   cd.get_all_tokens(topics=data),
                                   events_only=events_only,
                                   data_set='val' if data is helpers.VAL else 'test',
                                   topics=data)

        mentions_by_topic = pred.get_mentions_by_topic(events_only=events_only, split_into_topics=split_into_topics)

        x, all_mentions = pred.extract_features(mentions_by_topic)

        if x_train is None:
            x_train, y_train = build_y_data(x, all_mentions, cluster_singletons=cluster_singletons, remove_train_singletons=remove_train_singletons)
            train_pred = pred
        else:
            x_val, y_val = build_y_data(x, all_mentions, cluster_singletons=PUT_VAL_SINGELTONS) # always false for validation data
            val_pred = pred

    assert len(x_train)==len(y_train) and len(x_val)==len(y_val)

    return train_pred, np.array(x_train), np.array(y_train), train_mentions, \
           val_pred, np.array(x_val), np.array(y_val), val_mentions


# this must be called on jc-gpu1 with THEANO_FLAGS=device=gpu1 python neural_cluster_model.py
if __name__ == '__main__':
    events_only = True
    train_set = set(helpers.TRAIN).difference(helpers.VAL)
    test_set = helpers.VAL
    split_into_topics = False

    train_pred, x_train, y_train, train_mentions, val_pred, x_val, y_val, val_mentions = \
        initialize_data_sets(events_only, test_set, split_into_topics,
                             cluster_singletons=PUT_SINGLETONS, remove_train_singletons=REMOVE_TRAIN_SINGLETONS)

    if LEMMA_PRED:
        lpred = get_lemma_predictor(helpers.VAL) # start with validation one
        test_comp = None
        best_score = 0
        best_delta = 0
        for delta in np.linspace(0,1,101):
            tmp, preds = lpred.predict(build_test_comparison=test_comp is None, delta=delta)
            test_comp = tmp if test_comp is None else test_comp
            score = bcubed(test_comp, preds)[2] # returns r,p,f1
            if score > best_score:
                best_score = score
                best_delta = delta
            print 'Delta %0.2f gets us %0.5f accuracy!'%(delta, score)

        exit(0)
        # probably can be optimized, but just need tfidf to be built

        initialize_data_sets(events_only, helpers.TEST, split_into_topics,
                             cluster_singletons=PUT_SINGLETONS, remove_train_singletons=REMOVE_TRAIN_SINGLETONS)

        ltestpred = get_lemma_predictor(helpers.TEST)
        gold, preds = ltestpred.predict(delta=best_delta)
        analyze_singleton_accuracy(gold, preds)
        ltestpred.save_predictions_mention_based(preds, 'HEAD_LEMMA_DELTA')
        exit(0)

    if SINGLETON_PREDICTOR:
        pred = SingletonPredictor(x_train, y_train, x_val, y_val)
        pred.validation_optimization()
        exit(0)

    if PCA_TESTS:
        test_pred, test_x, test_mentions = get_final_test_set(events_only)
        if OPTIMAL_THRESHOLD > 0:
            print '\nDoing PCA-%d with thresh %0.3f to build test set clusters!'%(HOURGLASS_LAYER_SIZE, OPTIMAL_THRESHOLD)
            pca = PCA(n_components=HOURGLASS_LAYER_SIZE)
            pca.fit(x_train)
            new_test_x = pca.transform(test_x)
            gold, predictions = test_pred.predict(new_test_x, test_mentions, threshold=OPTIMAL_THRESHOLD, metric='cosine')
            test_pred.save_predictions_mention_based(predictions, add_name='PCA-%d'%HOURGLASS_LAYER_SIZE)
        else:
            best, best_dim,best_thresh = (0,0,0), 0,0
            for dim in (200,300,400,500,750,1000,1500,):
                pca = PCA(n_components=dim)
                pca.fit(x_train)
                new_val_x = pca.transform(deepcopy(x_val))
                print 'Clustering the new PCA-based validation data to find optimal threshold...'
                score, threshold, all_scores = cluster(val_pred, new_val_x, y_val, THRESH_SEARCH_RANGE, get_all_scores=True)
                print 'PCA-%d score and threshold:'%dim, score, threshold

                if score[2] > best[2]:
                    best = score # score is (r,p,f1,)
                    best_dim = dim
                    best_thresh = threshold
            print best, best_thresh, 'PCA dimensionality=',best_dim
        exit(0)

    print 'Learning rate: %0.7f; Epochs %d'%(LEARNING_RATE,EPOCHS)

    if ANALYZE_RESULTS:
        print 'Analyzing results...'
        analyze_neural_results(RESULTS_FILE, x_train, y_train, x_val, y_val, val_pred,
                               get_test_set_predictions=GET_TEST_PREDICTIONS, oracle_topics=ORACLE_TOPICS)
    else:
        results, best_reps, rep_function = prepare_and_run_model(train_pred, val_pred,
                                                                 THRESH_RANGE,
                                                                 x_train, y_train,
                                                                 x_val, y_val,
                                                                 batchsize=BATCHSIZE,
                                                                 epochs=EPOCHS)

        np.save(RESULTS_FILE, np.array([results, best_reps, rep_function]))

