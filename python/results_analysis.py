from matplotlib import pyplot as plt
import numpy as np

RESULTS_DIR = 'neural_results/'

def build_clustering_graph(save_name, all_scores):
    """
    :param save_name: string
    :param all_scores: Dictionary mapping thresholds to tuples of (recall, precision, f1)
    :return:
    """
    x = all_scores.keys()
    rpf1s = all_scores.values()
    recalls = [t[0] for t in rpf1s]
    precisions = [t[1] for t in rpf1s]
    f1s = [t[2] for t in rpf1s]

    f, axarr = plt.subplots(3, sharex=True)  # sharing x axis
    axarr[0].set_title('Agglomerative clustering results from threshold tuning')

    axarr[0].plot(x, recalls, 'r-')
    axarr[0].set_ylabel('B3 recall')

    axarr[1].plot(x, precisions, 'b-')
    axarr[1].set_ylabel('B3 precision')

    axarr[2].plot(x, f1s, 'g-')
    axarr[2].set_ylabel('B3 F1')
    axarr[2].set_xlabel('Threshold')

    f.savefig(RESULTS_DIR + save_name + 'thresh_eval.png')
    plt.clf()

def build_results_graphs(save_name, epochs, train_loss, val_acc):
    bound = 0.2

    fig = plt.figure()
    axis1 = fig.add_subplot(211)
    plt.plot(epochs, train_loss, 'r-')
    plt.ylabel('Training loss')
    plt.axis([-1,max(epochs)+1, min(train_loss)-bound, max(train_loss)+bound])

    axis2 = fig.add_subplot(212)
    plt.plot(epochs, val_acc, 'b-')
    plt.xlabel('Epoch')
    plt.ylabel('Validation accuracy')
    plt.axis([-1,max(epochs)+1, min(val_acc)-bound, max(val_acc)+bound])

    fig.savefig(RESULTS_DIR + save_name + '.png')
    plt.clf()

if __name__ == '__main__':
    fake_data = {i:[i, i**2, i**3] for i in map(lambda x: x/100., range(100))}
    build_clustering_graph('test_delete.png', fake_data)
