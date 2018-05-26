import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics.pairwise import euclidean_distances,cosine_distances,manhattan_distances

metrics = {'euclidean':euclidean_distances, 'cosine':cosine_distances, 'manhattan':manhattan_distances}

class AgglomerativeClusterer(object):

    def __init__(self, x_matrix, distance_metric='euclidean', train_data=None):
        if train_data is not None:
            normalizer = StandardScaler()
            normalizer.fit(train_data)
            self.x = normalizer.transform(x_matrix)
        else:
            self.x = x_matrix
        self.dist_matrix = metrics[distance_metric](self.x)
        self.sim_matrix = 1.0 - 2.0*(self.dist_matrix / np.max(self.dist_matrix))

    def cluster(self, threshold, linktype='single', preset_predictions=None, minimum_keeping_threshold=0):
        '''  '''
        orig_sim_matrix = self.sim_matrix
        sim_matrix = np.copy(self.sim_matrix)
        nitems = sim_matrix.shape[0]
        eliminated = np.array([], dtype=np.int32)

        # set diagonals to neg inf: can't merge something with itself
        sim_matrix[np.arange(nitems), np.arange(nitems)] = -np.Inf
        clusters = {i:np.array([i]) for i in xrange(nitems)}

        # set pre_pairings
        if preset_predictions is None:
            preset_cluster_pairings = []
        else:
            preset_cluster_pairings = predictions_to_cluster_pairings(preset_predictions)
                
                
        for step in xrange(nitems - 1):
            # to merge
            #print sim_matrix

            i, j = np.unravel_index(np.argmax(sim_matrix), sim_matrix.shape)
            sim = sim_matrix[i,j]
 
            if len(preset_cluster_pairings) > step: 
                i, j = preset_cluster_pairings[step] # this allows us to init with preds made by l-delta
                # mkt parameter to say that only keep the pairing if sim
                # is good enough, according to the parameter's value
                if sim < minimum_keeping_threshold:
                    continue
            else:
                if sim < threshold:
                    break
            
            smaller, larger = min(i,j), max(i,j)
            # print '%d: merging %d into %d with similarity %.4f' % (step, larger, smaller, sim)
            clusters[smaller] = np.concatenate((clusters[smaller], clusters[larger]))
            del clusters[larger]

            eliminated = np.concatenate((eliminated, [larger]))

            if linktype == 'single':
                newvector = np.maximum(sim_matrix[i,:], sim_matrix[j,:])
            else:
                # complete-link
                # print 'orig', orig_sim_matrix[clusters[smaller], :]
                newvector = np.min(orig_sim_matrix[clusters[smaller], :], axis = 0)

            sim_matrix[smaller, :] = newvector
            sim_matrix[:, smaller] = newvector

            sim_matrix[eliminated, :] = -np.Inf
            sim_matrix[:, eliminated] = -np.Inf

            sim_matrix[smaller, smaller] = -np.Inf

        return clusters.values()

def predictions_to_cluster_pairings(preds):
    # assume preds is an ordered list of chains
    # i.e., [1, 1, 2, 3, 1, 2]
    # with chains corresponding to idxs:
    # [0,1,4], [2,5], [3]
    # with pairs corresponding to:
    # (0,1), (0,4), (2,5)
    pairs = []
    got_chains = set()
    for i in xrange(len(preds)-1):
        for j in xrange(i+1, len(preds)):
            if preds[i] == preds[j] and not preds[i] in got_chains:
                pairs.append((i,j))
        got_chains.add(preds[i])
    return pairs

if __name__ == '__main__':
    x_data = [[1,2,3,5,6],[12,1,0,0,1],[2,3,11,0,-5],[0,1,0,0,1],[4,4,4,4,4]]
    c = AgglomerativeClusterer(x_data)
    print c.sim_matrix
    print c.cluster(threshold=0.35, linktype='single')

