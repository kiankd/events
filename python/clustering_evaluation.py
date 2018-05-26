import bcubed as b3
from simple_helpers import update_list_dict

def bcubed(gold_lst, predicted_lst):
    """
    Takes gold, predicted.
    Returns recall, precision, f1score
    """
    gold = {i:{cluster} for i,cluster in enumerate(gold_lst)}
    pred = {i:{cluster} for i,cluster in enumerate(predicted_lst)}
    precision = b3.precision(pred, gold)
    recall = b3.recall(pred, gold)
    return recall, precision, b3.fscore(precision, recall)

def item_based_to_class_based(lst):
    d = {}
    for item,class_cluster in enumerate(lst):
        update_list_dict(d, class_cluster, item)
    for class_cluster in d:
        d[class_cluster] = set(d[class_cluster])
    return d

if __name__ == '__main__':
    gold = [1,1,1,1,1, 2,2, 3,3,3,3,3]
    pred = [1,1,1,1,1, 2,2, 2,2,2,2,2]
    print bcubed(gold, pred)
