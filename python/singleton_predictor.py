from sklearn.svm import LinearSVC, SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler

SINGLETON_CLASS = 0
NON_SINGLETON_CLASS = 1
CLASSIFIER = {
    'logreg':LogisticRegression,
    'dtree':DecisionTreeClassifier,
    'lsvm':LinearSVC,
    'svm':SVC,
    'forest':RandomForestClassifier,
}

def convert_y(y):
    if y is None:
        return y

    num_clust = 0
    for i in xrange(len(y)):
        if y[i] != SINGLETON_CLASS:
            num_clust += 1
            y[i] = NON_SINGLETON_CLASS
    print '%d cluster samples, %d singleton samples.'%(num_clust, len(y)-num_clust)
    return y

class SingletonPredictor(object):

    def __init__(self, train_x, train_y, val_x=None, val_y=None, test_x=None, test_y=None):
        self.train_x = train_x
        self.train_y = convert_y(train_y)
        self.val_x = val_x
        self.val_y = convert_y(val_y)
        self.test_x = test_x
        self.test_y = convert_y(test_y)

        # normalizer = StandardScaler()
        # self.train_x = normalizer.fit_transform(self.train_x)
        # if self.val_x is not None: self.val_x = normalizer.transform(self.val_x)
        # if self.test_x is not None: self.test_x = normalizer.transform(self.test_x)

    def validation_optimization(self):
        for model_constructor in CLASSIFIER.itervalues():
            model = model_constructor()
            model.fit(self.train_x, self.train_y)
            predictions = model.predict(self.val_x)
            print 'Using this model: %s,'%model.__class__,
            print 'we get the following results (R,P,F1):'
            print recall_score(self.val_y, predictions), \
                    precision_score(self.val_y, predictions), \
                    f1_score(self.val_y, predictions)
            print
