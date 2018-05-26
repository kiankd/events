import simple_helpers as helpers
import numpy as np
from itertools import combinations
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.linear_model import LogisticRegression,LogisticRegressionCV
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from scipy.sparse import hstack,csr_matrix,vstack


class PairwiseDocumentPredictor(object):
    def __init__(self, train_docs, test_docs):

        self.train_docs = {fname:helpers.list_to_string([t.word for t in train_docs[fname]]) for fname in train_docs}
        self.test_docs  = {fname:helpers.list_to_string([t.word for t in test_docs[fname]]) for fname in test_docs}
        self.train_pairs = [pair for pair in combinations(train_docs.keys(), 2)]
        self.test_pairs  = [pair for pair in combinations(test_docs.keys(), 2)]
        self.vectorizer  = TfidfVectorizer(stop_words=helpers.STOPS, min_df=1)

        train_words = ''
        for doc_str in self.train_docs.itervalues():
            train_words += doc_str + ' '
        self.vectorizer.fit([train_words])
        print 'Finished fitting tf-idf vectorizer.'

        self.model = None

    def extract_features(self, doc_dict, pairs):
        print 'Extracting pairwise document features for %d pairs...'%len(pairs)
        x = csr_matrix([])
        y = []
        for i,pair in enumerate(pairs):
            x = vstack([x,self.pairwise_doc_features(doc_dict, pair[0], pair[1])])
            y.append(1 if helpers.documents_in_same_topic(pair) else 0)
            if i % 10000 == 0:
                print '%0.2f%%...'%(float(i) / len(pairs))
        return x,y

    def pairwise_doc_features(self, ddict, d1, d2):
        d1_vec = self.vectorizer.transform([ddict[d1]])
        d2_vec = self.vectorizer.transform([ddict[d2]])
        similarity = cosine_similarity(d1_vec, d2_vec)[0]
        return hstack([d1_vec, d2_vec, csr_matrix([similarity])])

    def build_features(self):
        x,y = self.extract_features(self.train_docs, self.train_pairs)
        xhat, yhat = self.extract_features(self.test_docs, self.test_pairs)
        np.save(helpers.DOCUMENT_DATASET_FILE, np.array([x, y, xhat, yhat]))

    def load_data(self):
        data = np.load(helpers.DOCUMENT_DATASET_FILE)
        return data[0], data[1], data[2], data[3]

    def train_model(self, x, y, model=LogisticRegression()):
        self.model = model
        self.model.fit(x, y)

    def predict(self, xhat, yhat):
        pred = self.model.predict(xhat)
        print 'Precision, recall, F1:'
        for f in [precision_score, recall_score, f1_score]:
            print f(yhat, pred),
        print

    def run_model_tests(self, models=None):
        if not models:
            models = [LogisticRegression(), LogisticRegressionCV(), SVC(), LinearSVC(), RandomForestClassifier()]
        x, y, xhat, yhat = self.load_data()

        for model in models:
            print model
            print 'Training...'
            self.train_model(x, y, model=model)
            self.predict(xhat, yhat)
            print