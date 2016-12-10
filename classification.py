from sklearn.neural_network import MLPClassifier
import glob
import os.path as op
import cPickle as pickle
import numpy as np
import random

from data_utils import *
from paragraph2vec import *
from sgd import *
'''with open("/data/class.npy", "w") as f:
        pickle.dump(X, f)'''

def load_saved_params():
    """ A helper function that loads previously saved parameters and resets iteration start """
    st = 0
    for f in glob.glob("./saved_params_*.npy"):
        iter = int(op.splitext(op.basename(f))[0].split("_")[2])
        if (iter > st):
            st = iter
            
    if st > 0:
        with open("./saved_params_%d.npy" % st, "r") as f:
            params = pickle.load(f)
            state = pickle.load(f)
        return params
    else:
        return None

oldx = load_saved_params()


random.seed(314)
dataset = StanfordSentiment()
tokens = dataset.tokens()
nWords = len(tokens)#the number of unique words
nParagraphs = 25000#for IMDB dataset

# We are going to train 150-dimensional vectors for this assignment
dimVectors = 150

# Context size
C = 10

# Reset the random seed to make sure that everyone gets the same results #TODO???
random.seed(31415)
np.random.seed(9265)

wordVectors = np.concatenate(((np.random.rand(nParagraphs, dimVectors) - .5) / \
	dimVectors, oldx[75000:]), axis=0)

wordVectors0 = sgd(
    lambda vec, ds, words: word2vec_sgd_wrapper(tokens, vec, ds, words, nParagraphs, 
    	negSamplingCostAndGradient, w2v = False), 
    wordVectors, dataset, C, nParagraphs, 0.05, 25, None, False, w2v = False)

with open("./vectors.npy", "w") as f:
        pickle.dump(wordVectors0, f)

x0 = oldx[:75000]

X = x0[:25000]


y = np.asarray([0] * 12500 + [1] * 12500)

clf = MLPClassifier(hidden_layer_sizes = (50, 2))

clf.fit(X, y)                         

p = clf.predict(wordVectors0[:25000])

print "precision"
print len(p[p == y])/25000.0

