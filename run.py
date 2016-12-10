import random
import numpy as np
import pickle
from data_utils import *

from paragraph2vec import *
from sgd import *

# Reset the random seed to make sure that everyone gets the same results

random.seed(314)
dataset = StanfordSentiment()
tokens = dataset.tokens()
nWords = len(tokens)#the number of unique words
nParagraphs = 75000#for IMDB dataset

# We are going to train 150-dimensional vectors for this assignment
dimVectors = 150

# Context size
C = 10

# Reset the random seed to make sure that everyone gets the same results #TODO???
random.seed(31415)
np.random.seed(9265)

wordVectors = np.concatenate(((np.random.rand(nParagraphs, dimVectors) - .5) / \
	dimVectors, np.random.rand(nWords, dimVectors) - .5), axis=0)
#TODO inizialization normal distribution centre =0 deviation = 1/10, 1/100

wordVectors0 = sgd(
    lambda vec, ds, words: word2vec_sgd_wrapper(skipgram, tokens, vec, ds, words, nParagraphs, 
    	negSamplingCostAndGradient), 
    wordVectors, dataset, C, nParagraphs, 0.05, 25, None, True)

