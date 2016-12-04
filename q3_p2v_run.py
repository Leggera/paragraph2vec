import random
import numpy as np
import pickle
from cs224d.data_p2v_utils import *
import matplotlib.pyplot as plt

from q3_paragraph2vec import *
from q3_sgd import *

# Reset the random seed to make sure that everyone gets the same results

dataset = StanfordSentiment()
tokens = dataset.tokens()
exit()
C = 5
word = dataset.getContext(C)
it = iter(word)

line_num = 0
for i in it:
    p_id, context = i
    print p_id
    print context
    line_num += 1
    if line_num == 500:
        break
exit()


'''
path = "/data/NLP/sensegram/word2vec_c/word2vec1/aclImdb/train"
line_num = 0
for filename in os.listdir(path + "/neg"):                
    with open(path + "/neg/" + filename, "r") as f:
        for line in f:
            print line
            print "PARAGRAPH", filename
            line_num += 1
            if line_num == 15:
                break'''
exit()
with open("/data/NLP/sensegram/word2vec_c/word2vec1/data_p2v.txt", "r") as f:
    line_num = 0
    for line in f:
        print line
        print "PARAGRAH"
        line_num += 1
        if line_num == 15:
            break
exit()
random.seed(314)
print 'a'
dataset = StanfordSentiment()
print 'b'
tokens = dataset.tokens()
print 'c'
nWords = len(tokens)#2771466#1913160188 - not unique
nParagraphs = 50#74895585#dataset.numSentences() #TODO
print nWords

# We are going to train 10-dimensional vectors for this assignment
dimVectors = 10#100 #TODO

# Context size
C = 5#10 #TODO
#word = dataset.getRandomContext(C)
word = dataset.getContext(C)#can input random(C) inside this function
it = iter(word)
# Reset the random seed to make sure that everyone gets the same results
random.seed(31415)
np.random.seed(9265)
print 'd'
wordVectors = np.concatenate(((np.random.rand(nParagraphs, dimVectors) - .5) / \
	dimVectors, np.random.rand(nWords, dimVectors)), axis=0)
print wordVectors.shape
print 'e'

wordVectors0 = sgd(
    lambda vec: word2vec_sgd_wrapper(skipgram, tokens, vec, dataset, C, it, 
    	negSamplingCostAndGradient), 
    wordVectors, 0.3, 40000, None, True, PRINT_EVERY=10)
print "sanity check: cost at convergence should be around or below 10"

# sum the input and output word vectors
print wordVectors0[:nWords,:].shape
print wordVectors0[nWords:,:].shape

'''wordVectors = (wordVectors0[:nWords,:] + wordVectors0[nWords:,:])

# Visualize the word vectors you trained
_, wordVectors0, _ = load_saved_params()
wordVectors = (wordVectors0[:nWords,:] + wordVectors0[nWords:,:])
visualizeWords = ["the", "a", "an", ",", ".", "?", "!", "``", "''", "--", 
	"good", "great", "cool", "brilliant", "wonderful", "well", "amazing",
	"worth", "sweet", "enjoyable", "boring", "bad", "waste", "dumb", 
	"annoying"]
visualizeIdx = [tokens[word] for word in visualizeWords]
visualizeVecs = wordVectors[visualizeIdx, :]
temp = (visualizeVecs - np.mean(visualizeVecs, axis=0))
covariance = 1.0 / len(visualizeIdx) * temp.T.dot(temp)
U,S,V = np.linalg.svd(covariance)
coord = temp.dot(U[:,0:2]) 

for i in xrange(len(visualizeWords)):
    plt.text(coord[i,0], coord[i,1], visualizeWords[i], 
    	bbox=dict(facecolor='green', alpha=0.1))
    
plt.xlim((np.min(coord[:,0]), np.max(coord[:,0])))
plt.ylim((np.min(coord[:,1]), np.max(coord[:,1])))

plt.savefig('q3_word_vectors.png')
plt.show()'''
